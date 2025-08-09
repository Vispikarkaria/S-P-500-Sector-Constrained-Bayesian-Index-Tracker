# app.py
# ============================================================
# S&P 500 Sector-Constrained Bayesian Index Tracker (Web App)
# ============================================================
# - Select sectors (GICS) to include
# - Pull S&P 500 constituents + sectors (Wikipedia)
# - Pull prices for ^GSPC and selected stocks (Yahoo Finance)
# - Estimate sector weights, fit Bayesian ridge tracker w/ sector penalty
# - Evaluate next ~6 months OOS and plot results
# - Simple 6-month AR(1) forecast for index and mapped portfolio
#
# Not investment advice. For research/education only.
# ============================================================

import os
import json
import time
import warnings
import datetime as dt
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import cvxpy as cp
from sklearn.linear_model import BayesianRidge
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------- UI Styling ----------
st.set_page_config(page_title="S&P 500 Sector Tracker", layout="wide")
st.markdown(
    """
    <style>
    .small {font-size: 0.85rem; color: #555;}
    .metric {font-size: 1.2rem;}
    .stPlotlyChart {background: white;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False, ttl=60*60)
def get_sp500_constituents() -> pd.DataFrame:
    """Scrape S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, timeout=30).text
    df = pd.read_html(html)[0]
    df = df.rename(columns={
        "Symbol":"Ticker", "Security":"Name",
        "GICS Sector":"Sector", "GICS Sub-Industry":"SubIndustry"
    })
    df["Ticker"] = df["Ticker"].str.replace(".","-", regex=False).str.upper().str.strip()
    return df[["Ticker","Name","Sector","SubIndustry"]]

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_market_caps(tickers: List[str]) -> pd.Series:
    """Fetch (approx) market caps via yfinance."""
    out = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            cap = None
            try:
                cap = tk.fast_info.get("market_cap", None)
            except Exception:
                pass
            if cap is None:
                info = tk.info
                cap = info.get("marketCap", None)
            if cap and np.isfinite(cap):
                out[t] = float(cap)
        except Exception:
            continue
    return pd.Series(out, name="MarketCap", dtype="float64")

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_prices(tickers: List[str], start, end, interval="1d") -> pd.DataFrame:
    """Batch download adjusted close prices with yfinance."""
    df = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    px = df["Close"] if "Close" in df else df
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px.dropna(how="all")

def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna()

def estimate_lambda_bayes(R: pd.DataFrame, y: pd.Series, l2_floor: float = 1e-6) -> float:
    """Evidence-maximizing BayesianRidge ⇒ effective L2 strength."""
    X = R.values; Y = y.values
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Ys = (Y - Y.mean()) / (Y.std() + 1e-12)
    br = BayesianRidge(fit_intercept=True, compute_score=True)
    br.fit(Xs, Ys)
    lam = float(br.lambda_ / max(br.alpha_, 1e-12))
    return max(lam, l2_floor)

def greedy_select(R: pd.DataFrame, y: pd.Series, lam: float, K: int | None) -> List[str]:
    """Greedy forward selection by residual correlation to limit names."""
    if (K is None) or (K >= R.shape[1]): return list(R.columns)
    chosen = []
    remain = set(R.columns)
    # start with max |corr|
    corr = R.corrwith(y).abs().sort_values(ascending=False)
    first = corr.index[0]
    chosen.append(first); remain.remove(first)
    for _ in range(1, K):
        Xc = R[chosen].values; yc = y.values
        XtX = Xc.T @ Xc
        w = np.linalg.solve(XtX + lam*np.eye(len(chosen)), Xc.T @ yc)
        resid = yc - Xc @ w
        best, score = None, -1.0
        for t in remain:
            x = R[t].values
            num = np.dot(resid-resid.mean(), x-x.mean())
            den = np.sqrt(((resid-resid.mean())**2).sum() * ((x-x.mean())**2).sum()) + 1e-18
            c = abs(num)/den
            if c > score: score, best = c, t
        chosen.append(best); remain.remove(best)
    return chosen

def solve_qp(R: np.ndarray, y: np.ndarray, A: np.ndarray, s_tgt: np.ndarray,
             lam: float, gamma: float, long_only: bool, cap: float | None) -> np.ndarray:
    """Constrained Bayesian ridge with sector penalty."""
    T,N = R.shape
    w = cp.Variable(N)
    loss = (1.0/T)*cp.sum_squares(R@w - y) + lam*cp.sum_squares(w)
    if gamma>0 and s_tgt.sum()>0:
        loss += gamma*cp.sum_squares(A@w - s_tgt)
    cons = [cp.sum(w)==1]
    if long_only:
        cons += [w >= 0]
        if cap is not None:
            cons += [w <= cap]
    prob = cp.Problem(cp.Minimize(loss), cons)
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        raise RuntimeError("QP failed; relax constraints/gamma/lam or reduce K.")
    return np.array(w.value).ravel()

def eval_tracking(R: pd.DataFrame, y: pd.Series, w: pd.Series) -> tuple[pd.Series, Dict[str,float]]:
    rp = (R[w.index] @ w.values).rename("Portfolio")
    te = rp - y
    out = {
        "TE_ann_vol": float(te.std()*np.sqrt(252)),
        "TE_ann_mean": float(te.mean()*252),
        "R2": float(np.corrcoef(rp, y)[0,1]**2) if len(rp)>2 else np.nan,
        "Beta": float(np.cov(rp, y, ddof=1)[0,1] / (rp.var(ddof=1)+1e-12)),
        "Names": int((w.values>1e-6).sum())
    }
    return rp, out

# ---------- Sidebar Controls ----------
st.sidebar.header("Configuration")
ALL_SECTORS = [
    "Information Technology","Health Care","Financials","Communication Services",
    "Consumer Discretionary","Consumer Staples","Energy","Industrials",
    "Materials","Real Estate","Utilities"
]
sel_sectors = st.sidebar.multiselect(
    "Sectors to include",
    options=ALL_SECTORS,
    default=["Information Technology","Health Care","Financials"]
)
lookback_years = st.sidebar.slider("Lookback window (years)", 2, 10, 3)
oos_months = st.sidebar.slider("Out-of-sample window (months)", 3, 12, 6)
universe_cap = st.sidebar.slider("Universe size (top by market cap)", 50, 500, 300, step=25)
max_names = st.sidebar.slider("Target number of names (K)", 10, 100, 40, step=5)
per_name_cap = st.sidebar.slider("Per-name cap", 0.02, 0.15, 0.08, step=0.01)
sector_penalty = st.sidebar.slider("Sector penalty γ", 0.0, 20.0, 5.0, step=0.5)
long_only = st.sidebar.checkbox("Long-only", value=True)
use_full_sector_target = st.sidebar.checkbox("Use full S&P sector mix as target", value=True)
run_button = st.sidebar.button("Run tracker", type="primary")

st.title("S&P 500 Sector-Constrained Bayesian Index Tracker")
st.caption("Pick sectors, then run. The app pulls live data and builds a shrinkage-regularized tracker with sector control.")

# ---------- Main Logic ----------
if run_button:
    try:
        start_all = dt.date.today() - dt.timedelta(days=int(365*lookback_years + 365))
        end_all   = dt.date.today()
        idx_symbol = "^GSPC"

        if len(sel_sectors) == 0:
            st.warning("Please select at least one sector.")
            st.stop()

        with st.status("Loading S&P 500 constituents...", expanded=False):
            sp500 = get_sp500_constituents()
            st.write(f"Fetched {len(sp500)} constituents.")
        # Sector targets reference set: either full S&P or selected-only
        base_for_sector_target = sp500.copy() if use_full_sector_target else sp500[sp500["Sector"].isin(sel_sectors)].copy()

        # Filter to selected sectors for investable universe
        universe0 = sp500[sp500["Sector"].isin(sel_sectors)].copy()
        if universe0.empty:
            st.error("No tickers found in the chosen sectors.")
            st.stop()

        with st.status("Fetching market caps...", expanded=False):
            caps_all = fetch_market_caps(base_for_sector_target.Ticker.tolist())
            caps_sel = fetch_market_caps(universe0.Ticker.tolist())
        base_for_sector_target = base_for_sector_target.merge(caps_all.rename("MarketCap"), left_on="Ticker", right_index=True, how="left").dropna(subset=["MarketCap"])
        universe0 = universe0.merge(caps_sel.rename("MarketCap"), left_on="Ticker", right_index=True, how="left").dropna(subset=["MarketCap"])

        # Sector target weights (normalized)
        sector_caps_ref = base_for_sector_target.groupby("Sector")["MarketCap"].sum()
        sector_target = (sector_caps_ref / sector_caps_ref.sum()).sort_values(ascending=False)

        # Cap universe size
        universe = universe0.sort_values("MarketCap", ascending=False).head(universe_cap).reset_index(drop=True)

        with st.status("Downloading prices (this can take a minute)...", expanded=False):
            px_idx_all = fetch_prices([idx_symbol], start_all, end_all, "1d")
            px_ast_all = fetch_prices(universe.Ticker.tolist(), start_all, end_all, "1d")

        common = px_idx_all.index.intersection(px_ast_all.index)
        px_idx_all = px_idx_all.loc[common]
        px_ast_all = px_ast_all.loc[common].dropna(axis=1, how="any")

        # Align universe to available prices
        universe = universe[universe.Ticker.isin(px_ast_all.columns)].reset_index(drop=True)

        # Returns
        R_all = log_returns(px_ast_all)
        y_all = log_returns(px_idx_all).iloc[:,0]
        common2 = R_all.index.intersection(y_all.index)
        R_all = R_all.loc[common2]
        y_all = y_all.loc[common2]

        # Split IS/OOS
        oos_days = int(30.42*oos_months)
        if len(R_all) < oos_days + 252:
            st.error("Not enough history for requested OOS window. Increase lookback or reduce OOS months.")
            st.stop()
        R_is = R_all.iloc[:-oos_days, :]
        y_is = y_all.iloc[:-oos_days]
        R_oos = R_all.iloc[-oos_days:, :]
        y_oos = y_all.iloc[-oos_days:]

        # Sector matrix for IS set
        sectors = sorted(universe.Sector.unique())
        S = len(sectors)
        row = {s:i for i,s in enumerate(sectors)}
        # Build A (S x N) aligned to R_is columns after selection
        # First, estimate lambda and do greedy selection
        lam = estimate_lambda_bayes(R_is, y_is, 1e-6)
        sel = greedy_select(R_is, y_is, lam, max_names)
        R_is = R_is[sel]; R_oos = R_oos[sel]

        # Rebuild sector mapping for selected
        universe_sel = universe[universe.Ticker.isin(sel)].reset_index(drop=True)
        sectors_sel = sorted(universe_sel.Sector.unique())
        row_sel = {s:i for i,s in enumerate(sectors_sel)}
        N = len(sel); S_sel = len(sectors_sel)
        A_sel = np.zeros((S_sel, N))
        for j,t in enumerate(sel):
            s = universe_sel.loc[universe_sel.Ticker==t, "Sector"].values[0]
            A_sel[row_sel[s], j] = 1.0

        # Build sector target vector aligned to sectors_sel
        s_tgt = np.array([sector_target.get(s, 0.0) for s in sectors_sel], dtype=float)
        if s_tgt.sum() > 0: s_tgt = s_tgt / s_tgt.sum()

        # Solve QP for weights
        w = solve_qp(R_is.values, y_is.values, A_sel, s_tgt, lam, sector_penalty, long_only, per_name_cap)
        weights = pd.Series(w, index=R_is.columns, name="Weight").sort_values(ascending=False)

        # Evaluate IS & OOS
        rp_is, met_is = eval_tracking(R_is, y_is, weights)
        rp_oos, met_oos = eval_tracking(R_oos, y_oos, weights)

        # ---------- Outputs ----------
        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("Weights")
            weights_df = (weights.to_frame()
                          .merge(universe_sel[["Ticker","Name","Sector","MarketCap"]], left_index=True, right_on="Ticker")
                          .sort_values("Weight", ascending=False))
            st.dataframe(weights_df.style.format({"Weight":"{:.4%}","MarketCap":"{:.0f}"}), use_container_width=True, height=420)
            st.download_button("Download weights CSV", weights_df.to_csv(index=False).encode(), "weights.csv", "text/csv")

        with right:
            st.subheader("Tracking Metrics")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"**In-sample R²:** {met_is['R2']:.3f}")
                st.markdown(f"**In-sample TE vol (ann):** {met_is['TE_ann_vol']:.2%}")
                st.markdown(f"**In-sample Beta:** {met_is['Beta']:.3f}")
            with c2:
                st.markdown(f"**OOS R²:** {met_oos['R2']:.3f}")
                st.markdown(f"**OOS TE vol (ann):** {met_oos['TE_ann_vol']:.2%}")
                st.markdown(f"**OOS Beta:** {met_oos['Beta']:.3f}")
            st.markdown(f"**Names:** {met_oos['Names']} | **λ:** {lam:.3g} | **γ:** {sector_penalty}")

            # Sector bars (portfolio vs target)
            st.subheader("Sector Mix (Portfolio vs Target)")
            port_sector_weights = pd.Series(A_sel @ weights.loc[sel].values, index=sectors_sel, name="Portfolio")
            tgt_sector_weights = pd.Series([sector_target.get(s,0.0) for s in sectors_sel], index=sectors_sel, name="Target")
            if tgt_sector_weights.sum() > 0:
                tgt_sector_weights = tgt_sector_weights / tgt_sector_weights.sum()

            sec_df = pd.concat([port_sector_weights, tgt_sector_weights], axis=1).fillna(0)
            sec_df = sec_df.sort_values("Target", ascending=False)
            fig_sec = go.Figure()
            fig_sec.add_bar(x=sec_df.index, y=sec_df["Target"], name="Target")
            fig_sec.add_bar(x=sec_df.index, y=sec_df["Portfolio"], name="Portfolio")
            fig_sec.update_layout(barmode="group", height=360, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_sec, use_container_width=True)

        # Cumulative growth plots
        st.subheader("Cumulative Growth")
        cum_is = pd.concat([rp_is.rename("Portfolio"), y_is.rename("Index")], axis=1).cumsum().apply(np.exp)
        fig_is = px.line(cum_is, title="In-sample cumulative growth", height=360)
        fig_is.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_is, use_container_width=True)

        cum_oos = pd.concat([rp_oos.rename("Portfolio"), y_oos.rename("Index")], axis=1).cumsum().apply(np.exp)
        fig_oos = px.line(cum_oos, title="Out-of-sample (next ~6 months) cumulative growth", height=360)
        fig_oos.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_oos, use_container_width=True)

        # Simple 6M forecast via AR(1) on index, mapped by beta
        from statsmodels.tsa.ar_model import AutoReg
        beta_is = met_is["Beta"]
        ar = AutoReg(y_is.values, lags=1, old_names=False).fit()
        steps = len(y_oos)  # forecast horizon ~ same as OOS window
        y_fore = ar.predict(start=len(y_is), end=len(y_is)+steps-1)
        p_fore = beta_is * y_fore
        df_fore = pd.DataFrame({"Index_fore":y_fore, "Portfolio_fore":p_fore})
        cum_fore = df_fore.cumsum().apply(np.exp)
        fig_fore = px.line(cum_fore.rename(columns={"Index_fore":"Index","Portfolio_fore":"Portfolio"}),
                           title="Forecast (next ~6 months) cumulative growth", height=360)
        fig_fore.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_fore, use_container_width=True)

        # Downloads
        metrics_full = {
            "IS": met_is,
            "OOS": met_oos,
            "lambda": lam,
            "gamma": sector_penalty,
            "long_only": long_only,
            "per_name_cap": per_name_cap,
            "lookback_years": lookback_years,
            "oos_months": oos_months,
            "universe_cap": universe_cap,
            "K": max_names,
            "sectors": sel_sectors
        }
        st.download_button("Download metrics JSON", json.dumps(metrics_full, indent=2).encode(), "metrics.json", "application/json")

        st.caption("Done. Tip: adjust γ to pull the portfolio sector mix closer to the S&P target; increase K to reduce tracking error; increase λ for more shrinkage.")
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.exception(e)
else:
    st.info("Select sectors and parameters in the sidebar, then click **Run tracker**.")
