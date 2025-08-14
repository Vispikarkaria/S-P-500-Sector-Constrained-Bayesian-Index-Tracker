# make_gpu_notebook.py
# ------------------------------------------------------------
# Generates a GPU-accelerated Jupyter Notebook that teaches
# a sector-constrained Bayesian S&P 500 index tracker.
# The notebook solves the tracking problem on GPU using PyTorch
# with projected gradient descent (PGD) and a capped-simplex
# projection (sum(w)=1, 0<=w<=cap).
# ------------------------------------------------------------

import nbformat as nbf
from textwrap import dedent
from datetime import date

nb = nbf.v4.new_notebook()
cells = []

# 1) Title & Overview
cells.append(nbf.v4.new_markdown_cell(dedent(f"""
# Sector-Constrained Bayesian Index Tracker — **GPU-Accelerated** Tutorial

**Last updated:** {date.today().isoformat()}

This notebook reproduces the S&P 500 sector-constrained Bayesian tracker, but the **optimization** is done with
**PyTorch on GPU** (if available). We replace the cvxpy QP with **Projected Gradient Descent (PGD)**, which:

- Minimizes: \\\\( \\frac1T \\|Rw - y\\|^2 + \\lambda \\|w\\|^2 + \\gamma \\|Aw - s^*\\|^2 \\\\)
- Subject to: \\\\( \\sum_i w_i = 1, \\; 0 \\le w_i \\le c \\\\)  (long-only and per-name cap)

Projection uses a **capped simplex** operator implemented with a fast **bisection water-filling** routine in torch.

> For research and education only. Not investment advice.
""")))

# 2) Requirements
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 0. Requirements

Run the cell below (only once) to install dependencies.

- `torch` (GPU if available; please pick the correct CUDA/ROCm build for your machine)
- `pandas`, `numpy`, `yfinance`, `requests`, `lxml`, `tqdm`, `matplotlib`
- `scikit-learn` (for BayesianRidge hyperparameter λ)
- `statsmodels` (for simple AR(1) forecast)

> **Install Torch**: visit https://pytorch.org/get-started/locally/ for the exact command for your CUDA/ROCm.  
> Example for CUDA 12.1 wheels:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
# If running on a fresh environment, uncomment and run once (adjust torch line per your system).
# !pip install --quiet pandas numpy yfinance scikit-learn lxml requests tqdm matplotlib statsmodels
# Example (CUDA 12.1):  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU-only fallback is fine too:  pip install torch torchvision torchaudio
""")))

# 3) Imports & Config
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 1. Imports and configuration
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
import os, json, warnings, math
import datetime as dt
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests, yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import BayesianRidge
warnings.filterwarnings("ignore")

# --- Reproducibility-ish ---
np.random.seed(42)
torch.manual_seed(42)

# --- Device selection ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# --- User config ---
LOOKBACK_YEARS = 3
OOS_MONTHS     = 6
FREQUENCY      = "1d"
INDEX_SYMBOL   = "^GSPC"
UNIVERSE_CAP   = 300
MAX_NAMES      = 40
PER_NAME_CAP   = 0.08
LONG_ONLY      = True
SECTOR_PENALTY = 5.0
L2_FLOOR       = 1e-6
OUTPUT_DIR     = "./notebook_outputs_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_SECTORS = [
    "Information Technology",
    "Health Care",
    "Financials",
    # "Communication Services","Consumer Discretionary","Consumer Staples",
    # "Energy","Industrials","Materials","Real Estate","Utilities"
]

today = dt.date.today()
END_DATE = today
START_DATE = END_DATE - dt.timedelta(days=int(365*LOOKBACK_YEARS + 365))
print(f"Training window: {START_DATE} → {END_DATE}")
""")))

# 4) Get S&P list
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 2. Get the S&P 500 constituents (Wikipedia) and normalize tickers
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def get_sp500_constituents() -> pd.DataFrame:
    html = requests.get(WIKI_URL, timeout=30).text
    df = pd.read_html(html)[0].rename(columns={
        "Symbol":"Ticker", "Security":"Name",
        "GICS Sector":"Sector","GICS Sub-Industry":"SubIndustry"
    })
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False).str.upper().str.strip()
    return df[["Ticker","Name","Sector","SubIndustry"]]

sp500_all = get_sp500_constituents()
sp500 = sp500_all[sp500_all["Sector"].isin(ALLOWED_SECTORS)].copy()
if sp500.empty:
    raise ValueError("No tickers in selected sectors; adjust ALLOWED_SECTORS.")
print(f"Sectors: {sorted(sp500['Sector'].unique())} | candidates: {len(sp500)}")
""")))

# 5) Market caps & sector target
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 3. Market caps → sector target weights
We approximate the S&P sector mix by summing live market caps (proxy) and normalizing.
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def fetch_market_caps(tickers: List[str]) -> pd.Series:
    out = {}
    for t in tqdm(tickers, desc="Market caps"):
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
            pass
    return pd.Series(out, name="MarketCap", dtype="float64")

caps_ref = fetch_market_caps(sp500_all.Ticker.tolist())
ref_df = sp500_all.merge(caps_ref.rename("MarketCap"),
                         left_on="Ticker", right_index=True, how="left").dropna(subset=["MarketCap"])
sector_caps_ref = ref_df.groupby("Sector")["MarketCap"].sum()
sector_target = (sector_caps_ref / sector_caps_ref.sum()).sort_values(ascending=False)

caps_sel = fetch_market_caps(sp500.Ticker.tolist())
sp500 = sp500.merge(caps_sel.rename("MarketCap"),
                    left_on="Ticker", right_index=True, how="left").dropna(subset=["MarketCap"])
sp500 = sp500.sort_values("MarketCap", ascending=False).head(UNIVERSE_CAP).reset_index(drop=True)
print("Universe after cap:", len(sp500))
""")))

# 6) Prices & returns
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 4. Download prices and compute log-returns; split IS/OOS
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def fetch_prices(tickers: List[str], start, end, interval="1d") -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    px = df["Close"] if "Close" in df else df
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px.dropna(how="all")

tickers = sp500.Ticker.tolist()
px_ast = fetch_prices(tickers, START_DATE, END_DATE, FREQUENCY)
px_idx = fetch_prices([INDEX_SYMBOL], START_DATE, END_DATE, FREQUENCY)

common = px_ast.index.intersection(px_idx.index)
px_ast = px_ast.loc[common].dropna(axis=1, how="any")
px_idx = px_idx.loc[common]
tickers = [t for t in tickers if t in px_ast.columns]
sp500 = sp500[sp500.Ticker.isin(tickers)].reset_index(drop=True)

def logret(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna()

R = logret(px_ast)                    # T x N
y = logret(px_idx).iloc[:,0]          # T vector
oos_days = int(30.42*OOS_MONTHS)
if len(R) < oos_days + 252:
    raise ValueError("Not enough history for requested OOS window.")
R_is, R_oos = R.iloc[:-oos_days,:], R.iloc[-oos_days:,:]
y_is, y_oos = y.iloc[:-oos_days],     y.iloc[-oos_days:]
R_is.shape, R_oos.shape
""")))

# 7) Sector matrix and targets
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 5. Sector aggregation matrix (A) and aligned target (s*)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
sectors = sorted(sp500.Sector.unique())
row = {s:i for i,s in enumerate(sectors)}
A = np.zeros((len(sectors), len(tickers)))
for j,t in enumerate(tickers):
    s = sp500.loc[sp500.Ticker==t, "Sector"].values[0]
    A[row[s], j] = 1.0

s_tgt = np.array([sector_target.get(s,0.0) for s in sectors], dtype=float)
if s_tgt.sum() > 0:
    s_tgt = s_tgt / s_tgt.sum()

A.shape, s_tgt[:5]
""")))

# 8) Lambda via BayesianRidge
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 6. Estimate ridge strength λ via Bayesian evidence (CPU, tiny cost)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def estimate_lambda_bayes(R: pd.DataFrame, y: pd.Series, l2_floor: float=1e-6) -> float:
    X = R.values; Y = y.values
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Ys = (Y - Y.mean()) / (Y.std() + 1e-12)
    br = BayesianRidge(fit_intercept=True, compute_score=True)
    br.fit(Xs, Ys)
    lam = float(br.lambda_ / max(br.alpha_, 1e-12))
    return max(lam, l2_floor)

ridge_lambda = estimate_lambda_bayes(R_is, y_is, L2_FLOOR)
print("lambda (BayesianRidge):", ridge_lambda)
""")))

# 9) Greedy selection (CPU)
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 7. Optional: Greedy forward selection to cap number of names (K)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
def greedy_select(R: pd.DataFrame, y: pd.Series, lam: float, K: Optional[int]) -> List[str]:
    if K is None or K >= R.shape[1]:
        return list(R.columns)
    chosen = []
    remaining = set(R.columns)
    corr = R.corrwith(y).abs().sort_values(ascending=False)
    first = corr.index[0]; chosen.append(first); remaining.remove(first)
    for _ in range(1, K):
        Xc = R[chosen].values; yc = y.values
        XtX = Xc.T @ Xc
        w = np.linalg.solve(XtX + lam*np.eye(len(chosen)), Xc.T @ yc)
        resid = yc - Xc @ w
        best,score=None,-1.0
        for t in remaining:
            x = R[t].values
            num = np.dot(resid-resid.mean(), x-x.mean())
            den = np.sqrt(((resid-resid.mean())**2).sum()*((x-x.mean())**2).sum()) + 1e-18
            c = abs(num)/den
            if c>score: score,best=c,t
        chosen.append(best); remaining.remove(best)
    return chosen

selected = greedy_select(R_is, y_is, ridge_lambda, MAX_NAMES)
R_is = R_is[selected]; R_oos = R_oos[selected]
sel_mask = sp500.Ticker.isin(selected)
sp500_sel = sp500[sel_mask].reset_index(drop=True)
tickers_sel = list(R_is.columns)

# rebuild A and target for selection
sectors_sel = sorted(sp500_sel.Sector.unique())
row_sel = {s:i for i,s in enumerate(sectors_sel)}
A_sel = np.zeros((len(sectors_sel), len(tickers_sel)))
for j,t in enumerate(tickers_sel):
    s = sp500_sel.loc[sp500_sel.Ticker==t,"Sector"].values[0]
    A_sel[row_sel[s], j] = 1.0
s_tgt_sel = np.array([sector_target.get(s,0.0) for s in sectors_sel])
if s_tgt_sel.sum() > 0:
    s_tgt_sel = s_tgt_sel / s_tgt_sel.sum()
R_is.shape, R_oos.shape
""")))

# 10) GPU PGD Solver
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 8. **GPU solver**: Projected Gradient Descent (PGD) on PyTorch

Objective  
\\[
\\min_w \\; \\frac{1}{T}\\lVert Rw - y\\rVert^2 + \\lambda\\lVert w\\rVert^2 + \\gamma\\lVert Aw - s^*\\rVert^2
\\quad \\text{s.t.} \\quad \\sum_i w_i = 1,\\; 0\\le w_i \\le c.
\\]

We:
1. Take a gradient step on GPU;  
2. **Project** onto the **capped simplex** \\(\\{{w\\mid w\\ge0, w\\le c, \\sum w=1}\\}\\) using a **bisection water-filling** operator.
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
# ---- Torch helpers ----
def np_to_torch(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def project_capped_simplex(v: torch.Tensor, cap: float, sum_to: float = 1.0,
                           iters: int = 50) -> torch.Tensor:
    """
    Project v onto {w | 0 <= w <= cap, sum(w) = sum_to} using bisection.
    We find tau s.t. sum( clamp(v - tau, 0, cap) ) = sum_to.
    """
    # If already feasible (within tiny tolerance), quick path
    w = torch.clamp(v, 0.0, cap)
    s = w.sum()
    if abs(float(s.item()) - sum_to) < 1e-6:
        return w

    # Bisection bounds for tau
    lo = (v - cap).min() - 1.0
    hi = v.max() + 1.0
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        w = torch.clamp(v - mid, 0.0, cap)
        s = w.sum()
        if s > sum_to:
            lo = mid
        else:
            hi = mid
    w = torch.clamp(v - hi, 0.0, cap)
    # Normalize tiny drift
    if w.sum().item() != 0:
        w = w * (sum_to / w.sum())
        w = torch.clamp(w, 0.0, cap)
        # Small second renorm if caps caused rounding
        if abs(w.sum().item() - sum_to) > 1e-6:
            w = w * (sum_to / max(w.sum().item(), 1e-12))
            w = torch.clamp(w, 0.0, cap)
    return w

def solve_pgd_gpu(Rm: np.ndarray, y: np.ndarray, A: np.ndarray, s_tgt: np.ndarray,
                  lam: float, gamma: float, cap: Optional[float], long_only: bool = True,
                  steps: int = 5000, lr: float = 0.05, tol: float = 1e-8,
                  verbose: bool = True) -> np.ndarray:
    """
    GPU-accelerated PGD. Returns optimal weights as numpy array.
    If no GPU is available, runs on CPU (still using torch).
    """
    T, N = Rm.shape
    R_t = np_to_torch(Rm, device)
    y_t = np_to_torch(y.reshape(-1,1), device)
    A_t = np_to_torch(A, device)
    s_t = np_to_torch(s_tgt.reshape(-1,1), device)

    # init feasible w: uniform over N (respect caps if provided)
    if cap is None:
        cap_val = 1.0
    else:
        cap_val = float(cap)
    w = torch.full((N,1), fill_value=1.0/N, device=device, dtype=torch.float32)
    if long_only:
        w = project_capped_simplex(w.squeeze(), cap=cap_val, sum_to=1.0).unsqueeze(1)
    w.requires_grad_(True)

    prev_obj = float("inf")
    opt = torch.optim.SGD([w], lr=lr)  # simple & deterministic; Adam also works.

    for i in range(steps):
        opt.zero_grad()
        resid = R_t @ w - y_t                # (T x 1)
        track = (resid.pow(2).sum()) / T
        ridge = lam * (w.pow(2).sum())
        sector = torch.tensor(0.0, device=device)
        if gamma > 0 and s_t.abs().sum() > 0:
            sector = gamma * ((A_t @ w - s_t).pow(2).sum())
        obj = track + ridge + sector
        obj.backward()
        opt.step()

        # Projection to constraints
        with torch.no_grad():
            w_vec = w.squeeze()
            if long_only:
                w_vec = project_capped_simplex(w_vec, cap=cap_val, sum_to=1.0)
            else:
                # If shorting allowed, just enforce sum=1 (no cap); do affine projection
                shift = (w_vec.sum() - 1.0) / N
                w_vec = w_vec - shift
            w[:] = w_vec.unsqueeze(1)

        # Stopping rule
        if i % 100 == 0 or i == steps-1:
            cur = obj.item()
            if verbose and i % 500 == 0:
                print(f"iter {i:5d}  obj={cur:.6g}")
            if abs(prev_obj - cur) < tol:
                if verbose:
                    print(f"Converged at iter {i}, obj={cur:.6g}")
                break
            prev_obj = cur

    return w.detach().cpu().squeeze().numpy()

# Move data to GPU solver
R_is_np = R_is.values.astype(np.float32)
R_oos_np = R_oos.values.astype(np.float32)
y_is_np  = y_is.values.astype(np.float32)
y_oos_np = y_oos.values.astype(np.float32)
A_np     = A_sel.astype(np.float32)
s_np     = s_tgt_sel.astype(np.float32)

weights = solve_pgd_gpu(
    Rm=R_is_np, y=y_is_np,
    A=A_np, s_tgt=s_np,
    lam=ridge_lambda, gamma=SECTOR_PENALTY,
    cap=PER_NAME_CAP if LONG_ONLY else None,
    long_only=LONG_ONLY,
    steps=4000, lr=0.05, tol=1e-9, verbose=True
)
weights[:10], weights.sum()
""")))

# 11) Metrics & plots
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 9. Evaluate IS/OOS tracking and plot cumulative growth
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
w_ser = pd.Series(weights, index=tickers_sel, name="Weight").sort_values(ascending=False)

def eval_tracking(R: pd.DataFrame, y: pd.Series, w: pd.Series):
    rp = (R[w.index] @ w.values).rename("Portfolio")
    te = rp - y
    out = {
        "TE_ann_vol": float(te.std()*np.sqrt(252)),
        "TE_ann_mean": float(te.mean()*252),
        "R2": float(np.corrcoef(rp, y)[0,1]**2) if len(rp)>2 else np.nan,
        "Beta": float(np.cov(rp, y, ddof=1)[0,1]/(rp.var(ddof=1)+1e-12)),
        "Names": int((w.values>1e-6).sum())
    }
    return rp, out

rp_is, met_is   = eval_tracking(R_is, y_is, w_ser)
rp_oos, met_oos = eval_tracking(R_oos, y_oos, w_ser)

print("In-sample:", json.dumps(met_is, indent=2))
print("OOS:", json.dumps(met_oos, indent=2))

# Plots (single-plot, no explicit colors)
cum_is = pd.concat([rp_is.rename("Portfolio"), y_is.rename("Index")], axis=1).cumsum().apply(np.exp)
plt.figure(figsize=(9,4)); plt.plot(cum_is.index, cum_is["Portfolio"], label="Portfolio"); plt.plot(cum_is.index, cum_is["Index"], label="Index")
plt.title("In-sample cumulative growth"); plt.ylabel("Growth of $1"); plt.legend(); plt.tight_layout(); plt.show()

cum_oos = pd.concat([rp_oos.rename("Portfolio"), y_oos.rename("Index")], axis=1).cumsum().apply(np.exp)
plt.figure(figsize=(9,4)); plt.plot(cum_oos.index, cum_oos["Portfolio"], label="Portfolio"); plt.plot(cum_oos.index, cum_oos["Index"], label="Index")
plt.title("Out-of-sample (~6 months) cumulative growth"); plt.ylabel("Growth of $1"); plt.legend(); plt.tight_layout(); plt.show()
""")))

# 12) Sector mix plot & save artifacts
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 10. Sector mix and artifact exports
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
# Sector mix (portfolio vs target)
A_sel_df = pd.DataFrame(A_sel, index=sectors_sel, columns=tickers_sel)
port_sector = A_sel_df @ w_ser
tgt_sector = pd.Series([sector_target.get(s,0.0) for s in sectors_sel], index=sectors_sel)
if tgt_sector.sum()>0: tgt_sector = tgt_sector/tgt_sector.sum()

plt.figure(figsize=(10,4)); plt.bar(port_sector.index, port_sector.values); plt.xticks(rotation=45, ha="right")
plt.title("Portfolio sector weights"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4)); plt.bar(tgt_sector.index, tgt_sector.values); plt.xticks(rotation=45, ha="right")
plt.title("Target sector weights"); plt.tight_layout(); plt.show()

# Save weights & metrics
weights_df = (w_ser.to_frame()
              .merge(sp500_sel[["Ticker","Name","Sector","MarketCap"]],
                     left_index=True, right_on="Ticker")
              .sort_values("Weight", ascending=False))
weights_df.to_csv(os.path.join(OUTPUT_DIR, "weights_gpu.csv"), index=False)

metrics = {
    "IS": met_is, "OOS": met_oos,
    "lambda": float(ridge_lambda), "gamma": float(SECTOR_PENALTY),
    "long_only": LONG_ONLY, "per_name_cap": PER_NAME_CAP,
    "lookback_years": LOOKBACK_YEARS, "oos_months": OOS_MONTHS,
    "universe_cap": UNIVERSE_CAP, "K": MAX_NAMES,
    "sectors": ALLOWED_SECTORS, "device": str(device)
}
with open(os.path.join(OUTPUT_DIR, "metrics_gpu.json"), "w") as f:
    json.dump(metrics, f, indent=2)

weights_df.head(10)
""")))

# 13) Forecast
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 11. Simple 6-month forecast (AR(1) on index, mapped by beta)
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
from statsmodels.tsa.ar_model import AutoReg
beta_is = met_is["Beta"]
ar = AutoReg(y_is.values, lags=1, old_names=False).fit()
steps = len(y_oos)
y_fore = ar.predict(start=len(y_is), end=len(y_is)+steps-1)
p_fore = beta_is * y_fore
df_fore = pd.DataFrame({"Index_fore":y_fore, "Portfolio_fore":p_fore})
cum_fore = df_fore.cumsum().apply(np.exp)
plt.figure(figsize=(9,4)); plt.plot(cum_fore.index, cum_fore["Portfolio_fore"], label="Portfolio (forecast)")
plt.plot(cum_fore.index, cum_fore["Index_fore"], label="Index (forecast)")
plt.title("Forecast (~6 months) cumulative growth"); plt.ylabel("Growth of $1"); plt.legend(); plt.tight_layout(); plt.show()
df_fore.head()
""")))

# 14) Notes & next steps
cells.append(nbf.v4.new_markdown_cell(dedent("""
## 12. Notes

- The GPU path accelerates the **optimization** stage (matrix ops + PGD).  
- Data loading and BayesianRidge hyperparameter estimation remain CPU-bound (small cost).
- You can switch the optimizer to **Adam** (often converges faster), tune `lr`, and increase `steps` for tighter solutions.
- For **shorting**, set `LONG_ONLY=False` and modify projection to only enforce `sum(w)=1` (already handled above).

**Next steps:** turnover constraints, transaction costs, rolling rebalances, Black-Litterman priors, or exact cardinality via MIP (CPU).
""")))

# Build & save notebook
nb["cells"] = cells
notebook_path = "sp500_sector_tracker_tutorial_gpu.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote:", notebook_path)
