# S\&P 500 Sector-Constrained Bayesian Index Tracker (Streamlit App)

An interactive web app that lets you **choose GICS sectors**, then builds a **long-only, market-cap-aware, Bayesian index tracker** that targets the **S\&P 500**. It downloads the current S\&P 500 constituents and sectors, fetches prices, learns **regularized weights** with a **sector mix penalty**, evaluates **6-month out-of-sample** tracking, and produces a simple **6-month forecast**.

> For research and education only. This is **not** investment advice.

---

## Table of contents

* [What this app does](#what-this-app-does)
* [Key features](#key-features)
* [How it works (high level)](#how-it-works-high-level)
* [Installation](#installation)
* [Quick start](#quick-start)
* [UI walkthrough](#ui-walkthrough)
* [Outputs and files](#outputs-and-files)
* [Interpreting metrics](#interpreting-metrics)
* [Configuration tips](#configuration-tips)
* [Method details](#method-details)
* [Data sources and caveats](#data-sources-and-caveats)
* [Performance tips](#performance-tips)
* [Troubleshooting](#troubleshooting)
* [Extending the app](#extending-the-app)
* [Security and privacy](#security-and-privacy)
* [License](#license)

---

## What this app does

* Pulls the **current S\&P 500** list and **GICS sectors** from Wikipedia.
* Lets you **select sectors** (e.g., Technology, Health Care, Financials).
* Downloads **adjusted prices** for **^GSPC** (index) and selected constituents.
* Learns a **Bayesian ridge** tracker with **sector-mix control** and optional **cardinality** (limit the number of names).
* Evaluates **next \~6 months** out-of-sample (OOS) tracking.
* Produces a simple **6-month forecast** (AR(1) on index returns mapped by portfolio beta).
* Exports **weights**, **metrics**, and plots.

---

## Key features

* **Sector-aware tracking:** penalizes deviation from a target sector mix (either full S\&P or your selected sectors).
* **Bayesian regularization:** automatic shrinkage via evidence-maximizing **BayesianRidge**, reducing overfit.
* **Cardinality control:** greedy forward selection to keep a compact portfolio (e.g., 20–60 names).
* **Convex optimization:** solved as a QP with **cvxpy** (long-only, sum of weights = 1, optional per-name cap).
* **Live data:** current constituents from Wikipedia, prices from Yahoo Finance via **yfinance**.
* **Transparent outputs:** CSV and JSON exports, cumulative growth plots, sector mix comparison.

---

## How it works (high level)

1. **Universe & sectors**
   Scrape S\&P 500 constituents from Wikipedia, normalize tickers (e.g., `BRK.B → BRK-B`), and keep only the sectors you select.

2. **Sector targets**
   Estimate sector weights by summing live market caps (approximate; float-adjustment is ignored) and normalizing. Optionally use **full S\&P** sector mix as the target even if you invest in a subset.

3. **Returns and split**
   Compute **daily log-returns** for stocks and index over a lookback window (default 3 years). Hold out the **last \~6 months** for OOS evaluation.

4. **Hyperparameter (λ) via Bayesian evidence**
   Fit **BayesianRidge** on standardized in-sample returns to estimate an effective **L2 penalty** λ (MAP interpretation).

5. **Greedy forward selection (optional)**
   Iteratively add the stock that best explains the current residual vs the index to cap the number of names (**K**).

6. **Optimization**
   Solve a QP that minimizes tracking error + λ ||w||² + γ ||A w − s\*||² subject to long-only, sum of weights = 1, and optional per-name cap.

7. **Evaluation and forecast**
   Report OOS tracking metrics and plot cumulative returns. Fit an AR(1) to the index and map via portfolio beta to produce a **toy 6-month projection**.

---

## Installation

```bash
# 1) Create a fresh environment (recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install streamlit pandas numpy yfinance cvxpy osqp scikit-learn lxml requests tqdm matplotlib plotly statsmodels
```

> Notes
>
> * `cvxpy` will use **OSQP** by default for QPs; we install `osqp`.
> * On some platforms, cvxpy may require build tools; see cvxpy docs if install fails.

---

## Quick start

1. Save the app file as `app.py` (use the code you already generated).
2. Run:

```bash
streamlit run app.py
```

3. In the sidebar, choose your **sectors**, **lookback**, **OOS window**, **universe size**, **K**, **per-name cap**, and **sector penalty γ**.
4. Click **Run tracker**.

---

## UI walkthrough

* **Sectors to include**: GICS sectors you want in the investable set.
* **Lookback window (years)**: history used to fit the model (e.g., 3–5 years).
* **Out-of-sample window (months)**: next period (e.g., 6 months) used to evaluate tracking.
* **Universe size**: keep top-N by market cap within selected sectors before selection; speeds things up.
* **Target number of names (K)**: greedy selection cap; balances tracking error vs simplicity.
* **Per-name cap**: maximum weight per stock (e.g., 8%).
* **Sector penalty γ**: higher values pull portfolio’s sector mix closer to target.
* **Long-only**: if checked, weights ≥ 0 with sum = 1.
* **Use full S\&P sector mix as target**: if enabled, the sector target uses the full index, not only your chosen sectors.

---

## Outputs and files

* **Weights table**: ticker, name, sector, market cap, and optimized weight.
* **Download weights CSV**: portfolio weights for your record.
* **Tracking metrics**: in-sample and OOS **R²**, **tracking-error volatility (annualized)**, **beta**, **names**, λ and γ.
* **Sector Mix plot**: compares **portfolio** vs **target** sector weights.
* **Cumulative growth plots**: in-sample and OOS cumulative return growth of \$1 for portfolio vs index.
* **Download metrics JSON**: full config and metrics for reproducibility.

Files are generated in your working directory if you add manual saves; the app offers direct CSV/JSON downloads.

---

## Interpreting metrics

* **R²**: proportion of variance in index returns explained by the portfolio. Higher is better.
* **TE ann vol**: annualized standard deviation of daily **(portfolio − index)** returns. Lower is better.
* **Beta**: sensitivity of the portfolio to the index. Near 1 means similar risk.
* **Names**: number of holdings with non-negligible weight.
* **λ (lambda)**: ridge strength estimated via Bayesian evidence; higher is stronger shrinkage.
* **γ (gamma)**: sector penalty. Larger values enforce sector alignment more strictly (possibly at the cost of tracking error).

---

## Configuration tips

* **Tighter tracking**: increase **K**, reduce **λ**, and/or reduce **γ** if sector mix is too restrictive.
* **Simpler portfolio**: decrease **K**, increase **λ**, increase **per-name cap** (but watch concentration).
* **Closer sector match**: increase **γ**.
* **Data sparseness**: increase **lookback** to stabilize estimates.
* **If OOS window fails**: you may need more history; reduce OOS months or increase lookback.

---

## Method details

* **Regularization via BayesianRidge**
  We standardize in-sample returns and fit `BayesianRidge` to learn the effective ridge ratio. The MAP solution corresponds to L2-penalized least squares, which stabilizes weights and reduces overfitting.

* **Cardinality (K names)**
  Exact cardinality constraints make the problem mixed-integer (slow). Instead, we use a **greedy forward selection**: iteratively add the stock most correlated with the current **residual** relative to the index, then refit until K names are selected.

* **Sector penalty**
  Let **A** be the sector aggregation matrix (rows=sectors, cols=stocks) and **s\*** the target sector vector. We penalize **‖A w − s\*‖²** to keep the portfolio’s sector mix aligned with the target while respecting your sector availability.

* **Constraints**

  * Sum of weights = 1
  * Long-only (optional): w ≥ 0
  * Per-name cap (optional): w ≤ cap
    Solved with **cvxpy** using **OSQP**; falls back to **SCS** if needed.

* **OOS evaluation and forecast**
  OOS returns are computed by applying in-sample weights to OOS asset returns. Forecast is intentionally simple (AR(1) on index, scaled by in-sample portfolio beta); you can replace with Black-Litterman, VAR, or bootstrap.

---

## Data sources and caveats

* **Constituents**: Wikipedia “List of S\&P 500 companies” (may change and occasionally break scraping).
* **Sectors**: **GICS** sectors from the same table.
* **Prices**: Yahoo Finance via **yfinance** adjusted closes.
* **Market caps**: from `fast_info.market_cap` or `info["marketCap"]` as a proxy (float-adjustment not guaranteed).

Limitations:

* Market caps are an approximation and can lag or be missing for some tickers.
* Survivorship bias is not addressed; we use the **current** list of constituents.
* Corporate actions, rebalances, and symbol changes can cause gaps.
* Yahoo throttling may occur; try smaller universes or rerun later.

---

## Performance tips

* Reduce **universe size** (e.g., top 200–300 by cap) before K-selection.
* Increase **lookback** to stabilize covariance structure.
* Run on Python 3.10+ with up-to-date `numpy`/`scikit-learn`/`cvxpy`.
* If OSQP is slow, ensure it is properly installed; try `prob.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7)` (already set).
* Use Streamlit’s caching (already used for scraping and downloads).

---

## Troubleshooting

* **cvxpy install fails**: install build tools, then `pip install cvxpy osqp`. On Apple Silicon, ensure a recent Python and `pip`.
* **“QP failed” / no solution**: relax per-name cap, reduce γ (sector penalty), increase λ (ridge), or increase K.
* **“Not enough history for OOS”**: increase lookback years or shorten OOS months.
* **Missing market caps**: some tickers return `None`; the app drops those from the universe.
* **Plot shows gaps**: ensure enough overlapping dates after dropping NAs; reduce universe size.

---

## Extending the app

* **Exact cardinality**: switch to a MISOCP/MIP solver with binary variables (slower).
* **Shorting**: uncheck long-only, add an L1/Gross exposure constraint, and guard against extreme leverage.
* **Different priors**: replace ridge with elastic-net or hierarchical priors.
* **Better forecast**: plug in Black-Litterman, block bootstrap, or a full macro/sector model.
* **Rebalance simulation**: add rolling windows and transaction cost modeling.
* **Persistence**: add a simple backend (SQLite) to store runs and compare scenarios.

---

## Security and privacy

* The app fetches **public data** and does not store credentials.
* All computations run **locally** unless you deploy the app to a remote server.
* Downloads (weights, metrics) are generated on demand for your session.

---

## License

Specify your preferred license (e.g., MIT) here. Include attribution if you redistribute or publish derivative work.

---

### A final note

Real-world index tracking must consider **float-adjusted caps, liquidity, trading costs, taxes, corporate actions, reconstitutions, and compliance**. Treat this app as a **transparent sandbox** for exploring sector-aware, shrinkage-regularized index tracking.
