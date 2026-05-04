"""
Part 2 — Swaption Calibration (DD and SABR)
============================================
Calibrates Displaced-Diffusion and SABR volatility models to a
swaption market surface, including a synthetic 30×30 smile.

  1. Displaced-Diffusion calibration  (σ, β per slice)
  2. SABR calibration  (α, ρ, ν per slice; β_CEV = 0.75 fixed)
  3. Vol profiles and smile fit plots

"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import brentq, minimize, differential_evolution
from scipy.interpolate import interp1d
from scipy.stats import norm
import openpyxl

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import Part 1 results
# ---------------------------------------------------------------------------
from part1_bootstrapping_swap_curves import (
    tenor_to_years,
    years_to_label,
    load_market_data,
    df_interp,
    build_payment_schedule,
    bootstrap_ois,
    bootstrap_term_sofr,
    T_ois as T_ois_arr,
    DF_ois as DF_ois_arr,
    fwd_t as fwd_t_p1,
    fwd_r as fwd_r_p1,
)

print("Part 1 bootstrapping codebase imported successfully.")
print("  OIS curve : {} pillars, T in [0, {:.0f}]".format(
    len(T_ois_arr) - 1, T_ois_arr[-1]))
print("  Term SOFR : {} quarterly forward rates\n".format(len(fwd_t_p1)))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FILEPATH   = "Swap and Swaption Markets.xlsx"
BETA_SABR  = 0.75   # Fixed CEV exponent for SABR (P2)

# Strike column offsets from ATM in the Swaption sheet (11 columns)
STRIKES_BPS = np.array([-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200])
BPS_OFFSETS = STRIKES_BPS * 1e-4

# Tenors available in the swaption market grid (used for 30×30 smile)
TENORS_AVAIL = [1.0, 2.0, 3.0, 5.0, 10.0]

# Display labels for parameter tables
EXPIRIES  = ["1Y", "5Y", "10Y", "30Y"]
TENORS_U  = ["1Y", "2Y", "3Y", "5Y", "10Y", "30Y"]


# ---------------------------------------------------------------------------
# Discount curve wrappers
# ---------------------------------------------------------------------------

def make_discount_callable(T_arr, D_arr):
    """
    Wrap a bootstrap output (T_arr, D_arr) into a callable D(t).
    Uses linear interpolation inside the pillar range and flat-forward
    extrapolation beyond the last pillar.
    """
    inner  = interp1d(T_arr, D_arr, kind="linear", fill_value="extrapolate")
    T_max  = T_arr[-1]
    D_max  = float(inner(T_max))
    D_max1 = float(inner(max(T_max - 1.0, 0.0)))
    r_long = -np.log(max(D_max / D_max1, 1e-12))

    def D(t: float) -> float:
        if t <= 0:
            return 1.0
        if t <= T_max:
            return max(float(inner(t)), 1e-12)
        return max(D_max * np.exp(-r_long * (t - T_max)), 1e-12)

    return D


# Build OIS callable directly from Part 1 results
print("[OIS]       Using Part 1 bootstrap results …")
D_ois = make_discount_callable(T_ois_arr, DF_ois_arr)

# Re-bootstrap Term SOFR and build a callable
print("[Term SOFR] Bootstrapping via Part 1 bootstrap_term_sofr() …")
_, _, fwd_t_p2, fwd_r_p2 = bootstrap_term_sofr(
    load_market_data(FILEPATH)[2], T_ois_arr, DF_ois_arr
)
alpha_q   = 0.25
q_grid_p2 = np.concatenate([[0.0], fwd_t_p2 + alpha_q])
d_ts_vals = [1.0]
for fr in fwd_r_p2:
    d_ts_vals.append(d_ts_vals[-1] / (1.0 + fr * alpha_q))
T_ts_arr = np.array(q_grid_p2)
D_ts_arr = np.array(d_ts_vals)
D_ts = make_discount_callable(T_ts_arr, D_ts_arr)

print("\n  Check points (Part 1 curves reused):")
print("  {:>4}  {:>10}  {:>10}".format("T", "D_OIS", "D_TS"))
print("  " + "-" * 28)
for t in [1, 5, 10, 20, 30, 50]:
    print("  {:>4}  {:>10.6f}  {:>10.6f}".format(t, D_ois(t), D_ts(t)))


# ---------------------------------------------------------------------------
# Forward swap rate and annuity
# ---------------------------------------------------------------------------

def annuity(T_exp: float, T_ten: float, D_ois_fn, freq: float = 1.0) -> float:
    """OIS-discounted fixed-leg annuity factor A(T_exp, T_exp + T_ten)."""
    pts = np.arange(T_exp + freq, T_exp + T_ten + 1e-9, freq)
    return float(sum(D_ois_fn(t) * freq for t in pts))


def fwd_swap_rate(T_exp: float, T_ten: float, D_ois_fn, D_ts_fn) -> float:
    """
    Multi-curve par forward swap rate.
    F = (D_TS(T_exp) - D_TS(T_exp + T_ten)) / A_OIS
    """
    A = annuity(T_exp, T_ten, D_ois_fn)
    return (D_ts_fn(T_exp) - D_ts_fn(T_exp + T_ten)) / A


# ---------------------------------------------------------------------------
# Synthetic 30×30 smile construction
# ---------------------------------------------------------------------------

def build_synthetic_smile(raw_vols: dict, T_ten_target: float,
                           base_T_exp: float = 10.0) -> list:
    """
    Construct a smile for (base_T_exp, T_ten_target) by:
      1. Linearly extrapolating the ATM vol across tenors at base_T_exp.
      2. Borrowing the smile shape (skew = vol - ATM) from the longest
         available tenor slice to avoid negative vols in the wings.

    Parameters
    ----------
    raw_vols      : dict  — {(T_exp, T_ten): [vol_decimal, …, 11 strikes]}
    T_ten_target  : float — target tenor (e.g. 30.0 for 30×30)
    base_T_exp    : float — expiry row to use for extrapolation (default 10Y)

    Returns
    -------
    list of 11 synthetic vols in % (matching the strike grid)
    """
    atm_by_tenor = [raw_vols[(base_T_exp, Tn)][5] for Tn in TENORS_AVAIL]
    fn_atm       = interp1d(TENORS_AVAIL, atm_by_tenor,
                            kind="linear", fill_value="extrapolate")
    atm_target   = max(float(fn_atm(T_ten_target)), 0.01)

    vols_base = raw_vols[(base_T_exp, max(TENORS_AVAIL))]
    atm_base  = vols_base[5]
    skew      = [v - atm_base for v in vols_base]
    return [max(atm_target + sk, 0.01) for sk in skew]


def build_swaption_slices(swaption_rows, raw_vols_decimal):
    """
    Assemble slice dicts (one per expiry × tenor cell) and append
    the synthetic 30×30 slice.

    Each slice dict contains:
      label, exp_str, ten_str, T_exp, T_ten, F, A, strikes, vols_pct
    """
    slices = []
    for (exp_str, ten_str, vols_pct) in swaption_rows:
        T_exp = tenor_to_years(exp_str)
        T_ten = tenor_to_years(ten_str)
        F     = fwd_swap_rate(T_exp, T_ten, D_ois, D_ts)
        A     = annuity(T_exp, T_ten, D_ois)
        slices.append({
            "label":    "{}x{}".format(exp_str, ten_str),
            "exp_str":  exp_str,
            "ten_str":  ten_str,
            "T_exp":    T_exp,
            "T_ten":    T_ten,
            "F":        F,
            "A":        A,
            "strikes":  F + BPS_OFFSETS,
            "vols_pct": vols_pct,
        })

    # Synthetic 30×30 slice
    F_30 = fwd_swap_rate(30.0, 30.0, D_ois, D_ts)
    A_30 = annuity(30.0, 30.0, D_ois)
    vols_30_pct = [
        v * 100.0
        for v in build_synthetic_smile(raw_vols_decimal, 30.0)
    ]
    slices.append({
        "label":     "30Yx30Y",
        "exp_str":   "30Y",
        "ten_str":   "30Y",
        "T_exp":     30.0,
        "T_ten":     30.0,
        "F":         F_30,
        "A":         A_30,
        "strikes":   F_30 + BPS_OFFSETS,
        "vols_pct":  vols_30_pct,
        "synthetic": True,
    })
    return slices


# ---------------------------------------------------------------------------
# Option pricing models
# ---------------------------------------------------------------------------

def black_price(F, K, T, sigma, A=1.0, is_payer=True):
    """
    Black-76 swaption price.
    V_payer = A * [F*N(d1) - K*N(d2)]
    """
    if sigma < 1e-8 or K <= 0:
        return (max(F - K, 0.0) if is_payer else max(K - F, 0.0)) * A
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_payer:
        return A * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return A * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black_iv(price, F, K, T, A=1.0, is_payer=True):
    """Invert Black-76 for lognormal implied vol via Brent's method."""
    intrinsic = (max(F - K, 0.0) if is_payer else max(K - F, 0.0)) * A
    if price <= intrinsic + 1e-12:
        return np.nan
    f = lambda s: black_price(F, K, T, s, A, is_payer) - price
    if f(20.0) < 0:
        return np.nan
    if f(1e-6) > 0:
        return 1e-6
    try:
        return brentq(f, 1e-6, 20.0, xtol=1e-10)
    except Exception:
        return np.nan


def dd_price(F, K, T, sigma, beta, A=1.0, is_payer=True):
    """
    Displaced-Diffusion swaption price (Neo & Tee 2018, §3.2).
    SDE: dF_t = σ [β F_t + (1-β) F_0] dW_t
    Equivalent to Black-76 on displaced variables:
      F' = F/β,  K' = K + (1-β)/β · F,  σ' = β·σ
    """
    if beta <= 0:
        return (max(F - K, 0.0) if is_payer else max(K - F, 0.0)) * A
    F_p   = F / beta
    K_p   = K + (1.0 - beta) / beta * F
    sig_e = beta * sigma
    if F_p <= 0 or K_p <= 0:
        return (max(F - K, 0.0) if is_payer else max(K - F, 0.0)) * A
    return black_price(F_p, K_p, T, sig_e, A, is_payer)


def dd_implied_lognormal_vol(F, K, T, sigma, beta, A, is_payer=True):
    """Convert a DD price to Black-76 lognormal implied vol."""
    if K <= 0 or beta <= 0:
        return np.nan
    return black_iv(dd_price(F, K, T, sigma, beta, A, is_payer), F, K, T, A, is_payer)


def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """
    Hagan et al. (2002) SABR lognormal implied vol approximation.
    SDE: dF_t = α_t F_t^β dW_t,  dα_t = ν α_t dZ_t,  <dW,dZ> = ρ dt
    """
    eps = 1e-7
    if F <= 0 or K <= 0 or T <= 0:
        return 0.001
    if abs(F - K) < eps:
        # ATM approximation
        FK_1mb      = F ** (2.0 * (1.0 - beta))
        FK_half_1mb = F ** (1.0 - beta)
        ex = (
            (1.0 - beta) ** 2 * alpha ** 2 / (24.0 * FK_1mb)
            + rho * beta * nu * alpha / (4.0 * FK_half_1mb)
            + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
        )
        return max(alpha / F ** (1.0 - beta) * (1.0 + ex * T), 1e-4)

    log_fk      = np.log(F / K)
    FK_half_1mb = (F * K) ** ((1.0 - beta) / 2.0)
    FK_1mb      = (F * K) ** (1.0 - beta)
    D    = 1.0 + (1.0 - beta) ** 2 / 24.0 * log_fk ** 2 + (1.0 - beta) ** 4 / 1920.0 * log_fk ** 4
    z    = (nu / alpha) * FK_half_1mb * log_fk
    carg = (np.sqrt(max(1.0 - 2.0 * rho * z + z ** 2, 0.0)) + z - rho) / (1.0 - rho)
    if carg <= 0:
        return 0.001
    chi  = np.log(carg)
    zchi = z / chi if abs(chi) > eps else 1.0
    ex   = (
        (1.0 - beta) ** 2 * alpha ** 2 / (24.0 * FK_1mb)
        + rho * beta * nu * alpha / (4.0 * FK_half_1mb)
        + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
    )
    return max(alpha / (FK_half_1mb * D) * zchi * (1.0 + ex * T), 1e-4)


# ---------------------------------------------------------------------------
# P1 — Displaced-Diffusion calibration
# ---------------------------------------------------------------------------

def calibrate_dd_single(mkt_vols_pct, strikes, F, T, A):
    """
    Calibrate (σ, β) for one (expiry, tenor) swaption slice.

    Objective: price-space RMSE (OTM convention).
    Optimiser: differential_evolution (global) → L-BFGS-B (polish).

    Returns
    -------
    sigma : float
    beta  : float
    rmse  : float — per-contract price RMSE normalised by annuity
    """
    mkt_prices = []
    for K, v_pct in zip(strikes, mkt_vols_pct):
        if v_pct is None or (isinstance(v_pct, float) and np.isnan(v_pct)) or K <= 0:
            continue
        is_p = (K >= F)
        mkt_prices.append((K, black_price(F, K, T, v_pct / 100.0, A, is_p), is_p))
    if not mkt_prices:
        return 0.20, 0.5, np.nan

    def objective(params):
        sig, b = params
        if sig < 1e-4 or b <= 0:
            return 1e9
        return sum(
            ((dd_price(F, K, T, sig, b, A, ip) - pm) / A) ** 2
            for K, pm, ip in mkt_prices
        )

    BOUNDS_DD = [(0.001, 5.0), (1e-4, 1.0)]
    rg = differential_evolution(
        objective, bounds=BOUNDS_DD,
        seed=42, maxiter=2000, popsize=15, tol=1e-10,
        mutation=(0.5, 1.5), recombination=0.9,
    )
    rl = minimize(
        objective, rg.x, method="L-BFGS-B", bounds=BOUNDS_DD,
        options={"maxiter": 10_000, "ftol": 1e-14, "gtol": 1e-10},
    )
    sig, b = np.clip(rl.x, [bd[0] for bd in BOUNDS_DD], [bd[1] for bd in BOUNDS_DD])
    rmse = np.sqrt(rl.fun / len(mkt_prices))
    return float(sig), float(b), float(rmse)


def calibrate_dd_grid(swaption_slices):
    """
    Calibrate DD across the full swaption grid.

    Returns
    -------
    sigma_table : dict {(exp_str, ten_str): σ}
    beta_table  : dict {(exp_str, ten_str): β}
    """
    sigma_table, beta_table = {}, {}
    print("  {:>12} {:>7} {:>8} {:>8}  RMSE".format("Slice", "F (%)", "σ", "β"))
    print("  " + "-" * 48)
    for sl in swaption_slices:
        sig, b, rmse = calibrate_dd_single(
            sl["vols_pct"], sl["strikes"], sl["F"], sl["T_exp"], sl["A"]
        )
        key = (sl["exp_str"], sl["ten_str"])
        sigma_table[key] = sig
        beta_table[key]  = b
        sl["dd_params"]  = (sig, b)
        rmse_str = "{:.6f}".format(rmse) if np.isfinite(rmse) else "  n/a  "
        print("  {:>12}  {:>6.3f}  {:>8.4f}  {:>8.4f}  {}".format(
            sl["label"], sl["F"] * 100, sig, b, rmse_str
        ))
    return sigma_table, beta_table


# ---------------------------------------------------------------------------
# P2 — SABR calibration
# ---------------------------------------------------------------------------

def calibrate_sabr_single(mkt_vols_pct, strikes, F, T, beta=BETA_SABR):
    """
    Calibrate (α, ρ, ν) for one slice with β fixed.

    Objective: vol-space RMSE.
    Optimiser: differential_evolution (global) → L-BFGS-B (polish).

    Returns
    -------
    alpha    : float
    rho      : float
    nu       : float
    rmse_bps : float — vol RMSE in basis points
    """
    valid = [
        (K, v / 100.0)
        for K, v in zip(strikes, mkt_vols_pct)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and K > 0
    ]
    if not valid:
        return 0.01, 0.0, 0.3, np.nan

    ks  = np.array([x[0] for x in valid])
    mvs = np.array([x[1] for x in valid])
    atm_vol = mvs[int(np.argmin(np.abs(ks - F)))]
    alpha0  = atm_vol * F ** (1.0 - beta)

    def objective(params):
        a, rho, nu = params
        if a <= 0 or nu <= 0 or abs(rho) >= 0.999:
            return 1e9
        return sum(
            (sabr_vol(F, K, T, a, beta, rho, nu) - mv) ** 2
            for K, mv in zip(ks, mvs)
        )

    BOUNDS_SABR = [(1e-4, 1.0), (-0.998, 0.998), (1e-4, 5.0)]
    rg = differential_evolution(
        objective, bounds=BOUNDS_SABR,
        seed=42, maxiter=3000, popsize=20, tol=1e-12,
        mutation=(0.5, 1.5), recombination=0.9,
    )
    rl = minimize(
        objective, x0=rg.x, method="L-BFGS-B", bounds=BOUNDS_SABR,
        options={"maxiter": 20_000, "ftol": 1e-15, "gtol": 1e-10},
    )
    a, rho, nu = np.clip(
        rl.x,
        [b[0] for b in BOUNDS_SABR],
        [b[1] for b in BOUNDS_SABR],
    )
    rmse_bps = np.sqrt(rl.fun / len(valid)) * 1e4
    return float(a), float(rho), float(nu), float(rmse_bps)


def calibrate_sabr_grid(swaption_slices, beta=BETA_SABR):
    """
    Calibrate SABR across the full swaption grid.

    Returns
    -------
    alpha_table : dict {(exp_str, ten_str): α}
    rho_table   : dict {(exp_str, ten_str): ρ}
    nu_table    : dict {(exp_str, ten_str): ν}
    """
    alpha_table, rho_table, nu_table = {}, {}, {}
    print("  {:>12} {:>7} {:>8} {:>8} {:>8}  RMSE(bp)".format(
        "Slice", "F (%)", "α", "ρ", "ν"))
    print("  " + "-" * 58)
    for sl in swaption_slices:
        a, rho, nu, rmse = calibrate_sabr_single(
            sl["vols_pct"], sl["strikes"], sl["F"], sl["T_exp"], beta
        )
        key = (sl["exp_str"], sl["ten_str"])
        alpha_table[key]   = a
        rho_table[key]     = rho
        nu_table[key]      = nu
        sl["sabr_params"]  = (a, beta, rho, nu)
        atm_chk = sabr_vol(sl["F"], sl["F"], sl["T_exp"], a, beta, rho, nu) * 100
        mkt_str = "{:.2f}%".format(sl["vols_pct"][5]) if sl["vols_pct"][5] is not None else "synth"
        print("  {:>12}  {:>6.3f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>6.2f}   "
              "[ATM: mdl={:.2f}% mkt={}]".format(
                  sl["label"], sl["F"] * 100, a, rho, nu, rmse, atm_chk, mkt_str
              ))
    return alpha_table, rho_table, nu_table


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def make_param_df(table, exp_keys=EXPIRIES, ten_keys=TENORS_U):
    """Pivot a {(exp, ten): val} dict into a DataFrame (expiry × tenor)."""
    rows = []
    for e in exp_keys:
        if not any(k[0] == e for k in table):
            continue
        row = {"Expiry": e}
        for t in ten_keys:
            row[t] = table.get((e, t), np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Expiry")


def print_param_table(df, caption, fmt="{:.4f}"):
    cols = [c for c in df.columns if not df[c].isna().all()]
    print("\n" + caption)
    print(df[cols].to_string(float_format=lambda x: fmt.format(x), na_rep="—"))


# ---------------------------------------------------------------------------
# P3 — Vol profiles and smile fit plots
# ---------------------------------------------------------------------------

def plot_vol_profiles(swaption_slices, save_path=None):
    """
    Plot DD vs SABR implied vol profiles over K ∈ [1%, 10%] for
    three target swaptions: Payer 1×1, Payer 10×10, Receiver 30×30.
    """
    K_range = np.linspace(0.01, 0.10, 300)
    targets = [("1Y", "1Y", True), ("10Y", "10Y", True), ("30Y", "30Y", False)]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (exp_str, ten_str, is_payer) in zip(axes, targets):
        sl = next(
            (s for s in swaption_slices
             if s["exp_str"] == exp_str and s["ten_str"] == ten_str),
            None,
        )
        if sl is None:
            ax.set_visible(False); continue

        F, A, T = sl["F"], sl["A"], sl["T_exp"]
        sig, b         = sl.get("dd_params",   (0.20, 0.5))
        a, b_s, rho, nu = sl.get("sabr_params", (0.05, BETA_SABR, -0.3, 0.4))

        dd_vols   = [dd_implied_lognormal_vol(F, K, T, sig, b, A, is_payer) for K in K_range]
        dd_vols   = [v * 100 if (v is not None and np.isfinite(v)) else np.nan for v in dd_vols]
        sabr_vols = [sabr_vol(F, K, T, a, b_s, rho, nu) * 100 for K in K_range]

        ax.plot(K_range * 100, dd_vols,   color="navy",       lw=2.2,
                label="DD  (σ={:.3f}, β={:.3f})".format(sig, b))
        ax.plot(K_range * 100, sabr_vols, color="darkorange", lw=2.2, ls="--",
                label="SABR  (α={:.3f}, ρ={:.3f}, ν={:.3f})".format(a, rho, nu))
        ax.axvline(F * 100, color="dimgrey", ls=":", lw=1.4,
                   label="ATM  {:.2f}%".format(F * 100))

        mkt_ks = [K for K, v in zip(sl["strikes"], sl["vols_pct"])
                  if v is not None and not (isinstance(v, float) and np.isnan(v))]
        mkt_vs = [v for v in sl["vols_pct"]
                  if v is not None and not (isinstance(v, float) and np.isnan(v))]
        lbl = "Synthetic" if sl.get("synthetic") else "Market"
        ax.scatter([k * 100 for k in mkt_ks], mkt_vs,
                   color="black", marker="o", s=28, zorder=5, label=lbl)

        ptype = "Payer" if is_payer else "Receiver"
        syn   = " (synthetic)" if sl.get("synthetic") else ""
        ax.set_title("{} {}×{}{}".format(ptype, exp_str, ten_str, syn),
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Strike K (%)")
        ax.set_ylabel("Lognormal Implied Vol (%)")
        ax.legend(fontsize=7.5)
        ax.set_xlim(1, 10); ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))

    fig.suptitle(
        "Part P3 – DD vs SABR Implied Vol Profiles  K ∈ [1%, 10%]\n"
        "(Multi-Curve: OIS discounting | Term SOFR projection)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_smile_fits(swaption_slices, save_path_prefix=None):
    """
    For each expiry row, plot DD and SABR smile fits vs market quotes.
    One figure per expiry, two panels (DD | SABR).
    """
    K_grid   = np.linspace(0.005, 0.10, 300)
    exp_list = sorted(set(sl["exp_str"] for sl in swaption_slices),
                      key=tenor_to_years)
    colors   = plt.cm.tab10.colors

    for exp_str in exp_list:
        subset = [sl for sl in swaption_slices if sl["exp_str"] == exp_str]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle("Smile Fit – Expiry {}  (Multi-Curve)".format(exp_str),
                     fontsize=12, fontweight="bold")

        for idx, sl in enumerate(subset):
            F, A, T = sl["F"], sl["A"], sl["T_exp"]
            c       = colors[idx % 10]

            sig, b         = sl.get("dd_params",   (0.20, 0.5))
            a, b_s, rho, nu = sl.get("sabr_params", (0.05, BETA_SABR, -0.3, 0.4))

            dd_v   = [dd_implied_lognormal_vol(F, K, T, sig, b, A, True) for K in K_grid]
            dd_v   = [v * 100 if (v is not None and np.isfinite(v)) else np.nan for v in dd_v]
            sabr_v = [sabr_vol(F, K, T, a, b_s, rho, nu) * 100 for K in K_grid]

            axes[0].plot(K_grid * 100, dd_v,   color=c, lw=1.8, label=sl["label"])
            axes[1].plot(K_grid * 100, sabr_v, color=c, lw=1.8, label=sl["label"])

            ks_p = [K for K, v in zip(sl["strikes"], sl["vols_pct"])
                    if v is not None and not (isinstance(v, float) and np.isnan(v))]
            vs_p = [v for v in sl["vols_pct"]
                    if v is not None and not (isinstance(v, float) and np.isnan(v))]
            axes[0].scatter([k * 100 for k in ks_p], vs_p, color=c, marker="o", s=25, zorder=5)
            axes[1].scatter([k * 100 for k in ks_p], vs_p, color=c, marker="o", s=25, zorder=5)

        for ax, title in [(axes[0], "DD Fit"), (axes[1], "SABR Fit")]:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Strike K (%)")
            ax.set_ylabel("Lognormal Vol (%)")
            ax.legend(fontsize=7.5, ncol=2)
            ax.set_xlim(0.5, 11); ax.set_ylim(bottom=0)

        fig.tight_layout()
        if save_path_prefix:
            fig.savefig("{}_exp{}.png".format(save_path_prefix, exp_str), dpi=150)
        plt.show()


def print_vol_tables(swaption_slices):
    """
    Print DD and SABR implied vol tables at K = 1%, 2%, …, 10%
    for the three target swaptions in Part 3.
    """
    K_table = np.arange(0.01, 0.101, 0.01)
    targets = [("1Y", "1Y", True), ("10Y", "10Y", True), ("30Y", "30Y", False)]

    for (exp_str, ten_str, is_payer) in targets:
        sl = next(
            (s for s in swaption_slices
             if s["exp_str"] == exp_str and s["ten_str"] == ten_str),
            None,
        )
        if sl is None:
            continue
        F, A, T = sl["F"], sl["A"], sl["T_exp"]
        sig, b         = sl.get("dd_params",   (0.20, 0.5))
        a, b_s, rho, nu = sl.get("sabr_params", (0.05, BETA_SABR, -0.3, 0.4))
        ptype = "Payer" if is_payer else "Receiver"
        syn   = " (synthetic)" if sl.get("synthetic") else ""

        rows = []
        for K in K_table:
            dd_v   = dd_implied_lognormal_vol(F, K, T, sig, b, A, is_payer)
            dd_v   = dd_v * 100 if (dd_v is not None and np.isfinite(dd_v)) else np.nan
            sabr_v = sabr_vol(F, K, T, a, b_s, rho, nu) * 100
            rows.append({
                "Strike K (%)":         "{:.0f}%".format(K * 100),
                "DD Implied Vol (%)":   dd_v,
                "SABR Implied Vol (%)": sabr_v,
                "Diff (DD-SABR) (%)":   dd_v - sabr_v if np.isfinite(dd_v) else np.nan,
            })

        df_vol = pd.DataFrame(rows).set_index("Strike K (%)")
        caption = "P3 – {} {}×{}{}: Implied Lognormal Vols (%)  [F = {:.4f}%]".format(
            ptype, exp_str, ten_str, syn, F * 100
        )
        print("\n" + caption)
        print(df_vol.to_string(float_format=lambda x: "{:.4f}%".format(x), na_rep="—"))


def print_summary(sigma_table, beta_table, alpha_table, rho_table, nu_table,
                  swaption_slices):
    print("=" * 70)
    print(" Part II – Calibration Summary")
    print("=" * 70)

    print("\n[P3] Target Swaption Details")
    print("  {:>12}  {:>8}  {:>8}  {:>10}".format("Slice", "F (%)", "Annuity", "ATM Vol"))
    print("  " + "-" * 44)
    for (exp_s, ten_s) in [("1Y", "1Y"), ("10Y", "10Y"), ("30Y", "30Y")]:
        sl = next(
            (s for s in swaption_slices
             if s["exp_str"] == exp_s and s["ten_str"] == ten_s),
            None,
        )
        if sl:
            atm_mkt = sl["vols_pct"][5]
            atm_str = "{:.2f}%".format(atm_mkt) if atm_mkt is not None else "synth"
            print("  {:>12}  {:>8.4f}  {:>8.4f}  {:>10}".format(
                sl["label"], sl["F"] * 100, sl["A"], atm_str
            ))

    sigs = [v for v in sigma_table.values() if np.isfinite(v)]
    bets = [v for v in beta_table.values()  if np.isfinite(v)]
    print("\n[P1] DD Parameter Ranges")
    print("  σ : min={:.4f}  max={:.4f}  mean={:.4f}".format(
        min(sigs), max(sigs), np.mean(sigs)))
    print("  β : min={:.4f}  max={:.4f}  mean={:.4f}".format(
        min(bets), max(bets), np.mean(bets)))

    alps = [v for v in alpha_table.values() if np.isfinite(v)]
    rhos = [v for v in rho_table.values()   if np.isfinite(v)]
    nus  = [v for v in nu_table.values()    if np.isfinite(v)]
    print("\n[P2] SABR Parameter Ranges")
    print("  α : min={:.4f}  max={:.4f}  mean={:.4f}".format(
        min(alps), max(alps), np.mean(alps)))
    print("  ρ : min={:.4f}  max={:.4f}  mean={:.4f}".format(
        min(rhos), max(rhos), np.mean(rhos)))
    print("  ν : min={:.4f}   max={:.4f}   mean={:.4f}".format(
        min(nus), max(nus), np.mean(nus)))

    print("=" * 65)


# ---------------------------------------------------------------------------
# Load swaption market data and build slices (executed on import)
# ---------------------------------------------------------------------------

def _load_and_calibrate(filepath=FILEPATH):
    """Load swaption data, build slices, run both calibrations."""
    wb    = openpyxl.load_workbook(filepath, data_only=True)
    ws_sw = wb["Swaption"]
    swaption_rows = []
    for row in ws_sw.iter_rows(min_row=4, values_only=True):
        if row[0] is None or row[1] is None:
            continue
        exp_str = str(row[0]).strip()
        ten_str = str(row[1]).strip()
        vols    = [row[i] for i in range(2, 13)]
        swaption_rows.append((exp_str, ten_str, vols))
    wb.close()
    print("  Loaded {:2d} swaption (expiry × tenor) cells".format(len(swaption_rows)))

    # Build raw vols dict (decimal) for smile construction
    raw_vols_dec = {}
    for (exp_str, ten_str, vols_pct) in swaption_rows:
        Te = tenor_to_years(exp_str); Tn = tenor_to_years(ten_str)
        raw_vols_dec[(Te, Tn)] = [
            (v / 100.0 if v is not None else np.nan) for v in vols_pct
        ]

    slices = build_swaption_slices(swaption_rows, raw_vols_dec)

    print("\n⏳ Calibrating Displaced-Diffusion model (all 16 slices) …")
    sigma_tbl, beta_tbl = calibrate_dd_grid(slices)
    print("✓ DD calibration complete")

    print("\n⏳ Calibrating SABR model (all 16 slices) …")
    alpha_tbl, rho_tbl, nu_tbl = calibrate_sabr_grid(slices)
    print("✓ SABR calibration complete")

    return slices, sigma_tbl, beta_tbl, alpha_tbl, rho_tbl, nu_tbl


# Run calibration and expose results as module-level names
_cal = _load_and_calibrate()
swaption_slices, sigma_table, beta_table, alpha_table, rho_table, nu_table = _cal


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_param_table(
        make_param_df(sigma_table),
        "P1 – Displaced-Diffusion: σ Parameter"
    )
    print_param_table(
        make_param_df(beta_table),
        "P1 – Displaced-Diffusion: β Parameter"
    )
    print_param_table(
        make_param_df(alpha_table),
        "P2 – SABR: α Parameter  (β_CEV = {:.2f} fixed)".format(BETA_SABR)
    )
    print_param_table(
        make_param_df(rho_table),
        "P2 – SABR: ρ Parameter"
    )
    print_param_table(
        make_param_df(nu_table),
        "P2 – SABR: ν Parameter"
    )
    print_vol_tables(swaption_slices)
    print_summary(sigma_table, beta_table, alpha_table, rho_table, nu_table,
                  swaption_slices)
    plot_vol_profiles(swaption_slices)
    plot_smile_fits(swaption_slices)
