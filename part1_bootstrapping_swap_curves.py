"""
Part 1 — Bootstrapping Swap Curves
====================================
Bootstraps three yield curves from market data:
  1. LIBOR single-curve vs multi-curve (OIS-discounted) discount factors
  2. OIS (SOFR) discount curve
  3. Term SOFR 3-month forward rates

"""

import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXCEL_INPUT = "Swap and Swaption Markets.xlsx"


# ---------------------------------------------------------------------------
# Tenor utilities
# ---------------------------------------------------------------------------

def tenor_to_years(t: str) -> float:
    """Convert tenor string ('6m', '1wk', '2y', '18m') to decimal years."""
    t = str(t).strip()
    if re.fullmatch(r"\d+wk", t, re.I):
        return int(re.findall(r"\d+", t)[0]) / 52
    if re.fullmatch(r"\d+m", t, re.I):
        return int(re.findall(r"\d+", t)[0]) / 12
    if re.fullmatch(r"\d+[yY]", t):
        return float(re.findall(r"\d+", t)[0])
    return float(t)


def years_to_label(y: float) -> str:
    """Convert decimal years to a readable tenor label."""
    if y < 1.0 / 12:
        return "{:.0f}wk".format(round(y * 52))
    if y < 1.0:
        return "{:.0f}m".format(round(y * 12))
    if y == int(y):
        return "{:.0f}y".format(y)
    return "{:.2f}y".format(y)


# ---------------------------------------------------------------------------
# Market data loading
# ---------------------------------------------------------------------------

def load_market_data(path=EXCEL_INPUT):
    """
    Read LIBOR (legacy), OIS (SOFR), and OIS (Term SOFR) market data.

    The OIS (SOFR) rates are hard-coded from the original market snapshot
    because the Excel sheet for that curve is not present in the workbook.

    Returns
    -------
    libor_data : list of (tenor_years, product_str, par_rate)
    ois_data   : list of (tenor_years, par_rate)
    ts_data    : list of (tenor_years, par_rate)
    """
    xl = pd.ExcelFile(path)

    # LIBOR instruments (deposits + IRS)
    df_lib = pd.read_excel(xl, sheet_name="LIBOR (legacy)", header=0, usecols=[0, 1, 2])
    df_lib.columns = ["Tenor", "Product", "Rate"]
    libor_data = [
        (tenor_to_years(str(r.Tenor)), str(r.Product).strip(), float(r.Rate))
        for _, r in df_lib.iterrows()
        if pd.notna(r.Tenor) and pd.notna(r.Rate)
    ]

    # OIS (SOFR) par rates — hard-coded market snapshot (rates in %)
    ois_raw = {
        "1wk":  3.65950, "2wk":  3.65801, "3wk":  3.66100,
        "1m":   3.67010, "2m":   3.66600, "3m":   3.65950,
        "4m":   3.64900, "5m":   3.63505, "6m":   3.61610,
        "7m":   3.59455, "8m":   3.57205, "9m":   3.55165,
        "10m":  3.53050, "11m":  3.51145, "12m":  3.49570,
        "18m":  3.41118, "2y":   3.40095, "3y":   3.43065,
        "4y":   3.48920, "5y":   3.55475, "6y":   3.62500,
        "7y":   3.69400, "8y":   3.75800, "9y":   3.81835,
        "10y":  3.87555, "12y":  3.98115, "15y":  4.10777,
        "20y":  4.21850, "25y":  4.24080, "30y":  4.21930,
        "40y":  4.12250, "50y":  4.01700,
    }
    ois_data = [(tenor_to_years(t), r / 100.0) for t, r in ois_raw.items()]

    # Term SOFR swap instruments
    df_ts = pd.read_excel(xl, sheet_name="OIS (Term SOFR)", header=0, usecols=[0, 1])
    df_ts.columns = ["Tenor", "Rate"]
    ts_data = [
        (tenor_to_years(str(r.Tenor)), float(r.Rate))
        for _, r in df_ts.iterrows()
        if pd.notna(r.Tenor) and pd.notna(r.Rate)
    ]

    print("  Loaded {:2d} LIBOR instruments".format(len(libor_data)))
    print("  Loaded {:2d} OIS (SOFR) instruments".format(len(ois_data)))
    print("  Loaded {:2d} Term-SOFR instruments".format(len(ts_data)))
    return libor_data, ois_data, ts_data


# ---------------------------------------------------------------------------
# Bootstrapping utilities
# ---------------------------------------------------------------------------

def df_interp(T_query, tenors, dfs):
    """
    Linear interpolation on discount factors.
    fill_value='extrapolate' is a safety net for edge cases only;
    under normal operation all queries fall within the pillar range.
    """
    f = interp1d(tenors, dfs, kind="linear", fill_value="extrapolate")
    return float(f(T_query))


def build_payment_schedule(start, end, freq_per_year):
    """
    Build a list of (t_start, t_end, year_frac) payment periods.
    Correctly handles non-integer maturities via a final stub period.

    Examples
    --------
    18m annual OIS  ->  [(0, 1, 1.0), (1, 1.5, 0.5)]   # stub at end
    2y semi-annual  ->  [(0, 0.5, 0.5), ..., (1.5, 2.0, 0.5)]
    """
    dt = 1.0 / freq_per_year
    payments = []
    t = start
    while t + dt <= end + 1e-9:
        payments.append((t, t + dt, dt))
        t = round(t + dt, 10)
    if t < end - 1e-9:                         # final stub period
        payments.append((t, end, round(end - t, 10)))
    return payments


# ---------------------------------------------------------------------------
# P1 — LIBOR single-curve bootstrap
# ---------------------------------------------------------------------------

def bootstrap_libor_single(libor_data, n_iter=5):
    """
    Single-curve bootstrap: LIBOR discounts itself.
    Day count 30/360, semi-annual fixed and floating legs.

    Returns
    -------
    tenors : np.ndarray
    dfs    : np.ndarray — D_LIBOR(0, T)
    """
    freq  = 2
    alpha = 1.0 / freq

    def _pass0():
        tenors = [0.0]; dfs = [1.0]
        for (T, product, rate) in libor_data:
            if product == "LIBOR":
                df_T = 1.0 / (1.0 + rate * T)
            else:
                schedule = build_payment_schedule(0, T, freq)
                sum_df = sum(
                    df_interp(te, tenors, dfs)
                    for (_, te, __) in schedule[:-1]
                )
                df_T = (1.0 - rate * alpha * sum_df) / (1.0 + rate * alpha)
            tenors.append(T); dfs.append(df_T)
        return tenors, dfs

    def _refine(tenors_prev, dfs_prev):
        tenors = [0.0]; dfs = [1.0]
        for (T, product, rate) in libor_data:
            if product == "LIBOR":
                df_T = 1.0 / (1.0 + rate * T)
            else:
                schedule = build_payment_schedule(0, T, freq)
                sum_df = sum(
                    df_interp(te, tenors_prev, dfs_prev)
                    for (_, te, __) in schedule[:-1]
                )
                df_T = (1.0 - rate * alpha * sum_df) / (1.0 + rate * alpha)
            tenors.append(T); dfs.append(df_T)
        return tenors, dfs

    tenors, dfs = _pass0()
    for _ in range(n_iter):
        tenors, dfs = _refine(tenors, dfs)
    return np.array(tenors), np.array(dfs)


# ---------------------------------------------------------------------------
# P1 — LIBOR multi-curve bootstrap (OIS discounting)
# ---------------------------------------------------------------------------

def bootstrap_libor_multi(libor_data, ois_tenors, ois_dfs):
    """
    Multi-curve bootstrap: LIBOR projection curve with OIS discounting.
    Fixed and floating legs both use OIS discount factors;
    the projection curve is solved pillar by pillar.

    Returns
    -------
    tenors : np.ndarray
    dfs    : np.ndarray — D_L(0, T)  projection discount factors
    """
    freq  = 2
    alpha = 1.0 / freq

    def _dl_interp(t, t_arr, d_arr):
        return float(np.interp(t, t_arr, d_arr))

    tenors = [0.0]; dfs = [1.0]
    for (T, product, rate) in libor_data:
        if product == "LIBOR":
            df_L = 1.0 / (1.0 + rate * T)
        else:
            schedule = build_payment_schedule(0, T, freq)
            c_half = rate * alpha

            # Fixed-leg PV (discounted at OIS)
            fixed_pv = sum(
                c_half * df_interp(te, ois_tenors, ois_dfs)
                for (_, te, __) in schedule
            )

            # Float PV for all periods except the last (projection DFs known)
            float_known = 0.0
            for (ts, te, _) in schedule[:-1]:
                dl_s = _dl_interp(ts, tenors, dfs)
                dl_e = _dl_interp(te, tenors, dfs)
                do_e = df_interp(te, ois_tenors, ois_dfs)
                float_known += (dl_s / dl_e - 1.0) * do_e

            # Solve for D_L(T) from the last payment period
            t_prev  = schedule[-1][0]
            dl_prev = _dl_interp(t_prev, tenors, dfs)
            do_T    = df_interp(T, ois_tenors, ois_dfs)

            A    = fixed_pv - float_known
            df_L = dl_prev * do_T / (A + do_T)

        tenors.append(T); dfs.append(df_L)

    return np.array(tenors), np.array(dfs)


# ---------------------------------------------------------------------------
# P2 — OIS (SOFR) bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ois(ois_data, n_iter=5):
    """
    Bootstrap D_OIS(0, T) from SOFR OIS par rates (annual fixed leg).
    Iterative self-referential bootstrap to handle stub periods.

    Returns
    -------
    tenors : np.ndarray
    dfs    : np.ndarray — D_OIS(0, T)
    """
    def _pass0():
        tenors = [0.0]; dfs = [1.0]
        for (T, rate) in ois_data:
            schedule = build_payment_schedule(0, T, freq_per_year=1)
            if len(schedule) == 0:
                df_T = 1.0 / (1.0 + rate * T)
            else:
                alpha_last = schedule[-1][2]
                sum_df = sum(
                    alpha * df_interp(te, tenors, dfs)
                    for (_, te, alpha) in schedule[:-1]
                )
                df_T = (1.0 - rate * sum_df) / (1.0 + rate * alpha_last)
            tenors.append(T); dfs.append(df_T)
        return tenors, dfs

    def _refine(tenors_prev, dfs_prev):
        tenors = [0.0]; dfs = [1.0]
        for (T, rate) in ois_data:
            schedule = build_payment_schedule(0, T, freq_per_year=1)
            if len(schedule) == 0:
                df_T = 1.0 / (1.0 + rate * T)
            else:
                alpha_last = schedule[-1][2]
                sum_df = sum(
                    alpha * df_interp(te, tenors_prev, dfs_prev)
                    for (_, te, alpha) in schedule[:-1]
                )
                df_T = (1.0 - rate * sum_df) / (1.0 + rate * alpha_last)
            tenors.append(T); dfs.append(df_T)
        return tenors, dfs

    tenors, dfs = _pass0()
    for _ in range(n_iter):
        tenors, dfs = _refine(tenors, dfs)
    return np.array(tenors), np.array(dfs)


# ---------------------------------------------------------------------------
# P3 — Term SOFR bootstrap
# ---------------------------------------------------------------------------

def bootstrap_term_sofr(ts_data, ois_tenors, ois_dfs):
    """
    Bootstrap a quarterly Term-SOFR-3M projection curve and extract
    3-month forward rates.  Fixed-leg payments are annual; floating leg
    resets quarterly.  OIS discount factors are used throughout.

    Returns
    -------
    q_grid     : np.ndarray  — quarterly time grid
    proj_dfs   : np.ndarray  — projection discount factors on q_grid
    fwd_tenors : np.ndarray  — start date of each 3M forward period
    fwd_rates  : np.ndarray  — 3M forward Term SOFR rates
    """
    alpha_q    = 0.25      # quarterly accrual factor
    freq_fixed = 1         # annual fixed-leg payments

    max_T  = ts_data[-1][0]
    n_q    = int(round(max_T / alpha_q))
    q_grid = np.round(np.arange(0, n_q + 1) * alpha_q, 10)

    proj_dfs    = np.full(len(q_grid), np.nan)
    proj_dfs[0] = 1.0

    def fill_intermediate(t_start, t_end, df_start, df_end):
        """Linearly interpolate quarterly DFs between two known pillars."""
        i_s = int(round(t_start / alpha_q))
        i_e = int(round(t_end   / alpha_q))
        if i_e <= i_s:
            return
        n    = i_e - i_s
        step = (df_end - df_start) / n
        for k in range(1, n):
            proj_dfs[i_s + k] = df_start + k * step
        proj_dfs[i_e] = df_end

    last_known_T  = 0.0
    last_known_df = 1.0

    for (T, rate) in ts_data:
        schedule_fixed = build_payment_schedule(0, T, freq_fixed)
        fixed_pv = sum(
            rate * df_interp(te, ois_tenors, ois_dfs)
            for (_, te, __) in schedule_fixed
        )

        def swap_pv(df_T_guess):
            fill_intermediate(last_known_T, T, last_known_df, df_T_guess)
            float_pv = 0.0
            for (ts, te, _) in build_payment_schedule(0, T, freq_per_year=4):
                i_ts = int(round(ts / alpha_q))
                i_te = int(round(te / alpha_q))
                d_ts = proj_dfs[i_ts]; d_te = proj_dfs[i_te]
                if np.isnan(d_ts) or np.isnan(d_te) or d_te == 0:
                    return np.nan
                fwd      = (d_ts / d_te - 1.0) / alpha_q
                d_ois_te = df_interp(te, ois_tenors, ois_dfs)
                float_pv += fwd * alpha_q * d_ois_te
            return fixed_pv - float_pv

        try:
            df_T_sol = brentq(swap_pv, 1e-6, 1.0, xtol=1e-10)
        except Exception:
            df_T_sol = df_interp(T, ois_tenors, ois_dfs)   # fallback

        fill_intermediate(last_known_T, T, last_known_df, df_T_sol)
        last_known_T  = T
        last_known_df = df_T_sol

    # Extract 3M forward rates from the projection curve
    fwd_tenors, fwd_rates = [], []
    for i in range(len(q_grid) - 1):
        d1 = proj_dfs[i]; d2 = proj_dfs[i + 1]
        if not (np.isnan(d1) or np.isnan(d2) or d2 == 0):
            fwd_tenors.append(q_grid[i])
            fwd_rates.append((d1 / d2 - 1.0) / alpha_q)

    return q_grid, proj_dfs, np.array(fwd_tenors), np.array(fwd_rates)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_p1_table(T_sc, DF_sc, T_mc, DF_mc, libor_data):
    par_rates = [r[2] for r in libor_data]
    df_p1 = pd.DataFrame({
        "Tenor (y)":           T_sc[1:],
        "Par Rate":            par_rates,
        "Single-Curve D(0,T)": DF_sc[1:],
        "Multi-Curve D(0,T)":  DF_mc[1:],
        "Spread (bp)":         (DF_sc[1:] - DF_mc[1:]) * 1e4,
    }).set_index("Tenor (y)")

    print("=" * 72)
    print("P1 – LIBOR Discount Factors: Single-Curve vs Multi-Curve")
    print("=" * 72)
    print(df_p1.to_string(
        float_format=lambda x: "{:.6f}".format(x),
        formatters={
            "Par Rate":    lambda x: "{:.4f}%".format(x * 100),
            "Spread (bp)": lambda x: "{:.2f}".format(x),
        }
    ))
    print("=" * 72)
    print("Note: positive spread = single-curve DF > multi-curve DF (LIBOR > OIS rate)")


def print_p2_table(T_ois, DF_ois):
    ois_f      = interp1d(T_ois, DF_ois, kind="linear", fill_value="extrapolate")
    ois_Ts     = T_ois[1:]
    ois_dfs    = [float(ois_f(T)) for T in ois_Ts]
    zero_rates = [-np.log(d) / T for d, T in zip(ois_dfs, ois_Ts)]

    df_p2 = pd.DataFrame({
        "Tenor":          [years_to_label(T) for T in ois_Ts],
        "Tenor (y)":      ois_Ts,
        "D_OIS(0,T)":     ois_dfs,
        "Zero Rate (cc)": zero_rates,
    }).set_index("Tenor")

    print("=" * 72)
    print("P2 – OIS (SOFR) Discount Factors  D_o(0,T),  T in [0, 50]")
    print("=" * 72)
    print(df_p2.to_string(
        float_format=lambda x: "{:.6f}".format(x),
        formatters={
            "Tenor (y)":      lambda x: "{:.4f}".format(x),
            "Zero Rate (cc)": lambda x: "{:.4f}%".format(x * 100),
        }
    ))
    print("=" * 72)


def print_p3_table(fwd_t, fwd_r):
    df_p3 = pd.DataFrame({
        "Start T (y)":  fwd_t,
        "End T+3M (y)": fwd_t + 0.25,
        "3M Fwd Rate":  fwd_r,
    }).set_index("Start T (y)")

    print("=" * 72)
    print("P3 – Bootstrapped 3M Forward Term-SOFR Rates  f(T, T+3M)")
    print("=" * 72)
    # Display a representative subset (every 4th quarter = annual spacing)
    display_idx = list(range(0, len(fwd_t), 4)) + [len(fwd_t) - 1]
    print(df_p3.iloc[display_idx].to_string(
        float_format=lambda x: "{:.6f}".format(x),
        formatters={
            "End T+3M (y)": lambda x: "{:.2f}".format(x),
            "3M Fwd Rate":  lambda x: "{:.4f}%".format(x * 100),
        }
    ))
    print("=" * 72)
    print("Total periods: {}  |  Min: {:.4f}%  |  Max: {:.4f}%  |  Mean: {:.4f}%".format(
        len(fwd_r),
        fwd_r.min()  * 100,
        fwd_r.max()  * 100,
        fwd_r.mean() * 100,
    ))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_p1(T_sc, DF_sc, T_mc, DF_mc, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(T_sc, DF_sc, "o-",  color="#1f77b4", lw=2.2, ms=7,
            label="Single-curve  (LIBOR discounting, uncollateralised)")
    ax.plot(T_mc, DF_mc, "s--", color="#d62728", lw=2.2, ms=7,
            label="Multi-curve  (OIS/SOFR discounting, collateralised)")
    mc_at_sc = np.interp(T_sc, T_mc, DF_mc)
    ax.fill_between(T_sc, DF_sc, mc_at_sc,
                    alpha=0.13, color="purple", label="DF spread (CSA impact)")
    ax.set_title("P1 – LIBOR IRS: Single-Curve vs Multi-Curve  D(0,T)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Maturity T (years)", fontsize=11)
    ax.set_ylabel("Discount Factor D(0,T)", fontsize=11)
    ax.set_xlim(0, 30); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_p2(T_ois, DF_ois, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(T_ois, DF_ois, "o-", color="#2ca02c", lw=2.2, ms=4,
            label="OIS (SOFR) Discount Factor D_o(0,T)")
    ax.set_title("P2 – OIS (SOFR) Discount Factor D_o(0,T),  T in [0, 50]",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Maturity T (years)", fontsize=11)
    ax.set_ylabel("Discount Factor D_o(0,T)", fontsize=11)
    ax.set_xlim(0, 50); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_p3(fwd_t, fwd_r, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(fwd_t + 0.125, fwd_r * 100,
            color="#ff7f0e", lw=2, label="3M Forward Term-SOFR Rate")
    ax.axhline(y=fwd_r.mean() * 100, color="grey", ls=":", lw=1.2,
               label="Mean = {:.3f}%".format(fwd_r.mean() * 100))
    ax.set_title("P3 – Bootstrapped 3M Forward Term-SOFR Rates  f(T, T+3M)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Rate Start Date T (years)", fontsize=11)
    ax.set_ylabel("Forward Rate (%)", fontsize=11)
    ax.set_xlim(0, 30); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "{:.2f}%".format(x)))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def print_summary(T_sc, DF_sc, T_mc, DF_mc, T_ois, DF_ois, fwd_r):
    print("=" * 65)
    print("Bootstrapping Summary")
    print("=" * 65)

    print("\nP1 – LIBOR Discount Factor Spread (Single minus Multi, basis points)")
    for T, sc, mc in zip(T_sc[1:], DF_sc[1:], DF_mc[1:]):
        if T in [1, 2, 5, 10, 15, 20, 30]:
            print("  D(0,{:2.0f}y):  SC={:.6f}  MC={:.6f}  spread={:.2f} bp".format(
                T, sc, mc, (sc - mc) * 1e4))

    print("\nP2 – OIS Zero Rates at key tenors")
    ois_f = interp1d(T_ois, DF_ois, kind="linear", fill_value="extrapolate")
    for T in [1, 2, 5, 10, 20, 30, 50]:
        d = float(ois_f(T))
        z = -np.log(d) / T
        print("  r(0,{:2d}y) = {:.4f}%  |  D_OIS = {:.6f}".format(int(T), z * 100, d))

    print("\nP3 – Forward Term-SOFR statistics")
    print("  Periods : {}".format(len(fwd_r)))
    print("  Min     : {:.4f}%".format(fwd_r.min()  * 100))
    print("  Max     : {:.4f}%".format(fwd_r.max()  * 100))
    print("  Mean    : {:.4f}%".format(fwd_r.mean() * 100))
    print("  Std Dev : {:.4f}%".format(fwd_r.std()  * 100))
    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Module-level bootstrap (executed on import and on direct run)
# ---------------------------------------------------------------------------

def _run_bootstrap(excel_path=EXCEL_INPUT):
    """Run all bootstraps and return results as a dict."""
    print("Loading market data from: {}".format(excel_path))
    libor_data, ois_data, ts_data = load_market_data(excel_path)

    print("\n[OIS] Bootstrapping …")
    T_ois, DF_ois = bootstrap_ois(ois_data)
    print("  Pillars : {}".format(len(T_ois) - 1))
    print("  D_OIS(0, 1y) = {:.6f}".format(df_interp(1,  T_ois, DF_ois)))
    print("  D_OIS(0,10y) = {:.6f}".format(df_interp(10, T_ois, DF_ois)))
    print("  D_OIS(0,30y) = {:.6f}".format(df_interp(30, T_ois, DF_ois)))

    print("\n[LIBOR Single] Bootstrapping …")
    T_sc, DF_sc = bootstrap_libor_single(libor_data)
    print("  Pillars       : {}".format(len(T_sc) - 1))
    print("  D_LIBOR(0,10y) = {:.6f}".format(df_interp(10, T_sc, DF_sc)))
    print("  D_LIBOR(0,30y) = {:.6f}".format(df_interp(30, T_sc, DF_sc)))

    print("\n[LIBOR Multi] Bootstrapping …")
    T_mc, DF_mc = bootstrap_libor_multi(libor_data, T_ois, DF_ois)
    print("  Pillars    : {}".format(len(T_mc) - 1))
    print("  D_L(0,10y) = {:.6f}".format(df_interp(10, T_mc, DF_mc)))
    print("  D_L(0,30y) = {:.6f}".format(df_interp(30, T_mc, DF_mc)))
    print("  Spread@10y : {:.2f} bp".format(
        (df_interp(10, T_sc, DF_sc) - df_interp(10, T_mc, DF_mc)) * 1e4))
    print("  Spread@30y : {:.2f} bp".format(
        (df_interp(30, T_sc, DF_sc) - df_interp(30, T_mc, DF_mc)) * 1e4))

    print("\n[Term SOFR] Bootstrapping …")
    _, _, fwd_t, fwd_r = bootstrap_term_sofr(ts_data, T_ois, DF_ois)
    print("  Quarterly periods : {}".format(len(fwd_t)))
    print("  Min fwd rate      : {:.4f}%".format(fwd_r.min()  * 100))
    print("  Max fwd rate      : {:.4f}%".format(fwd_r.max()  * 100))
    print("  Mean fwd rate     : {:.4f}%".format(fwd_r.mean() * 100))

    return dict(
        libor_data=libor_data,
        ois_data=ois_data,
        ts_data=ts_data,
        T_ois=T_ois, DF_ois=DF_ois,
        T_sc=T_sc,   DF_sc=DF_sc,
        T_mc=T_mc,   DF_mc=DF_mc,
        fwd_t=fwd_t, fwd_r=fwd_r,
    )


# Run bootstrap and expose results as module-level names for downstream imports
_results = _run_bootstrap()

libor_data = _results["libor_data"]
ois_data   = _results["ois_data"]
ts_data    = _results["ts_data"]
T_ois      = _results["T_ois"]
DF_ois     = _results["DF_ois"]
T_sc       = _results["T_sc"]
DF_sc      = _results["DF_sc"]
T_mc       = _results["T_mc"]
DF_mc      = _results["DF_mc"]
fwd_t      = _results["fwd_t"]
fwd_r      = _results["fwd_r"]


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print_p1_table(T_sc, DF_sc, T_mc, DF_mc, libor_data)
    print()
    print_p2_table(T_ois, DF_ois)
    print()
    print_p3_table(fwd_t, fwd_r)
    print()
    print_summary(T_sc, DF_sc, T_mc, DF_mc, T_ois, DF_ois, fwd_r)

    plot_p1(T_sc, DF_sc, T_mc, DF_mc)
    plot_p2(T_ois, DF_ois)
    plot_p3(fwd_t, fwd_r)
