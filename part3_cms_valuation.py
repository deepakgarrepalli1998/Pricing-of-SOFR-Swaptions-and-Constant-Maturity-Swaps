"""
Part 3 — CMS Valuation via Static Replication
==============================================
Values CMS legs and computes convexity corrections using the
Carr–Madan h''(K) static replication method (Tee & Kerkhof).

  1. PV of CMS 10Y semi-annual leg (5-year term)
  2. PV of CMS 2Y quarterly leg (10-year term)
  3. Forward swap rate vs CMS rate — convexity correction table & plots

Theory
------
Under the annuity measure, E^A[S_T] = F_S (forward swap rate).
CMS coupons are paid under the OIS payment-date measure, introducing
a convexity adjustment:

    CMS Rate = E^{T_pay}[S_T] = F_S + Convexity Adjustment

The adjustment is obtained via Carr–Madan replication:

    Conv. Adj. = (1/D_OIS(T_pay)) * [
        ∫_{F_S}^∞  h''(K) V_pay(K) dK  +
        ∫_0^{F_S}  h''(K) V_rec(K) dK
    ]

where h(K) = K / IRR(K) and IRR uses OIS-anchored discounting.

"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import Part 2 pricing functions and calibrated objects
# ---------------------------------------------------------------------------
from part2_swaption_calibration import (
    black_price,
    black_iv,
    sabr_vol,
    fwd_swap_rate,
    annuity,
    D_ois,
    D_ts,
    swaption_slices,
)

BETA_SABR = 0.75

print("Part 1 + Part 2 imported successfully.")
print("\nCurve check points:")
print("  {:>4}  {:>10}  {:>10}".format("T", "D_OIS", "D_TS"))
print("  " + "-" * 28)
for t in [1, 5, 10, 20, 30, 50]:
    print("  {:>4}  {:>10.6f}  {:>10.6f}".format(t, D_ois(t), D_ts(t)))
print("  Swaption slices: {}".format(len(swaption_slices)))
print("  SABR params available: {}".format(
    all("sabr_params" in sl for sl in swaption_slices)
))


# ---------------------------------------------------------------------------
# SABR parameter interpolation
# ---------------------------------------------------------------------------

def _build_sabr_interpolators(swaption_slices):
    """
    Build bilinear interpolators over the calibrated SABR grid so that
    SABR parameters can be evaluated at arbitrary (T_exp, T_ten).
    """
    sabr_calib = {}
    for sl in swaption_slices:
        if "sabr_params" in sl:
            a, b, rho, nu = sl["sabr_params"]
            sabr_calib[(sl["T_exp"], sl["T_ten"])] = {
                "alpha": a, "rho": rho, "nu": nu
            }

    exp_nodes = sorted(set(k[0] for k in sabr_calib))
    ten_nodes = sorted(set(k[1] for k in sabr_calib))

    def _build_per_param(key):
        d = {}
        for Tn in ten_nodes:
            xs = [Te for Te in exp_nodes if (Te, Tn) in sabr_calib]
            ys = [sabr_calib[(Te, Tn)][key] for Te in xs]
            if len(xs) >= 2:
                d[Tn] = interp1d(xs, ys, kind="linear", fill_value="extrapolate")
        return d

    ai = _build_per_param("alpha")
    ri = _build_per_param("rho")
    ni = _build_per_param("nu")

    def get_sabr_params(T_exp, T_ten):
        """Bilinear SABR interpolation for arbitrary (T_exp, T_ten)."""
        def _over_tenor(interps, Te):
            xs = sorted(interps)
            ys = [float(interps[t](Te)) for t in xs]
            if len(xs) > 1:
                return float(interp1d(xs, ys, kind="linear",
                                      fill_value="extrapolate")(T_ten))
            return ys[0]
        return (
            _over_tenor(ai, T_exp),
            BETA_SABR,
            _over_tenor(ri, T_exp),
            _over_tenor(ni, T_exp),
        )

    return get_sabr_params


get_sabr_params = _build_sabr_interpolators(swaption_slices)

# Sanity check at a calibrated node
a_chk, _, rho_chk, nu_chk = get_sabr_params(10.0, 10.0)
sl_10x10 = next(
    s for s in swaption_slices
    if s["T_exp"] == 10.0 and s["T_ten"] == 10.0
)
a_cal, _, rho_cal, nu_cal = sl_10x10["sabr_params"]
print("\nSABR interpolation check (10Y×10Y):")
print("  Calibrated:   alpha={:.4f}  rho={:+.4f}  nu={:.4f}".format(a_cal, rho_cal, nu_cal))
print("  Interpolated: alpha={:.4f}  rho={:+.4f}  nu={:.4f}".format(a_chk, rho_chk, nu_chk))
print("  Match: {}".format(
    np.allclose([a_cal, rho_cal, nu_cal], [a_chk, rho_chk, nu_chk], atol=1e-6)
))
print("\nCMS functions ready.")


# ---------------------------------------------------------------------------
# OIS-anchored IRR annuity and derivatives
# ---------------------------------------------------------------------------

def IRR_annuity(S, n, T_exp):
    """
    OIS-anchored IRR annuity factor.
    IRR(S, n, T_exp) = D_OIS(T_exp) * (1 - (1+S)^{-n}) / S
    """
    D_anc = D_ois(T_exp)
    if abs(S) < 1e-10:
        return D_anc * float(n)
    return D_anc * (1.0 - (1.0 + S) ** (-n)) / S


def IRR_deriv1(S, n, T_exp):
    """First derivative d(IRR)/dS — analytical."""
    D_anc = D_ois(T_exp)
    if abs(S) < 1e-10:
        return -D_anc * n * (n + 1) / 2.0
    raw = (n * (1.0 + S) ** (-(n + 1)) - (1.0 - (1.0 + S) ** (-n)) / S) / S
    return D_anc * raw


def IRR_deriv2(S, n, T_exp, dS=1e-5):
    """Second derivative d^2(IRR)/dS^2 via central finite difference."""
    return (IRR_deriv1(S + dS, n, T_exp) - IRR_deriv1(S - dS, n, T_exp)) / (2 * dS)


def h_double_prime(K, n, T_exp):
    """
    h''(K)  where  h(K) = K / IRR(K).

    Derived from the quotient rule:
        h''(K) = [-IRR'' * K - 2 * IRR'] / IRR^2 + 2 * K * IRR'^2 / IRR^3
    (Tee & Kerkhof, Eq. 2.22)
    """
    I   = IRR_annuity(K, n, T_exp)
    Ip  = IRR_deriv1(K, n, T_exp)
    Ipp = IRR_deriv2(K, n, T_exp)
    return (-Ipp * K - 2 * Ip) / I ** 2 + 2 * K * Ip ** 2 / I ** 3


# ---------------------------------------------------------------------------
# CMS convexity adjustment — static replication
# ---------------------------------------------------------------------------

def cms_conv_adj(T_exp, T_ten, T_pay, sabr_p):
    """
    Compute the CMS convexity adjustment via h''(K) static replication.
    Multi-curve: Term SOFR forward rates (Part 1/2), OIS discounting,
    OIS-anchored IRR annuity.

    Parameters
    ----------
    T_exp  : float — swaption expiry (= CMS fixing date)
    T_ten  : float — underlying swap tenor
    T_pay  : float — CMS payment date
    sabr_p : tuple — (alpha, beta, rho, nu)

    Returns
    -------
    F_S      : float — multi-curve forward swap rate
    conv_adj : float — convexity adjustment (F_S → CMS rate)
    cms_rate : float — F_S + conv_adj
    """
    a, b, rho, nu = sabr_p
    F_S   = fwd_swap_rate(T_exp, T_ten, D_ois, D_ts)
    n     = int(round(T_ten))
    IRR_F = IRR_annuity(F_S, n, T_exp)
    D_T   = D_ois(T_pay)

    # Integration range: [5% of F_S, 6× F_S] (truncated Carr–Madan)
    K_lo = max(F_S * 0.05, 1e-4)
    K_hi = F_S * 6.0

    def payer_integrand(K):
        sigma = sabr_vol(F_S, K, T_exp, a, b, rho, nu)
        V_pay = D_T * IRR_F * black_price(F_S, K, T_exp, sigma, A=1.0, is_payer=True)
        return h_double_prime(K, n, T_exp) * V_pay

    def recv_integrand(K):
        sigma = sabr_vol(F_S, K, T_exp, a, b, rho, nu)
        V_rec = D_T * IRR_F * black_price(F_S, K, T_exp, sigma, A=1.0, is_payer=False)
        return h_double_prime(K, n, T_exp) * V_rec

    try:
        I_pay, _ = quad(payer_integrand, F_S, K_hi, limit=200)
        I_rec, _ = quad(recv_integrand, K_lo, F_S, limit=200)
    except Exception:
        I_pay = I_rec = 0.0

    conv_adj = (I_rec + I_pay) / D_T
    return F_S, conv_adj, F_S + conv_adj


# ---------------------------------------------------------------------------
# P1 — CMS 10Y semi-annual, 5-year leg
# ---------------------------------------------------------------------------

def price_cms_10y_semiannual():
    """
    Value the CMS 10Y semi-annual leg over a 5-year term.
    Payment dates: 0.5Y, 1.0Y, …, 5.0Y.

    Returns
    -------
    pv   : float — total PV as a fraction of notional
    rows : list of dict — per-payment breakdown
    """
    print("=" * 70)
    print("P1 — CMS 10Y semi-annual, 5-year leg")
    print("        [Multi-curve + h\"(K) static replication]")
    print("=" * 70)

    rows = []
    pv   = 0.0
    for t in np.arange(0.5, 5.0 + 1e-9, 0.5):
        p          = get_sabr_params(t, 10.0)
        F_S, adj, cms = cms_conv_adj(t, 10.0, t, p)
        d          = D_ois(t)
        pv_pmt     = d * cms * 0.5
        pv        += pv_pmt
        rows.append({
            "Pay Date":             "{:.2f}Y".format(t),
            "Fwd Swap Rate (%)":    "{:.4f}".format(F_S * 100),
            "Conv. Adj (bps)":      "{:.2f}".format(adj * 1e4),
            "CMS Rate (%)":         "{:.4f}".format(cms * 100),
            "D_OIS(0,t)":           "{:.6f}".format(d),
            "PV Contrib (bps)":     "{:.4f}".format(pv_pmt * 1e4),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n  Total PV = {:.4f} bps = {:.6f}% of notional".format(
        pv * 1e4, pv * 100))
    return pv, rows


# ---------------------------------------------------------------------------
# P2 — CMS 2Y quarterly, 10-year leg
# ---------------------------------------------------------------------------

def price_cms_2y_quarterly():
    """
    Value the CMS 2Y quarterly leg over a 10-year term.
    Payment dates: 0.25Y, 0.50Y, …, 10.0Y (40 payments).

    Returns
    -------
    pv   : float — total PV as a fraction of notional
    rows : list of dict — per-payment breakdown
    """
    print("=" * 70)
    print("P2 — CMS 2Y quarterly, 10-year leg")
    print("        [Multi-curve + h\"(K) static replication]")
    print("=" * 70)

    rows = []
    pv   = 0.0
    for t in np.arange(0.25, 10.0 + 1e-9, 0.25):
        p          = get_sabr_params(t, 2.0)
        F_S, adj, cms = cms_conv_adj(t, 2.0, t, p)
        d          = D_ois(t)
        pv_pmt     = d * cms * 0.25
        pv        += pv_pmt
        rows.append({
            "Pay Date":             "{:.2f}Y".format(t),
            "Fwd Swap Rate (%)":    "{:.4f}".format(F_S * 100),
            "Conv. Adj (bps)":      "{:.2f}".format(adj * 1e4),
            "CMS Rate (%)":         "{:.4f}".format(cms * 100),
            "D_OIS(0,t)":           "{:.6f}".format(d),
            "PV Contrib (bps)":     "{:.4f}".format(pv_pmt * 1e4),
        })

    df = pd.DataFrame(rows)
    # Display first 8 + last 4 rows (40 rows total)
    print(pd.concat([df.head(8), df.tail(4)]).to_string(index=False))
    print("  … ({} total payments)".format(len(rows)))
    print("\n  Total PV = {:.4f} bps = {:.6f}% of notional".format(
        pv * 1e4, pv * 100))
    return pv, rows


# ---------------------------------------------------------------------------
# P3 — Forward swap rate vs CMS rate comparison
# ---------------------------------------------------------------------------

def compare_fwd_vs_cms():
    """
    Compare forward swap rates with CMS rates for 6 (expiry, tenor) pairs.
    Returns a DataFrame with forward rate, CMS rate, and convexity correction.
    """
    pairs = [(1, 1), (1, 10), (5, 1), (5, 10), (10, 1), (10, 10)]
    rows  = []
    for Te, Tn in pairs:
        p = get_sabr_params(float(Te), float(Tn))
        F_S, adj, cms = cms_conv_adj(float(Te), float(Tn), float(Te), p)
        rows.append({
            "Expiry × Tenor":       "{}Y × {}Y".format(Te, Tn),
            "Fwd Swap Rate (%)":    round(F_S  * 100, 5),
            "CMS Rate (%)":         round(cms  * 100, 5),
            "Conv. Adj. (bps)":     round(adj  * 1e4, 3),
            "Adj / Fwd (%)":        round(adj / F_S * 100, 3),
        })

    df = pd.DataFrame(rows).set_index("Expiry × Tenor")
    print("\n" + "=" * 70)
    print("P3 — Convexity Correction: Forward Swap Rate vs CMS Rate")
    print("=" * 70)
    print(df.to_string(
        float_format=lambda x: "{:.4f}".format(x),
        formatters={
            "Fwd Swap Rate (%)":  lambda x: "{:.4f}%".format(x),
            "CMS Rate (%)":       lambda x: "{:.4f}%".format(x),
            "Conv. Adj. (bps)":   lambda x: "{:.3f}".format(x),
            "Adj / Fwd (%)":      lambda x: "{:.3f}%".format(x),
        }
    ))
    return df


def plot_convexity_adjustments(save_path=None):
    """
    Two-panel plot:
      Left:  Convexity adj vs tenor  (fixed expiry)
      Right: Convexity adj vs expiry (fixed tenor)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tenors_p  = [1, 2, 3, 5, 10]
    expiries_p = [1, 5, 10]
    exp_range  = [1, 2, 3, 5, 7, 10]

    # Left: adj vs tenor
    ax = axes[0]
    for Te in expiries_p:
        adjs = []
        for Tn in tenors_p:
            p = get_sabr_params(float(Te), float(Tn))
            _, adj, _ = cms_conv_adj(float(Te), float(Tn), float(Te), p)
            adjs.append(adj * 1e4)
        ax.plot(tenors_p, adjs, marker="o", lw=2, label="{:.0f}Y expiry".format(Te))
    ax.set_xlabel("CMS Tenor (years)")
    ax.set_ylabel("Convexity Adjustment (bps)")
    ax.set_title("Convexity Correction vs Tenor\n(fixed expiry)")
    ax.legend()

    # Right: adj vs expiry
    ax = axes[1]
    for Tn in [1, 5, 10]:
        adjs = []
        for Te in exp_range:
            p = get_sabr_params(float(Te), float(Tn))
            _, adj, _ = cms_conv_adj(float(Te), float(Tn), float(Te), p)
            adjs.append(adj * 1e4)
        ax.plot(exp_range, adjs, marker="s", lw=2, label="{:.0f}Y tenor".format(Tn))
    ax.set_xlabel("Swaption Expiry (years)")
    ax.set_ylabel("Convexity Adjustment (bps)")
    ax.set_title("Convexity Correction vs Expiry\n(fixed tenor)")
    ax.legend()

    fig.suptitle(
        "Part P3 — CMS Convexity Adjustment\n"
        "(Multi-Curve: OIS discounting | Term SOFR projection | SABR from Part II)",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def print_summary(pv1, pv2):
    print("=" * 70)
    print(" Part III — Complete")
    print("=" * 70)
    print("\n  P1  CMS 10Y semi-annual (5yr leg):   PV = {:.4f} bps".format(pv1 * 1e4))
    print("  P2  CMS 2Y  quarterly  (10yr leg):  PV = {:.4f} bps".format(pv2 * 1e4))
    print("\n  P3   Convexity corrections computed for 6 (expiry, tenor) pairs")
    print("          Tenor effect:  longer tenor  → larger adjustment (more IRR concavity)")
    print("          Expiry effect: longer expiry → larger adjustment (more rate variance)")
    print("\n  Dependencies:")
    print("    Part 1: OIS discount curve, Term SOFR projection curve")
    print("    Part 2: SABR calibrated parameters, forward swap rates, Black-76 pricing")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # III.1 — CMS leg PVs
    pv1, _ = price_cms_10y_semiannual()
    print()
    pv2, _ = price_cms_2y_quarterly()

    # III.1 summary
    print("\n" + "=" * 50)
    print("  III.1 — CMS Leg PV Summary")
    print("=" * 50)
    print("  CMS 10Y semi-annual (5yr):   PV = {:.4f} bps".format(pv1 * 1e4))
    print("  CMS 2Y  quarterly  (10yr):  PV = {:.4f} bps".format(pv2 * 1e4))
    print("=" * 50)

    # P3 — forward vs CMS comparison
    compare_fwd_vs_cms()

    # Plots
    plot_convexity_adjustments()

    # Final summary
    print_summary(pv1, pv2)
