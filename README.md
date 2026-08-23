# Fixed Income Securities: Swap and Swaption Markets

> Completed as part of the **QF605 Fixed Income Securities** course module (MSc Quantitative Finance).
> End-to-end Python implementation covering interest rate curve bootstrapping, volatility model calibration, and CMS derivative pricing.

---

## Project Overview

This project builds a complete interest rate derivatives pricing pipeline, starting from raw market quotes and ending with marked-to-market valuations of Constant Maturity Swap (CMS) products. The work is divided into three self-contained notebooks that chain together sequentially: each notebook imports and builds upon the outputs of its predecessor.

```
part1_bootstrapping_swap_curves.ipynb
        │
        └──► part2_swaption_calibration.ipynb
                        │
                        └──► part3_cms_valuation.ipynb
```

**Market data source:** `Swap and Swaption Markets.xlsx` (LIBOR legacy, OIS/SOFR, Term SOFR, and Swaption implied vol surfaces).

---

## Repository Structure

```
├── part1_bootstrapping_swap_curves.ipynb   # Discount curve construction
├── part2_swaption_calibration.ipynb        # DD & SABR vol model calibration
├── part3_cms_valuation.ipynb               # CMS pricing & convexity analysis
└── Swap and Swaption Markets.xlsx          # Input market data (required)
```

> **Important:** All three notebooks must be kept in the same directory as the Excel file. Parts 2 and 3 dynamically import the preceding notebooks as modules at runtime, so no manual copy-pasting of code is needed.

---

## Requirements

```bash
pip install numpy pandas matplotlib scipy openpyxl nbformat jupyter
```

The `jupyter` package provides the notebook interface and kernel needed to open and execute the `.ipynb` files (skip it if you already have JupyterLab, Anaconda, or VS Code with the Jupyter extension). Tested on Python 3.11. No GPU or external API dependencies.

---

## How to Run

1. Clone the repository and ensure `Swap and Swaption Markets.xlsx` is in the same folder as the notebooks.
2. Run the notebooks **in order**: Part 1, then Part 2, then Part 3.
3. Parts 2 and 3 will automatically import and execute the preceding notebook(s) via `import_notebook_as_module`. You do not need to re-run them manually each time, but the `.ipynb` files must be present in the working directory.
4. **Expected runtimes:** Part 1 runs in seconds, while Part 2 takes about 2 minutes (DD and SABR calibration across all slices). The first cell of Part 3 also takes about 2 minutes with no visible output while it silently re-executes Parts 1 and 2 in the background. This is normal, not a hang.

---

## Part I: Bootstrapping Swap Curves (`part1_bootstrapping_swap_curves.ipynb`)

**Goal:** Construct discount factor curves from raw par rate quotes using iterative bootstrapping.

| Section | What it does |
|---|---|
| Imports & Config | Sets file paths and loads libraries |
| Market Data Loading | Parses LIBOR, OIS (SOFR), and Term SOFR sheets from Excel; converts tenor strings (e.g. `"18m"`, `"2y"`) to decimal years |
| Bootstrapping Utilities | `df_interp` (linear interpolation on discount factors), `build_payment_schedule` (handles stub periods correctly) |
| OIS Bootstrapping | Iterative multi-pass bootstrap of `D_OIS(0,T)` from SOFR par rates; 5 refinement passes for convergence |
| LIBOR Single-Curve | Bootstrap assuming LIBOR discounts itself (pre-2008 convention) |
| LIBOR Multi-Curve | Separate projection and discounting curves (post-crisis, OIS-collateralised convention) |
| Term SOFR 3M Forwards | Bootstraps 3-month forward rates from Term SOFR par quotes |
| Results & Plots | Side-by-side comparison of single- vs multi-curve LIBOR; OIS discount curve; forward SOFR term structure |
| Summary Statistics | Key discount factors and forward rates at benchmark tenors |

**Key outputs:** `T_ois`, `DF_ois`, `fwd_t`, `fwd_r`, which are consumed by Parts 2 and 3.

---

## Part II: Swaption Calibration (`part2_swaption_calibration.ipynb`)

**Goal:** Fit two volatility smile models, Displaced Diffusion (DD) and SABR, to the market swaption vol surface, then reprice across the full strike range.

| Section | What it does |
|---|---|
| Imports & Config | Loads Part 1 via `import_notebook_as_module`; sets global constants (`BETA_SABR = 0.75`, strike offsets in bps) |
| Market Data Loading | Reads the Swaption sheet (expiry × tenor × 11 strike columns); displays the raw implied vol surface |
| Discount Callables | Wraps Part 1 bootstrap arrays into smooth callable functions `D_ois(t)` and `D_ts(t)` |
| Forward Swap Rates & Annuities | Computes ATM forward swap rates `F(T_exp, T_ten)` and annuity factors under the multi-curve framework |
| Synthetic 30×30 Smile | Constructs the 30Y tenor smile by interpolation (not directly quoted in market data) |
| Black-76 Pricing | `black_price` and `black_iv`, used as the reference model for converting quoted lognormal vols |
| **II.1 Displaced Diffusion** | Calibrates `(σ, β)` per slice; `β < 0` implies negative rates are possible. Tables of σ and β across all expiry/tenor combinations |
| **II.2 SABR Calibration** | Calibrates `(α, ρ, ν)` per slice with `β = 0.75` fixed. Tables of α (vol level), ρ (skew), ν (vol-of-vol) |
| **II.3 Smile Fit Plots** | Overlays DD and SABR model vols against market quotes for three representative swaptions: payer 1×1, payer 10×10, receiver 30×30. Full smile fit plots for all expiries |
| Summary Statistics | Calibration RMSE and forward rates at key nodes |

**Key outputs:** `swaption_slices` (with calibrated `sabr_params` per slice), `D_ois`, `D_ts`, which are consumed by Part 3.

---

## Part III: CMS Valuation (`part3_cms_valuation.ipynb`)

**Goal:** Price Constant Maturity Swap legs using the SABR-implied vol smile and replicate the convexity adjustment between forward swap rates and CMS rates.

| Section | What it does |
|---|---|
| Imports & Config | Imports both Part 1 and Part 2 as modules; verifies curve consistency |
| SABR Parameter Interpolation | Builds bilinear interpolators for `(α, ρ, ν)` across arbitrary `(T_exp, T_ten)` pairs, which is required since CMS pricing needs the smile at non-grid expiries |
| CMS Theory (Markdown) | Explains the convexity adjustment: why `E[S(T)] > F(T)` and the replication formula using OTM swaption prices |
| IRR Annuity Machinery | Implements the Internal Rate of Return annuity `A_IRR(S)` and its first/second derivatives, the core inputs to the convexity integral |
| **III.1a CMS 10Y Leg** | Values a leg receiving CMS10Y semi-annually over 5 years. Each payment date: computes convexity adjustment via numerical integration over the SABR smile |
| **III.1b CMS 2Y Leg** | Values a leg receiving CMS2Y quarterly over 10 years |
| III.1 Summary | PV table comparing CMS legs with naive forward swap rate valuations |
| **III.2 Convexity Analysis** | Tabulates forward swap rates vs CMS rates for six combinations: 1×1, 1×10, 5×1, 5×10, 10×1, 10×10. Plots convexity adjustment magnitude |
| Discussion (Markdown) | Explains why convexity corrections grow with payment lag (longer maturity) and with tenor (higher vol and longer annuity duration) |
| Summary | Final PV outputs and key numerical results |

---

## Concepts Covered

- **Curve construction:** OIS bootstrapping, single-curve vs multi-curve LIBOR, Term SOFR forward rates
- **Vol modelling:** Lognormal (Black-76), Displaced Diffusion, SABR (Hagan et al.)
- **Derivatives pricing:** Swaptions, Constant Maturity Swaps, convexity adjustments
- **Numerical methods:** Iterative bootstrapping, Brent root-finding, bilinear interpolation, numerical integration (quadrature)

---

## Notes

- OIS (SOFR) par rates are hardcoded in Part 1 (the original Excel sheet for this tab was image-based and not machine-readable). All other data is read directly from Excel.
- The 30×30 swaption smile is synthetic, constructed by interpolating across the available tenor grid, as this maturity is not directly quoted in the market data.
- `BETA_SABR = 0.75` is fixed across all slices per the project specification.
