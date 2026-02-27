import argparse
from pathlib import Path
import sys
from typing import Iterable, List
import numpy as np
import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm
import wrds

def interpolate_rates(df: pl.DataFrame, rate_cols: List[str], tenors: List[int]) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col(col).cast(pl.Float64, strict=False).fill_null(strategy="forward").alias(col)
            for col in rate_cols
        ]
    )
    rate_values = df.select(rate_cols).to_numpy() / 100.0

    T_days = df["T"].to_numpy()
    n_rows = df.height
    interpolated_r = np.zeros(n_rows)

    interpolated_r[:] = rate_values[:, 0]

    for i in range(len(tenors) - 1):
        t_low = tenors[i]
        t_high = tenors[i + 1]

        # Mask for rows where T is between these two tenors
        mask = (T_days >= t_low) & (T_days < t_high)

        if np.any(mask):
            r_low = rate_values[mask, i]
            r_high = rate_values[mask, i + 1]
            dt = T_days[mask]

            # Linear Interpolation formula: y = y0 + (y1-y0) * (x-x0)/(x1-x0)
            fraction = (dt - t_low) / (t_high - t_low)
            interpolated_r[mask] = r_low + (r_high - r_low) * fraction

    mask_long = T_days >= tenors[-1]
    if np.any(mask_long):
        interpolated_r[mask_long] = rate_values[mask_long, -1]

    df = df.with_columns(pl.Series(name="rate", values=interpolated_r))
    return df


def _print_progress(prefix: str, step: int, total: int) -> None:
    if total <= 0 or not sys.stderr.isatty():
        return
    width = 30
    filled = min(width, int(width * step / total))
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r{prefix} [{bar}] {step}/{total}")
    if step >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _bs_price_vectorized(
    sigma: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    rf: np.ndarray,
    div: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (rf - div + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    discounted_s = s * np.exp(-div * t)
    discounted_k = k * np.exp(-rf * t)
    call_price = discounted_s * norm.cdf(d1) - discounted_k * norm.cdf(d2)
    put_price = discounted_k * norm.cdf(-d2) - discounted_s * norm.cdf(-d1)
    return np.where(is_call, call_price, put_price)


def _solve_implied_vols(
    S: np.ndarray,
    K: np.ndarray,
    tau: np.ndarray,
    price: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    cp: np.ndarray,
    show_progress: bool = True,
    progress_prefix: str = "IV",
) -> np.ndarray:
    sigma_low = 1e-6
    sigma_high = 10.0
    tol = 1e-8
    max_iter = 100

    n_rows = S.shape[0]
    sigma = np.full(n_rows, np.nan, dtype=np.float64)

    is_call = cp == "C"
    is_put = cp == "P"
    valid_inputs = (
        np.isfinite(S)
        & np.isfinite(K)
        & np.isfinite(tau)
        & np.isfinite(price)
        & (S > 0.0)
        & (K > 0.0)
        & (tau > 0.0)
        & (price > 0.0)
        & (is_call | is_put)
    )
    if not np.any(valid_inputs):
        return sigma

    discounted_s = S * np.exp(-q * tau)
    discounted_k = K * np.exp(-r * tau)
    lower_bound = np.where(is_call, np.maximum(0.0, discounted_s - discounted_k), np.maximum(0.0, discounted_k - discounted_s))
    upper_bound = np.where(is_call, discounted_s, discounted_k)

    solve_mask = valid_inputs & (price >= lower_bound - 1e-8) & (price <= upper_bound + 1e-8)
    solve_idx = np.where(solve_mask)[0]
    if solve_idx.size == 0:
        return sigma

    s = S[solve_idx]
    k = K[solve_idx]
    t = tau[solve_idx]
    rf = r[solve_idx]
    div = q[solve_idx]
    px = price[solve_idx]
    call_flag = is_call[solve_idx]

    lo = np.full(solve_idx.size, sigma_low, dtype=np.float64)
    hi = np.full(solve_idx.size, sigma_high, dtype=np.float64)

    f_lo = _bs_price_vectorized(lo, s, k, t, rf, div, call_flag) - px
    f_hi = _bs_price_vectorized(hi, s, k, t, rf, div, call_flag) - px

    finite = np.isfinite(f_lo) & np.isfinite(f_hi)
    bracketed = finite & (f_lo * f_hi <= 0.0)

    local_sigma = np.full(solve_idx.size, np.nan, dtype=np.float64)
    at_lo = bracketed & (np.abs(f_lo) <= tol)
    at_hi = bracketed & (np.abs(f_hi) <= tol)
    local_sigma[at_lo] = lo[at_lo]
    local_sigma[at_hi] = hi[at_hi]

    active = bracketed & np.isnan(local_sigma)
    if np.any(active):
        lo_a = lo[active]
        hi_a = hi[active]
        f_lo_a = f_lo[active]
        s_a = s[active]
        k_a = k[active]
        t_a = t[active]
        rf_a = rf[active]
        div_a = div[active]
        px_a = px[active]
        call_a = call_flag[active]

        if show_progress:
            _print_progress(f"{progress_prefix} bisection", 0, max_iter)

        last_step = 0
        for step in range(1, max_iter + 1):
            mid = 0.5 * (lo_a + hi_a)
            f_mid = _bs_price_vectorized(mid, s_a, k_a, t_a, rf_a, div_a, call_a) - px_a
            left = f_lo_a * f_mid <= 0.0
            hi_a = np.where(left, mid, hi_a)
            lo_a = np.where(left, lo_a, mid)
            f_lo_a = np.where(left, f_lo_a, f_mid)
            last_step = step

            if show_progress:
                _print_progress(f"{progress_prefix} bisection", step, max_iter)

            if np.max(hi_a - lo_a) <= tol:
                break

        if show_progress and last_step < max_iter:
            _print_progress(f"{progress_prefix} bisection", max_iter, max_iter)

        local_sigma[active] = 0.5 * (lo_a + hi_a)

    unresolved_local = np.where(np.isnan(local_sigma))[0]
    if unresolved_local.size > 0:
        if show_progress:
            _print_progress(f"{progress_prefix} brentq fallback", 0, unresolved_local.size)

        for step, local_idx in enumerate(unresolved_local, start=1):
            s_i = s[local_idx]
            k_i = k[local_idx]
            t_i = t[local_idx]
            rf_i = rf[local_idx]
            div_i = div[local_idx]
            px_i = px[local_idx]
            is_call_i = bool(call_flag[local_idx])

            def objective(vol: float) -> float:
                value = _bs_price_vectorized(
                    np.array([vol]),
                    np.array([s_i]),
                    np.array([k_i]),
                    np.array([t_i]),
                    np.array([rf_i]),
                    np.array([div_i]),
                    np.array([is_call_i]),
                )[0]
                return value - px_i

            try:
                f_lo_i = objective(sigma_low)
                hi_i = sigma_high
                f_hi_i = objective(hi_i)
                while np.isfinite(f_lo_i) and np.isfinite(f_hi_i) and f_lo_i * f_hi_i > 0.0 and hi_i < 20.0:
                    hi_i *= 2.0
                    f_hi_i = objective(hi_i)
                if np.isfinite(f_lo_i) and np.isfinite(f_hi_i) and f_lo_i * f_hi_i <= 0.0:
                    local_sigma[local_idx] = brentq(objective, sigma_low, hi_i, xtol=tol, rtol=tol, maxiter=100)
            except ValueError:
                pass

            if show_progress:
                _print_progress(f"{progress_prefix} brentq fallback", step, unresolved_local.size)

    sigma[solve_idx] = local_sigma
    return sigma


def add_black_scholes_greeks(df: pl.DataFrame, show_progress: bool = True) -> pl.DataFrame:
    """
    Add Black-Scholes Greeks for European options using implied volatility.
    Theta is returned as per-calendar-day decay.
    """
    n_rows = df.height

    S = df["S"].cast(pl.Float64, strict=False).to_numpy()
    K = df["K"].cast(pl.Float64, strict=False).to_numpy()
    tau = df["T"].cast(pl.Float64, strict=False).to_numpy() / 365.0
    price = df["Price"].cast(pl.Float64, strict=False).to_numpy().copy()
    r = np.nan_to_num(df["rate"].cast(pl.Float64, strict=False).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(
        df["dividend_yield"].cast(pl.Float64, strict=False).to_numpy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    cp = np.array([str(value).upper() if value is not None else "" for value in df["cp_flag"].to_list()])

    sigma = _solve_implied_vols(S, K, tau, price, r, q, cp, show_progress=show_progress, progress_prefix="Mid")

    if "Bid" in df.columns:
        bid = df["Bid"].cast(pl.Float64, strict=False).to_numpy()
        unresolved_idx = np.where(np.isnan(sigma))[0]
        if unresolved_idx.size > 0:
            sigma_bid = _solve_implied_vols(
                S[unresolved_idx],
                K[unresolved_idx],
                tau[unresolved_idx],
                bid[unresolved_idx],
                r[unresolved_idx],
                q[unresolved_idx],
                cp[unresolved_idx],
                show_progress=show_progress,
                progress_prefix="Bid",
            )
            solved_bid = np.isfinite(sigma_bid)
            if np.any(solved_bid):
                chosen_idx = unresolved_idx[solved_bid]
                sigma[chosen_idx] = sigma_bid[solved_bid]
                price[chosen_idx] = bid[chosen_idx]

    if "Ask" in df.columns:
        ask = df["Ask"].cast(pl.Float64, strict=False).to_numpy()
        unresolved_idx = np.where(np.isnan(sigma))[0]
        if unresolved_idx.size > 0:
            sigma_ask = _solve_implied_vols(
                S[unresolved_idx],
                K[unresolved_idx],
                tau[unresolved_idx],
                ask[unresolved_idx],
                r[unresolved_idx],
                q[unresolved_idx],
                cp[unresolved_idx],
                show_progress=show_progress,
                progress_prefix="Ask",
            )
            solved_ask = np.isfinite(sigma_ask)
            if np.any(solved_ask):
                chosen_idx = unresolved_idx[solved_ask]
                sigma[chosen_idx] = sigma_ask[solved_ask]
                price[chosen_idx] = ask[chosen_idx]

    delta = np.full(n_rows, np.nan, dtype=np.float64)
    gamma = np.full(n_rows, np.nan, dtype=np.float64)
    theta = np.full(n_rows, np.nan, dtype=np.float64)

    valid = (
        np.isfinite(S)
        & np.isfinite(K)
        & np.isfinite(tau)
        & np.isfinite(sigma)
        & (S > 0.0)
        & (K > 0.0)
        & (tau > 0.0)
        & (sigma > 0.0)
    )

    if np.any(valid):
        S_v = S[valid]
        K_v = K[valid]
        tau_v = tau[valid]
        sigma_v = sigma[valid]
        r_v = r[valid]
        q_v = q[valid]
        is_call = cp[valid] == "C"

        sqrt_tau = np.sqrt(tau_v)
        d1 = (np.log(S_v / K_v) + (r_v - q_v + 0.5 * sigma_v**2) * tau_v) / (sigma_v * sqrt_tau)
        d2 = d1 - sigma_v * sqrt_tau

        nd1 = norm.pdf(d1)
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)

        exp_q = np.exp(-q_v * tau_v)
        exp_r = np.exp(-r_v * tau_v)

        delta_v = np.where(is_call, exp_q * Nd1, exp_q * (Nd1 - 1.0))
        gamma_v = exp_q * nd1 / (S_v * sigma_v * sqrt_tau)

        theta_call = (
            -(S_v * exp_q * nd1 * sigma_v) / (2.0 * sqrt_tau)
            - r_v * K_v * exp_r * Nd2
            + q_v * S_v * exp_q * Nd1
        )
        theta_put = (
            -(S_v * exp_q * nd1 * sigma_v) / (2.0 * sqrt_tau)
            + r_v * K_v * exp_r * N_neg_d2
            - q_v * S_v * exp_q * N_neg_d1
        )
        theta_v = np.where(is_call, theta_call, theta_put) / 365.0

        delta[valid] = delta_v
        gamma[valid] = gamma_v
        theta[valid] = theta_v

    return df.with_columns(
        pl.Series(name="Price", values=price),
        pl.Series(name="Impl_Vol", values=sigma),
        pl.Series(name="delta", values=delta),
        pl.Series(name="gamma", values=gamma),
        pl.Series(name="theta", values=theta),
    )


def download_option_data(secid: str = "108105", years: Iterable[int] = (), option_type: str = "C") -> None:
    db = wrds.Connection()

    tenor_days = [30, 91, 182, 365, 730, 1095, 1825]
    rate_columns = ["dgs1mo", "dgs3mo", "dgs6mo", "dgs1", "dgs2", "dgs3", "dgs5"]

    for year in years:
        print(f"Connecting to WRDS to fetch {secid} for year {year}...")

        print(f"Fetching Yield Curve for {year}...")
        rates_df = db.raw_sql(f"""
                        SELECT 
                            date,
                            dgs1mo, dgs3mo, dgs6mo, dgs1, dgs2, dgs3, dgs5
                        FROM 
                            frb.rates_daily
                        WHERE 
                            date >= '{year}-01-01' AND date <= '{year}-12-31'
                    """)
        rates_df = pl.from_pandas(rates_df).with_columns(pl.col("date").cast(pl.Date))

        sql_query = f"""
                    SELECT
                        o.date,
                        o.exdate,
                        o.cp_flag,
                        o.strike_price / 1000.0 as strike,
                        o.best_bid as bid_price,
                        o.best_offer as ask_price,
                        (o.best_bid + o.best_offer) / 2.0 as option_price,
                        o.impl_volatility,
                        s.close as spot_price,
                        inf.exercise_style,
                        d.rate / 100.0 as dividend_yield,
                        c.vix,
                        v.hv_10,
                        v.hv_14,
                        v.hv_30,
                        v.hv_60,
                        v.hv_91,
                        f.sofr as sofr
                    FROM
                        optionm.opprcd{year} as o
                    LEFT JOIN
                        optionm.secprd as s
                        ON o.date = s.date AND o.secid = s.secid
                    LEFT JOIN
                        optionm.idxdvd as d
                        ON o.secid = d.secid 
                        AND o.date = d.date 
                        AND o.exdate = d.expiration
                    JOIN
                        cboe.cboe as c
                        ON c.date = s.date
                    JOIN (
                        SELECT 
                            date, 
                            secid,
                            MAX(CASE WHEN days = 10 THEN volatility END) as hv_10,
                            MAX(CASE WHEN days = 14 THEN volatility END) as hv_14,
                            MAX(CASE WHEN days = 30 THEN volatility END) as hv_30,
                            MAX(CASE WHEN days = 60 THEN volatility END) as hv_60,
                            MAX(CASE WHEN days = 91 THEN volatility END) as hv_91
                        FROM 
                            optionm.hvold{year}
                        WHERE 
                            secid = '{secid}'
                        GROUP BY 
                            date, secid
                    ) as v
                        ON o.date = v.date AND o.secid = v.secid
                    JOIN 
                        optionm.opinfd as inf
                        ON o.secid = inf.secid
                    LEFT JOIN
                        frb.rates_daily as f
                        ON o.date = f.date
                    WHERE
                        s.secid = '{secid}'
                        AND o.cp_flag = '{option_type}'
                        AND o.best_bid > 0.1 
                        AND o.volume > 1
                        AND inf.exercise_style = 'E'
                    """

        try:
            df = db.raw_sql(sql_query)
        except Exception as exc:
            print(f"Query failed for: {exc}")
            print("If needed, retry with a different --table-prefix.")
            continue

        print("Processing data...")
        df = (
            pl.from_pandas(df)
            .with_columns(
                pl.col("date").cast(pl.Date),
                pl.col("exdate").cast(pl.Date),
            )
            .with_columns((pl.col("exdate") - pl.col("date")).dt.total_days().alias("T"))
            .filter(pl.col("T") > 1)
        )

        df = df.join(rates_df, on="date", how="left")

        df = interpolate_rates(df, rate_columns, tenor_days)

        final_df = (
            df.select(
                [
                "date",
                "spot_price",
                "strike",
                "T",
                "vix",
                "bid_price",
                "ask_price",
                "option_price",
                "impl_volatility",
                "cp_flag",
                "exercise_style",
                "dividend_yield",
                "rate"
                ]
            )
            .rename(
                {
                    "spot_price": "S",
                    "strike": "K",
                    "bid_price": "Bid",
                    "ask_price": "Ask",
                    "option_price": "Price",
                    "impl_volatility": "Impl_Vol",
                }
            )
        )
        final_df = add_black_scholes_greeks(final_df).sort(["date", "K", "T"])

        final_df = final_df.drop_nans()

        path = Path(f"./data/{secid.lower()}/{year}_{option_type}_options_data.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_csv(path)
        print(f"Downloaded {final_df.height} rows. Saved to {path}")

    db.close()


def _parse_years(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="WRDS downloader for options and volatility surfaces.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_options = subparsers.add_parser("options", help="Download option chain-level records.")
    p_options.add_argument("--secid", default="108105")
    p_options.add_argument("--years", required=True, help="Comma-separated list, e.g. 2023,2024")
    p_options.add_argument("--cp-flag", default="C", choices=["C", "P"])

    args = parser.parse_args()
    years = _parse_years(args.years)

    if args.mode == "options":
        download_option_data(secid=args.secid, years=years, option_type=args.cp_flag)
        return

if __name__ == "__main__":
    main()
