#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 16:57:20 2026

@author: mirocastelein
"""

# test to see if git works

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from financepy.models.black_scholes_analytic import (
    bs_value,
    bs_delta
)
from financepy.utils.global_types import OptionTypes


# Parameters
S = 100.0      # spot
K = 100.0      # strike
T = 1.0        # maturity (years)
r = 0.04       # risk-free rate
q = 0.0        # dividend yield
sigma = 0.2    # volatility

#==============================================================================
#Q1.b
#==============================================================================

def DeltaHedge(K, S, r, mu, sigma, sigmaReal, T, N, seed=None):
    """
    Simulate delta hedging of a European PUT using a self-financing portfolio.

    Parameters
    ----------
    K : float
        Strike
    S : float
        Initial spot price S(0)
    r : float
        Continuously compounded risk-free rate
    mu : float
        Real-world drift of the stock (NOT necessarily r)
    sigma : float
        Volatility used for BS pricing/delta (hedging/implied vol)
    sigmaReal : float
        Vol used for simulating the stock path (realized/true vol)
    T : float
        Years to expiry
    N : int
        Hedging frequency per year (re-hedge every 1/N years)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    (S_T, payoff, realized_variance, replication_error) : tuple
        S_T : terminal stock price
        payoff : option payoff max(K - S_T, 0)
        realized_variance : annualized realized variance of log returns over [0,T]
        replication_error : (hedge portfolio value at T) - payoff
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigmaReal < 0 or sigma < 0:
        raise ValueError("Volatilities must be non-negative.")

    rng = np.random.default_rng(seed)

    steps = int(round(T * N))
    # Ensure at least 1 step
    steps = max(1, steps)
    dt = T / steps

    # --- Initial option value and delta (European PUT) ---
    # Price
    v = bs_value(S, T, K, r, q, sigma,
                 OptionTypes.EUROPEAN_PUT.value)

    # Delta
    delta = bs_delta(S, T, K, r, q, sigma,
                     OptionTypes.EUROPEAN_PUT.value)

    print("Price:", v)
    print("Delta:", delta)
    
    #To avoid typing out "OptionTypes.EUROPEAN_CALL.value" over and over again
    put_type = OptionTypes.EUROPEAN_PUT.value


    # Self-financing hedge portfolio:
    # Hold delta shares, rest in cash account.
    # Portfolio value equals option value initially:
    cash_psi = v - delta * S

    # For realized variance calculation
    sum_sq_log_returns = 0.0
    S_t = float(S)

    for i in range(1, steps + 1):
        # 1) Calculate option valua taking into account the shorter tau= (T - t) 
        ## [Eliminate risk]
        ## (Recompute delta using BS with remaining time and hedging vol sigma)
        t = i * dt
        tau = max(T - t, 0.0)
        
        # 2) Calculate delta (t) using the BS equation with the new value of S(t)
        ## (Evolve stock under real-world dynamics (GBM with mu, sigmaReal)
        eps = rng.standard_normal()
        S_next = S_t * math.exp((mu - 0.5 * sigmaReal**2) * dt + sigmaReal * eps * math.sqrt(dt))

        # log return for realized variance
        log_ret = math.log(S_next / S_t)
        sum_sq_log_returns += log_ret * log_ret

        # 3) B(t) has roled 1 day at risk-free rate; B(t) = B(t - dt) * exp(r dt)
        ## (Cash accrues risk-free over dt)
        cash_psi *= math.exp(r * dt)


        # At expiry tau=0, delta is not needed anymore; avoid potential division issues.
        if tau > 0.0:
            new_delta = bs_delta(S_next, tau, K, r, q, sigma, put_type)
        else:
            new_delta = 0.0
            

        # 4) Solve for the new cash position by keeping the hedge self-financing
        ## Self-financing rebalance at price S_next:
        # Buy/sell (new_delta - old_delta) shares, financed from cash.
        cash_psi -= (new_delta - delta) * S_next
        delta = new_delta

        #Update
        S_t = S_next

    #Terminal stock & pay-off
    S_T = S_t
    payoff = max(K - S_T, 0.0)

    # Terminal hedge portfolio value
    portfolio_T = delta * S_T + cash_psi

    # Annualized realized variance over [0,T]
    realized_variance = sum_sq_log_returns / T

    #Replication error
    replication_error = portfolio_T - payoff

    #Tuple with 4 elements:
    return S_T, payoff, realized_variance, replication_error




#==============================================================================
#Q1.c
#==============================================================================

def RunDeltaHedgeMC(K, S, r, mu, sigma, sigmaReal, T, N, n_paths=10_000, seed=123):
    """
    Monte Carlo wrapper around DeltaHedge to generate hedging errors over many paths.

    Parameters
    ----------
    (same as DeltaHedge) plus:
    n_paths : int
        Number of Monte Carlo simulation paths (default 10,000)
    seed : int
        Base seed for reproducibility

    Returns
    -------
    results : dict
        Dictionary containing arrays for:
            - "S_T" : terminal stock prices
            - "payoff" : option payoffs
            - "realized_variance" : realized variances
            - "hedging_error" : replication errors
        and summary stats for hedging error:
            - "error_mean", "error_std", "error_q05", "error_median", "error_q95"
    """
    rng = np.random.default_rng(seed)

    S_T_arr = np.empty(n_paths, dtype=float)
    payoff_arr = np.empty(n_paths, dtype=float)
    rvar_arr = np.empty(n_paths, dtype=float)
    err_arr = np.empty(n_paths, dtype=float)

    # Use independent seeds per path (reproducible but different paths)
    path_seeds = rng.integers(0, 2**32 - 1, size=n_paths, dtype=np.uint32)

    for i in range(n_paths):
        S_T, payoff, rv, err = DeltaHedge(
            K=K, S=S, r=r, mu=mu,
            sigma=sigma, sigmaReal=sigmaReal,
            T=T, N=N, seed=int(path_seeds[i])
        )
        S_T_arr[i] = S_T
        payoff_arr[i] = payoff
        rvar_arr[i] = rv
        err_arr[i] = err

    results = {
        "S_T": S_T_arr,
        "payoff": payoff_arr,
        "realized_variance": rvar_arr,
        "hedging_error": err_arr,
        # Summary stats for hedging error
        "error_mean": float(np.mean(err_arr)),
        "error_std": float(np.std(err_arr, ddof=1)),
        "error_q05": float(np.quantile(err_arr, 0.05)),
        "error_median": float(np.quantile(err_arr, 0.50)),
        "error_q95": float(np.quantile(err_arr, 0.95)),
    }
    return results




#==============================================================================
#Q1.d
#==============================================================================

# ---------- Vectorized normal CDF ----------
def norm_cdf(x):
    return norm.cdf(x)

# ---------- Vectorized BS put delta ----------
def bs_put_delta_vec(S, tau, K, r, sigma, q):
    """
    Vectorized Black–Scholes delta for European put.
    S: array
    tau: scalar (time to maturity)
    """
    #Delta in case of expiry
    if tau <= 0.0:
        # At expiry: delta is piecewise. Use a common convention.
        ## If S < K (put is in the money) --> Delta = -1
        ## If S > K (put is out of the money) --> Delta = 0
        # (This doesn't matter much because we set delta=0 at expiry anyway.)
        return np.where(S < K, -1.0, 0.0)

    # Usual BS delta
    # Delta put: e^{-qT}(N(d1)-1)
    sqrtT = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrtT)
    return np.exp(-q * tau) * (norm_cdf(d1) - 1.0)

# ---------- Only for time 0: Vectorized BS put value ----------
def bs_put_value_and_delta0_vec(S0, T, K, r, sigma, q):
    """
    Value and delta at inception for a put.
    Returns scalars (value, delta) for scalar S0.
    """
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    value = K * disc_r * norm_cdf(-d2) - S0 * disc_q * norm_cdf(-d1)
    delta = disc_q * (norm_cdf(d1) - 1.0)
    return float(value), float(delta)

# ---------- Main function: Fast multi-path delta hedge ----------
def hedge_errors_vectorized(
    K, S0, r, mu, sigma, sigmaReal, T, N,
    n_paths=10_000, seed=123, q=0.0
):
    """
    Vectorized delta-hedging simulation for many paths at once.

    Returns:
      S_T: (n_paths,) terminal stock prices
      err: (n_paths,) hedging errors (portfolio_T - payoff)
    """
    rng = np.random.default_rng(seed)

    steps = max(1, int(round(T * N)))
    dt = T / steps

    # Initial option value and delta (scalars)
    V0, delta0 = bs_put_value_and_delta0_vec(S0, T, K, r, sigma, q=q)

    # Path state vectors
    # Initially, every path starts at the same S0
    S = np.full(n_paths, S0, dtype=float)
    delta = np.full(n_paths, delta0, dtype=float)
    cash = np.full(n_paths, V0 - delta0 * S0, dtype=float)  # self-financing initial cash

    # precompute cash growth per step
    cash_growth = np.exp(r * dt)

    for i in range(1, steps + 1):
        # simulate one time step for all paths
        Z = rng.standard_normal(n_paths)
        S_next = S * np.exp((mu - 0.5 * sigmaReal**2) * dt + sigmaReal * np.sqrt(dt) * Z)

        # cash accrues at risk-free
        cash *= cash_growth

        # time to maturity after this step
        tau = max(T - i * dt, 0.0)

        if tau > 0.0:
            new_delta = bs_put_delta_vec(S_next, tau, K, r, sigma, q=q)
        else:
            new_delta = np.zeros_like(delta)

        # self-financing rebalance: buy/sell (new_delta - old_delta) shares
        cash -= (new_delta - delta) * S_next
        delta = new_delta
        S = S_next

    S_T = S
    payoff = np.maximum(K - S_T, 0.0)
    portfolio_T = delta * S_T + cash
    err = portfolio_T - payoff
    return S_T, err

# ---------- Parameters ----------
S0 = 100.0
K  = 100.0
r  = 0.04
T  = 1.0
sigma = 0.20
mu = 0.05
sigmaReal = sigma      
n_paths = 10_000

# ---------- Run and plot ----------
S12,  e12  = hedge_errors_vectorized(K, S0, r, mu, sigma, sigmaReal, T, N=12,  n_paths=n_paths, seed=1)
S52,  e52  = hedge_errors_vectorized(K, S0, r, mu, sigma, sigmaReal, T, N=52,  n_paths=n_paths, seed=2)
S252, e252 = hedge_errors_vectorized(K, S0, r, mu, sigma, sigmaReal, T, N=252, n_paths=n_paths, seed=3)

plt.figure(figsize=(9, 6))
plt.scatter(S12,  e12,  s=10, alpha=0.35, marker='o', color='red', label='N=12 (monthly)')
plt.scatter(S52,  e52,  s=10, alpha=0.35, marker='^', color='yellow', label='N=52 (weekly)')
plt.scatter(S252, e252, s=10, alpha=0.35, marker='s', color='green', label='N=252 (daily)')
plt.axhline(0.0, linewidth=1)
plt.xlabel("Terminal stock price $S(T)$")
plt.ylabel("Hedging error (portfolio value at T − payoff)")
plt.title("Delta-hedging error vs terminal stock price (European put)")
plt.legend()
plt.tight_layout()
plt.show()




#Still need to clean

#==============================================================================
#Q1.e
#==============================================================================

# ---------- Black–Scholes (European PUT) ----------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_value_and_delta(S, T, K, r, sigma):
    if T <= 0:
        value = max(K - S, 0.0)
        delta = -1.0 if S < K else (0.0 if S > K else -0.5)
        return value, delta

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    value = K * math.exp(-r * T) * norm_cdf(-d2) \
            - S * norm_cdf(-d1)

    delta = norm_cdf(d1) - 1.0

    return value, delta


# ---------- Single-path hedge ----------

def DeltaHedge(K, S0, r, mu, sigma, sigmaReal, T, N, seed=None):

    rng = np.random.default_rng(seed)

    steps = max(1, int(T * N))
    dt = T / steps

    V0, delta = bs_put_value_and_delta(S0, T, K, r, sigma)
    cash = V0 - delta * S0
    S = S0

    for i in range(1, steps + 1):

        Z = rng.standard_normal()
        S_new = S * math.exp((mu - 0.5 * sigmaReal**2) * dt
                             + sigmaReal * math.sqrt(dt) * Z)

        cash *= math.exp(r * dt)

        tau = max(T - i * dt, 0.0)

        if tau > 0:
            _, new_delta = bs_put_value_and_delta(S_new, tau, K, r, sigma)
        else:
            new_delta = 0.0

        cash -= (new_delta - delta) * S_new

        delta = new_delta
        S = S_new

    payoff = max(K - S, 0.0)
    portfolio = delta * S + cash

    return portfolio - payoff


# ---------- Monte Carlo stats ----------

def HedgeErrorStats(K, S0, r, mu, sigma, sigmaReal, T, N, n_paths=10000):

    errors = np.empty(n_paths)

    for i in range(n_paths):
        errors[i] = DeltaHedge(K, S0, r, mu,
                               sigma, sigmaReal, T, N)

    mean_error = np.mean(errors)
    var_error = np.var(errors, ddof=1)

    return mean_error, var_error


# ---------- Parameters ----------

S0 = 100
K = 100
r = 0.04
T = 1.0
sigma = 0.20
sigmaReal = 0.20
mu = 0.05

for N in [12, 52, 252]:
    mean_err, var_err = HedgeErrorStats(K, S0, r, mu,
                                        sigma, sigmaReal, T, N)

    print(f"N = {N:3d} | Mean error = {mean_err:.6f} | Variance = {var_err:.6f}")








#==============================================================================
#Q1.f
#==============================================================================

# ---------- Black–Scholes (European PUT) ----------
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_put_value_and_delta(S, T, K, r, sigma):
    if T <= 0:
        value = max(K - S, 0.0)
        delta = -1.0 if S < K else (0.0 if S > K else -0.5)
        return value, delta

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    value = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    delta = norm_cdf(d1) - 1.0
    return value, delta


# ---------- Delta hedge returning realized vol + replication error ----------
def DeltaHedge(K, S0, r, mu, sigma, sigmaReal, T, N, seed=None):
    rng = np.random.default_rng(seed)

    steps = max(1, int(T * N))
    dt = T / steps

    V0, delta = bs_put_value_and_delta(S0, T, K, r, sigma)
    cash = V0 - delta * S0
    S = S0

    sum_sq_log_returns = 0.0

    for i in range(1, steps + 1):
        Z = rng.standard_normal()
        S_new = S * math.exp((mu - 0.5 * sigmaReal**2) * dt +
                             sigmaReal * math.sqrt(dt) * Z)

        log_ret = math.log(S_new / S)
        sum_sq_log_returns += log_ret**2

        cash *= math.exp(r * dt)

        tau = max(T - i * dt, 0.0)
        if tau > 0:
            _, new_delta = bs_put_value_and_delta(S_new, tau, K, r, sigma)
        else:
            new_delta = 0.0

        cash -= (new_delta - delta) * S_new
        delta = new_delta
        S = S_new

    payoff = max(K - S, 0.0)
    portfolio = delta * S + cash

    realized_var = sum_sq_log_returns / T
    realized_vol = math.sqrt(realized_var)
    replication_error = portfolio - payoff

    return realized_vol, replication_error


# ---------- Run many paths + scatter plot ----------
def ScatterRealizedVolVsError(K, S0, r, mu, sigma, sigmaReal, T, N, n_paths=10_000, seed=123):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=n_paths, dtype=np.uint32)

    realized_vols = np.empty(n_paths)
    errors = np.empty(n_paths)

    for i in range(n_paths):
        rv, err = DeltaHedge(K, S0, r, mu, sigma, sigmaReal, T, N, seed=int(seeds[i]))
        realized_vols[i] = rv
        errors[i] = err

    plt.figure(figsize=(8, 5))
    plt.scatter(realized_vols, errors, s=10, alpha=0.35)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Realized volatility")
    plt.ylabel("Replication error")
    plt.title(f"Realized Volatility vs Replication Error (N={N})")
    plt.tight_layout()
    plt.show()

    return realized_vols, errors


# ---------- Example parameters (from your question) ----------
S0 = 100
K = 100
r = 0.04
T = 1.0
sigma = 0.20
mu = 0.05
sigmaReal = 0.20
N = 52  # change to 12 / 52 / 252 if needed

ScatterRealizedVolVsError(K, S0, r, mu, sigma, sigmaReal, T, N)







#==============================================================================
#Q1.g
#==============================================================================

# ---------- Black–Scholes (European PUT) ----------
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_put_value_and_delta(S, T, K, r, sigma):
    if T <= 0:
        value = max(K - S, 0.0)
        delta = -1.0 if S < K else (0.0 if S > K else -0.5)
        return value, delta

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    value = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    delta = norm_cdf(d1) - 1.0
    return value, delta

# ---------- Single-path delta hedge: returns hedging error ----------
def DeltaHedge_error(K, S0, r, mu, sigma, sigmaReal, T, N, seed=None):
    rng = np.random.default_rng(seed)

    steps = max(1, int(T * N))
    dt = T / steps

    V0, delta = bs_put_value_and_delta(S0, T, K, r, sigma)
    cash = V0 - delta * S0
    S = S0

    for i in range(1, steps + 1):
        Z = rng.standard_normal()
        S_new = S * math.exp((mu - 0.5 * sigmaReal**2) * dt +
                             sigmaReal * math.sqrt(dt) * Z)

        cash *= math.exp(r * dt)

        tau = max(T - i * dt, 0.0)
        if tau > 0:
            _, new_delta = bs_put_value_and_delta(S_new, tau, K, r, sigma)
        else:
            new_delta = 0.0

        cash -= (new_delta - delta) * S_new
        delta = new_delta
        S = S_new

    payoff = max(K - S, 0.0)
    portfolio = delta * S + cash
    return portfolio - payoff

# ---------- Monte Carlo stats for a given mu ----------
def hedge_error_stats_for_mu(K, S0, r, mu, sigma, sigmaReal, T, N, n_paths=10_000, seed=123):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=n_paths, dtype=np.uint32)

    errors = np.empty(n_paths)
    for i in range(n_paths):
        errors[i] = DeltaHedge_error(K, S0, r, mu, sigma, sigmaReal, T, N, seed=int(seeds[i]))

    mean_abs_error = np.mean(np.abs(errors))
    variance_error = np.var(errors, ddof=1)
    mean_error = np.mean(errors)   # optional, useful as a check

    return mean_error, mean_abs_error, variance_error

# ---------- Run for the requested mus and print a table ----------
S0 = 100.0
K = 100.0
r = 0.04
T = 1.0
sigma = 0.20
sigmaReal = 0.20
N = 52
n_paths = 10_000

mus = [0.025, 0.05, 0.075, 0.10]

rows = []
for j, mu in enumerate(mus):
    mean_err, mae, var_err = hedge_error_stats_for_mu(
        K, S0, r, mu, sigma, sigmaReal, T, N,
        n_paths=n_paths, seed=1000 + j
    )
    rows.append({
        "mu": mu,
        "MeanAbsError": mae,
        "Variance(HedgingError)": var_err
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))







