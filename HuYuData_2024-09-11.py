import numpy as np
import scipy.stats as si


# Define the Black-Scholes formula
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S : float : current stock price
    K : float : strike price
    T : float : time to expiration (in years)
    r : float : risk-free interest rate
    sigma : float : volatility of the stock price
    option_type : str : 'call' for call option, 'put' for put option
    """

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # For Call Option
    if option_type == 'call':
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)

    # For Put Option
    elif option_type == 'put':
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)

    return price


# Parameters
S = 150  # Current stock price (e.g., $150)
K = 155  # Strike price (e.g., $155)
T = 0.5  # Time to expiration in years (e.g., 6 months = 0.5 years)
r = 0.02  # Risk-free interest rate (e.g., 2%)
sigma = 0.25  # Volatility of the stock (e.g., 25%)

# Calculate Call and Put Option prices
call_price = black_scholes(S, K, T, r, sigma, option_type='call')
put_price = black_scholes(S, K, T, r, sigma, option_type='put')

# Print the results
print(f"Call Option Price: {call_price:.2f}")
print(f"Put Option Price: {put_price:.2f}")