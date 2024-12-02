import numpy as np
from scipy.stats import norm

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Calculate the Black-Scholes option price.
    
    Parameters:
    -----------
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Implied volatility (annualized)
    option_type : str
        Option type ('C' for Call, 'P' for Put)
        
    Returns:
    --------
    float
        Option price according to Black-Scholes formula
    """
    
    # Input validation
    if not isinstance(option_type, str) or option_type not in ['C', 'P']:
        raise ValueError("option_type must be either 'C' for Call or 'P' for Put")
    if any(x <= 0 for x in [S, K, T, sigma]):
        raise ValueError("S, K, T, and sigma must be positive")
    
    # Handle edge case of very small time to expiration
    if T < 1e-10:
        if option_type == 'C':
            return max(0, S - K)
        else:
            return max(0, K - S)
    
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate Call or Put price
    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) + K * np.exp(-r * T) - S
    
    return price

def run_tests():
    """Run a series of tests to verify the Black-Scholes implementation."""
    
    # Test Case 1: At-the-money options with zero interest rate
    # Put and call should have similar prices
    S = 100
    K = 100
    T = 1
    r = 0
    sigma = 0.2
    call_price = black_scholes(S, K, T, r, sigma, 'C')
    put_price = black_scholes(S, K, T, r, sigma, 'P')
    print("\nTest 1: At-the-money options (r=0)")
    print(f"Call price: {call_price:.4f}")
    print(f"Put price: {put_price:.4f}")
    print(f"Difference: {abs(call_price - put_price):.4f}")
    assert abs(call_price - put_price) < 0.0001, "ATM put-call prices should be equal when r=0"

    # Test Case 2: Put-call parity
    # C - P = S - K*exp(-rT)
    r = 0.05
    call_price = black_scholes(S, K, T, r, sigma, 'C')
    put_price = black_scholes(S, K, T, r, sigma, 'P')
    parity_left = call_price - put_price
    parity_right = S - K * np.exp(-r * T)
    print("\nTest 2: Put-call parity")
    print(f"C - P: {parity_left:.4f}")
    print(f"S - Ke^(-rT): {parity_right:.4f}")
    print(f"Difference: {abs(parity_left - parity_right):.4f}")
    assert abs(parity_left - parity_right) < 0.0001, "Put-call parity violated"

    # Test Case 3: Deep ITM and OTM options
    S = 100
    K_values = [50, 150]
    print("\nTest 3: Deep ITM/OTM options")
    for K in K_values:
        call_price = black_scholes(S, K, T, r, sigma, 'C')
        put_price = black_scholes(S, K, T, r, sigma, 'P')
        print(f"\nStrike = {K}:")
        print(f"Call price: {call_price:.4f}")
        print(f"Put price: {put_price:.4f}")
        # Verify call price approaches S-K for deep ITM and 0 for deep OTM
        if K < S:
            assert abs(call_price - (S - K * np.exp(-r * T))) < S * sigma * np.sqrt(T), "Deep ITM call price error"
        else:
            assert call_price < S * sigma * np.sqrt(T), "Deep OTM call price error"

    # Test Case 4: Zero volatility
    sigma = 0.0001  # Using very small sigma since we can't use exactly 0
    print("\nTest 4: Near-zero volatility")
    K_values = [90, 100, 110]
    for K in K_values:
        call_price = black_scholes(S, K, T, r, sigma, 'C')
        put_price = black_scholes(S, K, T, r, sigma, 'P')
        print(f"\nStrike = {K}:")
        print(f"Call price: {call_price:.4f}")
        print(f"Put price: {put_price:.4f}")
        # For zero vol, call = max(0, S - K*exp(-rT)) and put = max(0, K*exp(-rT) - S)
        theoretical_call = max(0, S - K * np.exp(-r * T))
        theoretical_put = max(0, K * np.exp(-r * T) - S)
        assert abs(call_price - theoretical_call) < 0.01, "Zero vol call price error"
        assert abs(put_price - theoretical_put) < 0.01, "Zero vol put price error"

if __name__ == "__main__":
    run_tests()