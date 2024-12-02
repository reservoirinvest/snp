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
    
    Example:
    --------
    >>> call_price = black_scholes(100, 100, 1, 0.05, 0.2, 'C')
    >>> put_price = black_scholes(100, 100, 1, 0.05, 0.2, 'P')
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
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    """
    Calculate the Greeks for a Black-Scholes option.
    
    Returns a dictionary containing Delta, Gamma, Theta, Vega, and Rho.
    """
    # Small change for finite difference
    eps = 1e-5
    
    # Calculate option price at current values
    price = black_scholes(S, K, T, r, sigma, option_type)
    
    # Delta (∂V/∂S)
    delta_price = black_scholes(S + eps, K, T, r, sigma, option_type)
    delta = (delta_price - price) / eps
    
    # Gamma (∂²V/∂S²)
    gamma_price_up = black_scholes(S + eps, K, T, r, sigma, option_type)
    gamma_price_down = black_scholes(S - eps, K, T, r, sigma, option_type)
    gamma = (gamma_price_up - 2*price + gamma_price_down) / (eps**2)
    
    # Theta (∂V/∂T)
    if T > eps:  # Avoid negative time to expiry
        theta_price = black_scholes(S, K, T - eps, r, sigma, option_type)
        theta = -(price - theta_price) / eps
    else:
        theta = 0.0
    
    # Vega (∂V/∂σ)
    vega_price = black_scholes(S, K, T, r, sigma + eps, option_type)
    vega = (vega_price - price) / eps
    
    # Rho (∂V/∂r)
    rho_price = black_scholes(S, K, T, r + eps, sigma, option_type)
    rho = (rho_price - price) / eps
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }