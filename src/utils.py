import asyncio
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pytz
import yaml
from dateutil import parser
from dotenv import find_dotenv, load_dotenv
from from_root import from_root
from ib_async import util
from loguru import logger
from scipy.stats import norm
from tqdm import tqdm

ROOT = from_root()

class Timediff:
    def __init__(self, td: timedelta, days: int, hours: int, minutes: int, seconds: float):
        self.td = td
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds

def do_in_chunks(func, func_params: dict, chunk_size: int = 44, msg: Optional[str] = None) -> dict:
    """Process payload in chunks using the provided function and its parameters."""
    
    if 'payload' not in func_params:
        raise ValueError("Missing 'payload' in func_params.")
    
    items = func_params.pop('payload')  # Extract items from func_params
    all_results = {}
    
    # Use func.__name__ as default message if msg is None
    msg = msg or func.__name__
    
    # Initialize tqdm progress bar
    with tqdm(total=len(items), desc=msg, unit="chunk") as pbar:
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            results = asyncio.run(func(chunk, **func_params))  # Call the function and collect results
            all_results.update(results)  # Combine results from each chunk
            
            # Update progress bar
            pbar.update(len(chunk))
    
    return all_results

def pickle_me(obj, file_path: Path):
    with open(str(file_path), "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_pickle(path: Path, print_msg: bool = True):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        if print_msg:
            logger.error(f"File not found: {path}")
        return None

def to_list(data):
    if isinstance(data, list):
        return list(flatten(data))

    try:
        return list(data)
    except TypeError:
        return [data]
    
def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def clean_ib_util_df(
    contracts: Union[list, pd.Series],
    eod=True,
    ist=True
    ) -> Union[pd.DataFrame, None]:
    """Cleans ib_async's util.df to keep only relevant columns"""

    # Ensure contracts is a list
    if isinstance(contracts, pd.Series):
        ct = contracts.to_list()
    elif not isinstance(contracts, list):
        logger.error(
            f"Invalid type for contracts: {type(contracts)}. Must be list or pd.Series."
        )
        return None
    else:
        ct = contracts

    # Try to create DataFrame from contracts
    try:
        udf = util.df(ct)
    except (AttributeError, ValueError) as e:
        logger.error(f"Error creating DataFrame from contracts: {e}")
        return None

    # Check if DataFrame is empty
    if udf.empty:
        return None

    # Select and rename columns
    udf = udf[
        [
            "symbol",
            "conId",
            "secType",
            "lastTradeDateOrContractMonth",
            "strike",
            "right",
        ]
    ]
    udf.rename(columns={"lastTradeDateOrContractMonth": "expiry"}, inplace=True)

    # Convert expiry to UTC datetime, if it exists
    if len(udf.expiry.iloc[0]) != 0:
        udf["expiry"] = udf["expiry"].apply(
            lambda x: convert_to_utc_datetime(x, eod=eod, ist=ist)
        )
    else:
        udf["expiry"] = pd.NaT

    # Assign contracts to DataFrame
    udf["contract"] = ct

    return udf

def convert_to_utc_datetime(date_string, eod=False, ist=True):
    try:
        dt = parser.parse(date_string)
    except ValueError as e:
        logger.error(f"Invalid date string format {e}")
        return np.nan

    if eod:
        if ist:
            timezone = pytz.timezone("Asia/Kolkata")
            dt = dt.replace(hour=15, minute=30, second=0)
        else:
            timezone = pytz.timezone("America/New_York")
            dt = dt.replace(hour=16, minute=0, second=0)

        dt = timezone.localize(dt)

    return dt.astimezone(pytz.UTC)

def get_port(MARKET: str, LIVE: bool=True) -> int:
    config = load_config(market=MARKET.upper())

    if LIVE is True:
        port = config.get("PORT")
    else:
        port = config.get('PAPER')

    return port

def load_config(market: str):
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path=dotenv_path)
    
    config_path = ROOT / "config" / f"{market.lower()}_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    for key, value in os.environ.items():
        if key in config:
            config[key] = value
    
    return config

def yes_or_no(question: str, default="n") -> bool:
    while True:
        answer = input(question + " (y/n): ").lower().strip()
        if not answer:
            return default == "y"
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        else:
            print("Please answer yes or no.")

def get_file_age(file_path: Path) -> Optional[Timediff]:
    if not file_path.exists():
        logger.info(f"{file_path} file is not found")
        return None

    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
    time_now = datetime.now()
    td = time_now - file_time

    return split_time_difference(td)

def split_time_difference(diff: timedelta) -> Timediff:
    days = diff.days
    hours, remainder = divmod(diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds += diff.microseconds / 1e6

    return Timediff(diff, days, hours, minutes, seconds)

def how_many_days_old(file_path: Path) -> float:
    file_age = get_file_age(file_path=file_path)

    seconds_in_a_day = 86400
    file_age_in_days = file_age.td.total_seconds() / seconds_in_a_day if file_age else None

    return file_age_in_days

def get_dte(s: Union[pd.Series, datetime], exchange: Optional[str] = None) -> Union[pd.Series, float]:
    now_utc = datetime.now(timezone.utc)

    if isinstance(s, pd.Series):
        try:
            if isinstance(s.iloc[0], str):
                if exchange == 'NSE':
                    s = pd.to_datetime(s).dt.tz_localize('Asia/Kolkata').apply(
                        lambda x: x.replace(hour=15, minute=30, second=0)
                    )
                else:
                    s = pd.to_datetime(s).dt.tz_localize('US/Eastern').apply(
                        lambda x: x.replace(hour=16, minute=0, second=0)
                    )
            return (s - now_utc).dt.total_seconds() / (24 * 60 * 60)
        except (TypeError, ValueError):
            return pd.Series([np.nan] * len(s))
    elif isinstance(s, datetime):
        return (s - now_utc).total_seconds() / (24 * 60 * 60)
    else:
        raise TypeError("Input must be a pandas Series or a datetime.datetime object")
    
def us_repo_rate():
    """Risk free US interest rate

    Returns:
        _type_: float (5.51)

    """
    tbill_yield = web.DataReader('DGS1MO', 'fred', start=datetime.now() -
                                 timedelta(days=365), end=datetime.now())['DGS1MO'].iloc[-1]
    return tbill_yield

def black_scholes(
    S: float,  # und_price
    K: float,  # strike
    T: float,  # dte converted to years-to-expiry
    r: float,  # risk-free rate
    sigma: float,  # implied volatility
    option_type: str,  # Put or Call right
) -> float:
    """Black-Scholes Option Pricing Model"""

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "P":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'C' for Call and 'P' for Put.")

    return price

if __name__ == '__main__':
    ROOT = from_root()
    datapath = ROOT/'data'/'snp_unds.pkl'
    stocks = pd.read_pickle(datapath)
