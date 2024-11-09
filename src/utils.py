import asyncio
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import pytz
import yaml
from dateutil import parser
from dotenv import find_dotenv, load_dotenv
from from_root import from_root
from ib_async import IB, Contract, Order, util
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

ROOT = from_root()
ACTIVESTATUS = os.getenv("ACTIVESTATUS", "")

@dataclass
class Portfolio:
    conId: int = 0
    symbol: str = "Dummy"
    secType: str = "STK"
    expiry: datetime = datetime.now()
    strike: float = 0.0
    right: str = None
    position: float = 0.0
    mktPrice: float = 0.0
    mktVal: float = 0.0
    avgCost: float = 0.0
    unPnL: float = 0.0
    rePnL: float = 0.0

    def empty(self):
        return pd.DataFrame([self.__dict__]).iloc[0:0]

@dataclass
class OpenOrder:
    conId: int = 0
    symbol: str = "Dummy"
    secType: str = "STK"
    expiry: datetime = datetime.now()
    strike: float = 0.0
    right: str = None
    orderId: int = 0
    order: Order = None
    permId: int = 0
    action: str = "SELL"
    totalQuantity: float = 0.0
    lmtPrice: float = 0.0 # Same as xPrice
    status: str = None

    def empty(self):
        return pd.DataFrame([self.__dict__]).iloc[0:0]

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

def get_ib(MARKET: str, cid: int = 10, LIVE: bool = True) -> IB:
    port = get_port(MARKET=MARKET, LIVE=LIVE)
    connection = IB().connect(port=port, clientId=cid)
    return connection

def get_port(MARKET: str, LIVE: bool=True) -> int:
    config = load_config(market=MARKET.upper())

    if LIVE is True:
        port = config.get("PORT")
    else:
        port = config.get('PAPER')

    return port

async def qualify_me(ib: IB, data: list, desc: str = "Qualifying contracts") -> list:
    data = to_list(data)
    tasks = [asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol) for c in data]
    await tqdm_asyncio.gather(*tasks, desc=desc)
    result = [r for t in tasks for r in t.result()]
    return result

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

def quick_pf(ib: IB) -> Union[None, pd.DataFrame]:
    pf = ib.portfolio()

    if pf != []:
        df_pf = util.df(pf)
        df_pf = (util.df(list(df_pf.contract)).iloc[:, :6]).join(
            df_pf.drop(columns=["account"])
        )
        df_pf = df_pf.rename(
            columns={
                "lastTradeDateOrContractMonth": "expiry",
                "marketPrice": "mktPrice",
                "marketValue": "mktVal",
                "averageCost": "avgCost",
                "unrealizedPNL": "unPnL",
                "realizedPNL": "rePnL",
            }
        )
    else:
        df_pf = Portfolio().empty()

    return df_pf

def get_open_orders(ib, is_active: bool = False) -> pd.DataFrame:
    df_openords = OpenOrder().empty()

    trades = ib.reqAllOpenOrders()

    dfo = pd.DataFrame([])

    if trades:
        all_trades_df = (
            clean_ib_util_df([t.contract for t in trades])
            .join(util.df(t.orderStatus for t in trades))
            .join(util.df(t.order for t in trades), lsuffix="_")
        )

        order = pd.Series([t.order for t in trades], name="order")

        all_trades_df = all_trades_df.assign(order=order)

        all_trades_df.rename(
            {"lastTradeDateOrContractMonth": "expiry"}, axis="columns", inplace=True
        )

        if 'symbol' not in all_trades_df.columns:
            if 'contract' in all_trades_df.columns:
                all_trades_df['symbol'] = all_trades_df['contract'].apply(lambda x: x.symbol)
            else:
                raise ValueError("Neither 'symbol' nor 'contract' column found in the DataFrame")

        dfo = all_trades_df[df_openords.columns]

        if is_active:
            dfo = dfo[dfo.status.isin(ACTIVESTATUS)]

    return dfo

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

def get_dte(s: Union[pd.Series, datetime]) -> Union[pd.Series, float]:
    now_utc = datetime.now(timezone.utc)

    if isinstance(s, pd.Series):
        try:
            return (s - now_utc).dt.total_seconds() / (24 * 60 * 60)
        except (TypeError, ValueError):
            return pd.Series([np.nan] * len(s))
    elif isinstance(s, datetime):
        return (s - now_utc).total_seconds() / (24 * 60 * 60)
    else:
        raise TypeError("Input must be a pandas Series or a datetime.datetime object")

async def get_an_option_chain(item:Contract, ib: IB, timeout: int=2):
    try:
        chain = await asyncio.wait_for(ib.reqSecDefOptParamsAsync(
        underlyingSymbol=item.symbol,
        futFopExchange="",
        underlyingSecType=item.secType,
        underlyingConId=item.conId,
        ), timeout=timeout)

        if chain:
            chain = chain[-1] if isinstance(chain, list) else chain
        return chain
    
    except asyncio.TimeoutError:
        logger.error(f"Timeout occurred while getting option chain for {item.symbol}")
        return None

async def chains(contracts: list, ib: IB) -> dict:
    tasks = [get_an_option_chain(item=c, ib=ib) for c in contracts]  # Use get_an_option_chain instead
    results = await asyncio.gather(*tasks)  # Gather results directly
    return {k: v for d in results for k, v in d.items()}  # Combine results into a single dictionary

if __name__ == '__main__':
    ROOT = from_root()
    datapath = ROOT/'data'/'snp_unds.pkl'
    stocks = pd.read_pickle(datapath)

    with get_ib('SNP') as ib:
        out = asyncio.run(ib.reqSecDefOptParamsAsynch('INTC', ""))

    print(out)