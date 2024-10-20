import asyncio
from dataclasses import dataclass
import itertools
import math
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Union, Iterable

import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import yaml
from dateutil import parser
from dotenv import find_dotenv, load_dotenv
from ib_async import IB, Contract, Index, LimitOrder, Order, Stock, Option, util
from loguru import logger
from scipy.stats import norm
from from_root import from_root

@dataclass
class Portfolio:
    conId: int = 0
    symbol: str = "Dummy"
    secType: str = "STK"
    expiry: datetime = datetime.now()
    strike: float = 0.0
    right: str = "?"
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
    right: str = "?"
    orderId: int = 0
    order: Order = None
    permId: int = 0
    action: str = "SELL"
    totalQuantity: float = 0.0
    lmtPrice: float = 0.0
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

ROOT = from_root()
MARKET = 'SNP'
ACTIVESTATUS = os.getenv("ACTIVESTATUS", "")

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

CONFIG = load_config(MARKET)
PUTSTDMULT = CONFIG.get("PUTSTDMULT")
CALLSTDMULT = CONFIG.get("CALLSTDMULT")
MAXDTE = CONFIG.get("MAXDTE")
MINEXPROM = CONFIG.get("MINEXPROM")

# Configure loguru
logger.remove()  # Remove the default logger
logger.add(str(ROOT / 'log' / 'snp.log'), level="INFO", rotation="1 MB", retention="7 days")  # Log INFO and ERROR messages to a file

class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise ValueError("Timer is running. Use .stop() to stop it")
        self._start_time = datetime.now()
        print(f'\n{self.name} started at {self._start_time.strftime("%d-%b-%Y %H:%M:%S")}')

    def stop(self) -> None:
        if self._start_time is None:
            raise ValueError("Timer is not running. Use .start() to start it")
        elapsed_time = datetime.now() - self._start_time
        hours, remainder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n...{self.name} took: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        self._start_time = None

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

def add_snp_indexes(df: pd.DataFrame, path_to_yaml_file: str) -> pd.DataFrame:
    with open(path_to_yaml_file, "r") as f:
        kv_pairs = yaml.load(f, Loader=yaml.FullLoader)

    dfs = [pd.DataFrame(list(kv_pairs[k].items()), columns=["symbol", "desc"]).assign(exchange=k) for k in kv_pairs.keys()]
    more_df = pd.concat(dfs, ignore_index=True)
    df_all = pd.concat([df, more_df], ignore_index=True)

    return df_all

def split_snp_stocks_and_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(secType=np.where(df.desc.str.contains("Index"), "IND", "STK"))
    return df

def make_snp_weeklies(indexes_path: Path):
    df_weekly_cboes = make_weekly_cboes()
    snps = get_snps()
    df_weekly_snps = df_weekly_cboes[df_weekly_cboes.symbol.isin(snps)].reset_index(drop=True)
    df_weeklies = add_snp_indexes(df_weekly_snps, indexes_path).pipe(split_snp_stocks_and_index)
    return df_weeklies

def make_weekly_cboes() -> pd.DataFrame:
    df = read_weeklys().pipe(rename_weekly_columns).pipe(remove_non_char_symbols)
    df = df.assign(exchange="SMART")
    return df

def rename_weekly_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["desc", "symbol"]
    return df

def remove_non_char_symbols(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.symbol.str.extract("([^a-zA-Z])").isna()[0]]
    return df

def get_snps() -> pd.Series:
    snp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    snps = pd.read_html(snp_url)[0]["Symbol"]
    return snps

def read_weeklys() -> pd.DataFrame:
    dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"
    df = pd.read_html(dls)[0]
    return df

def make_unqualified_snp_underlyings(df: pd.DataFrame) -> pd.DataFrame:
    contracts = [
        Stock(symbol=symbol, exchange=exchange, currency="USD")
        if secType == "STK"
        else Index(symbol=symbol, exchange=exchange, currency="USD")
        for symbol, secType, exchange in zip(df.symbol, df.secType, df.exchange)
    ]
    df = df.assign(contract=contracts)
    return df

async def qualify_me(ib: IB, data: list, desc: str = "Qualifying contracts") -> list:
    data = to_list(data)
    tasks = [asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol) for c in data]
    await tqdm_asyncio.gather(*tasks, desc=desc)
    result = [r for t in tasks for r in t.result()]
    return result

def assemble_snp_underlyings(live: bool = True, fresh: bool = True) -> pd.DataFrame:
    und_path = ROOT / 'data' / 'snp_unds.pkl'
    if not fresh:
        df = get_pickle(und_path)
        if df is not None and not df.empty:
            return df

    df = make_snp_weeklies(ROOT / "data" / "templates" / "snp_indexes.yml")
    df = make_unqualified_snp_underlyings(df)

    with get_ib(MARKET='SNP', LIVE=live) as ib:
        qualified_contracts = ib.run(qualify_me(ib, df.contract.tolist(), desc='Qualifying SNP Unds'))
        dfc = clean_ib_util_df(qualified_contracts)
        df = ib.run(get_mkt_prices(ib, dfc.contract, sleep=15, chunk_size=39))
    
    df.rename(columns={'conId': 'undId', 'iv': 'und_iv', 'hv': 'und_hv', 'price': 'undPrice'}, inplace=True)
    pickle_me(df, und_path)
    return df

def append_safe_strikes(df: pd.DataFrame, PUTSTDMULT: float, CALLSTDMULT: float) -> pd.DataFrame:
    # Vectorized calculation of standard deviations and safe strikes
    df["sdev"] = df["iv"] * df["undPrice"] * np.sqrt(df["dte"] / 365)
    df["safe_strike"] = np.where(
        df["right"] == "P",
        (df["undPrice"] - df["sdev"] * PUTSTDMULT).astype(int),
        (df["undPrice"] + df["sdev"] * CALLSTDMULT).astype(int),
    )
    # Vectorized calculation of intrinsic values
    df["intrinsic"] = np.maximum(
        df["strike"] - df["safe_strike"] * (df["right"] == "P"),
        df["safe_strike"] - df["strike"] * (df["right"] == "C"),
    )
    return df

def append_black_scholes(df: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    # Vectorized calculations for Black-Scholes pricing
    S, K, T, r = df["undPrice"].values, df["strike"].values, df["dte"].values / 365, risk_free_rate
    sigma = df.get('iv', df.get('und_iv', df.get('und_hv', df.get('hv', np.nan))))
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Vectorized calculation of option prices
    df["bsPrice"] = np.where(
        df["right"] == "C",
        S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
        K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    )
    return df

def get_closest_strike(df: pd.DataFrame, above: Optional[bool] = None) -> pd.DataFrame:
    undPrice = df["undPrice"].iloc[0]
    mask = df["strike"] > undPrice if above else df["strike"] < undPrice if above is not None else None

    # Filter DataFrame based on mask
    if mask is not None:
        df = df[mask]

    # Find the closest strike
    closest_index = (df["strike"] - undPrice).abs().idxmin()
    return df.loc[[closest_index]]

async def process_in_chunks(ib: IB, data: any, func: callable = None, func_args: dict = None, chunk_size: int = 25, chunk_desc: str = "processing chunk...") -> list:
    if not func:
        raise ValueError("A function must be provided for processing the data.")
        return None

    if not func_args:
        func_args = {}

    chunks = chunk_me(data, chunk_size)
    processed_data = []

    for chunk in tqdm(chunks, desc=chunk_desc):
        func_args["data"] = chunk

        try:
            result = await func(ib, **func_args)
            processed_chunk = [result]

        except (AttributeError, ValueError):
            try:
                if isinstance(chunk, (list, pd.Series)):
                    tasks = []
                    farg2 = func_args
                    for item in chunk:
                        farg2["data"] = item
                        t = [func(ib, **farg2)]
                        tasks.extend(t)
                    result = await asyncio.gather(*tasks)
                    processed_chunk = [clean_ib_util_df(chunk).join(pd.DataFrame(result)).drop(columns='localsymbol')]
                else:
                    raise TypeError(f"Invalid data type: {type(chunk)} for function {func.__name__}")

            except Exception as e2:
                raise ValueError(f"Function {func.__name__} does not accept data type {type(chunk)}, \nerror:{e2}") from e2

        processed_data.extend(processed_chunk)

    return processed_data

def get_xPrice(df: pd.DataFrame) -> pd.DataFrame:
    c = df.right == 'C'
    p = df.right == 'P'

    # Simplified vectorized operations for both call and put options
    conditions = [
        (c & df.undPrice.between(df.strike, df.safe_strike), df.safe_strike - df.undPrice + df[['bsPrice', 'price']].max(axis=1)),
        (c & df.strike.between(df.undPrice, df.safe_strike), df.safe_strike - df.strike + df[['bsPrice', 'price']].max(axis=1)),
        (c & df.safe_strike.between(df.undPrice, df.strike), df[['bsPrice', 'price']].max(axis=1)),
        (p & df.undPrice.between(df.safe_strike, df.strike), df.strike - df.safe_strike + df[['bsPrice', 'price']].max(axis=1)),
        (p & df.strike.between(df.safe_strike, df.undPrice), df.undPrice - df.safe_strike + df[['bsPrice', 'price']].max(axis=1)),
        (p & df.safe_strike.between(df.strike, df.undPrice), df[['bsPrice', 'price']].max(axis=1)),
    ]

    for condition, xPrice in conditions:
        df.loc[condition, 'xPrice'] = xPrice

    return df

def append_xPrice(df: pd.DataFrame, MINEXPROM: float) -> pd.DataFrame:
    df = df.drop(columns=["order"], errors="ignore")
    df = get_xPrice(df)

    margin = np.where(df.margin <= 0, np.nan, df.margin)

    rom = df.xPrice * df.lot / margin * 365 / df.dte
    df = df.assign(rom=rom)

    df = df[df.rom > MINEXPROM].reset_index(drop=True)

    df = df.loc[(df.xPrice / df.price).sort_values().index]

    return df

def make_snp_naked_puts(save: bool = False):
    timer = Timer("Making SNP nakeds")
    timer.start()

    file_path = ROOT / 'data' / 'snp_unds.pkl'
    df_unds = assemble_snp_underlyings(fresh=True)
    pickle_me(df_unds, file_path)

    df_ch = make_chains(df_unds, save=True)
    dfp = df_ch[(df_ch.right == "P") & (df_ch.dte <= MAXDTE)]
    dfe = dfp[dfp.groupby(['ib_symbol', 'dte']).dte.transform('min').astype('int') == dfp.dte.astype('int')]
    dfef = dfe[~dfe.undPrice.isnull()]

    dft = dfef.groupby(['ib_symbol', 'dte']).apply(get_closest_strike).reset_index(drop=True)
    dft = dft.sort_values(['ib_symbol', 'dte'])

    dft = process_volatility(dft)
    dft = append_safe_strikes(dft, PUTSTDMULT, CALLSTDMULT)
    dft = append_black_scholes(dft, us_repo_rate() / 100)

    contracts = [Option(s, util.formatIBDatetime(e)[:8], k, r, exchange='SMART', currency='USD')
                 for s, e, k, r in zip(dft.ib_symbol, dft.expiry, dft.strike, dft.right)]

    with get_ib(MARKET) as ib:
        cts = ib.run(process_in_chunks(ib, contracts, func=qualify_me, func_args={'desc': 'qualified'}, chunk_size=200, chunk_desc="Qualifying..."))
        res = ib.run(process_in_chunks(ib, cts, func=get_a_price_iv, func_args={'sleep': 15, 'gentick':''}, chunk_desc='Pricing'))

    df_price = pd.concat(res, ignore_index=True).drop(['secType', 'iv', 'hv'], axis=1)
    dfn = df_price.merge(dft, on=['ib_symbol', 'expiry', 'strike', 'right'])
    df = snp_marcom(dfn)
    df = df[df.price > 0]

    df_snp = append_xPrice(df.assign(lot=100), MINEXPROM)
    df_snp = df_snp.reset_index(drop=True).assign(lot=1)

    if save:
        pickle_me(df_snp, ROOT/'data'/str(MARKET.lower()+'_nakeds.pkl'))

    timer.stop()
    return df_snp

async def get_market_data(ib: IB, c: Contract, sleep: float = 2, gentick: str='106, 104'):
    tick = ib.reqMktData(c, genericTickList=gentick)
    try:
        await asyncio.sleep(sleep)
    finally:
        ib.cancelMktData(c)

    return tick

async def get_a_price_iv(ib:IB, data: Contract|Option|Stock, sleep: float = 15, gentick: str='106, 104') -> dict:
    mkt_data = await get_market_data(ib, data, sleep, gentick)
    data = mkt_data.__dict__

    price_dict = {k: v for k, v in data.items() if k in ['close', 'last']}
    localSymbol = data.get('contract').localSymbol

    undPrice = price_dict.get('last') if not pd.isnull(price_dict.get('last')) else price_dict.get('close')
    iv = data.get('impliedVolatility')
    hv = data.get('histVolatility')

    if math.isnan(undPrice):
        logger.info(f"No price found for {localSymbol}!")

    return {"localsymbol": localSymbol, "price": undPrice, "iv": iv, "hv": hv}

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

def arrange_orders(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    maxmargin = kwargs.get("maxmargin", 5000)
    how_many = kwargs.get("how_many", 2)
    puts_only = kwargs.get("puts_only", False)

    if not puts_only:
        gc = (
            df[df.right == "C"]
            .assign(ratio=df.safe_strike / df.strike)
            .sort_values("ratio")
            .groupby("ib_symbol")
        )
        df_calls = gc.head(how_many).sort_values(
            ["ib_symbol", "strike"], ascending=[True, False]
        )
        dfc = df_calls[df_calls.margin < maxmargin]
        dfc = dfc.assign(ratio=dfc.safe_strike / dfc.strike).sort_values("ratio")
    else:
        dfc = pd.DataFrame([])

    gc = (
        df[df.right == "P"]
        .assign(ratio=df.strike / df.safe_strike)
        .sort_values("ratio")
        .groupby("ib_symbol")
    )
    df_puts = gc.head(how_many).sort_values(
        ["ib_symbol", "strike"], ascending=[True, False]
    )
    dfp = df_puts[df_puts.margin < maxmargin]
    dfp = dfp.assign(ratio=dfp.strike / dfp.safe_strike).sort_values("ratio")

    df_nakeds = pd.concat([dfc, dfp], axis=0, ignore_index=True)
    df_nakeds = df_nakeds[df_nakeds.xPrice > df_nakeds.price]
    df_nakeds = df_nakeds.reset_index(drop=True)
    df_nakeds = df_nakeds.loc[
        df_nakeds["xPrice"].div(df_nakeds["price"]).sort_values().index
    ]

    return df_nakeds

def make_ib_orders(df: pd.DataFrame) -> tuple:
    contracts = df.contract.to_list()
    orders = [
        LimitOrder(action="SELL", totalQuantity=abs(int(q)), lmtPrice=p)
        for q, p in zip(df.lot, df.xPrice)
    ]

    cos = tuple((c, o) for c, o in zip(contracts, orders))

    return cos

def place_snp_orders():
    config = load_config(MARKET)
    port = config.get('PORT')
    MARGINPERORDER = config.get('MARGINPERORDER')

    nakeds_path = ROOT / 'data' / 'snp_nakeds.pkl'
    if not yes_or_no(f'snp_nakeds.pkl is {how_many_days_old(nakeds_path):.2f} days old. Want to load?'):
        print('Aborting order process.')
        return None

    df_opts = get_pickle(nakeds_path)
    
    with IB().connect(port=port, clientId=10) as ib:
        dfo = get_open_orders(ib)
        dfp = quick_pf(ib)

    remove_ib_syms = set(dfo.symbol.to_list() if not dfo.empty else []) | set(dfp.symbol.to_list() if not dfp.empty else [])

    dft = df_opts[~df_opts.ib_symbol.isin(remove_ib_syms)].reset_index(drop=True)
    print(f'\n{len(dft)} options available for order placement\n')
    print(dft[['ib_symbol', 'undPrice', 'strike', 'safe_strike', 'right', 'dte', 'bsPrice', 'price', 'xPrice', 'margin', 'rom']].head())

    df_nakeds = arrange_orders(dft, maxmargin=MARGINPERORDER)
    cos = make_ib_orders(df_nakeds)

    if not yes_or_no(f"Do you want to place {len(cos)} orders?"):
        print("Order placement aborted.")
        return df_nakeds

    with IB().connect(port=port, clientId=10) as ib:
        ordered = place_orders(ib=ib, cos=cos)

    filename = f"{datetime.now().strftime('%Y%m%d_%I_%M_%p')}_snp_naked_orders.pkl"
    pickle_me(ordered, ROOT / "data" / "xn_history" / filename)

    logger.info(f"{len(ordered)} Orders placed and saved successfully.")
    print(util.df(ordered).head())

def place_orders(ib: IB, cos: Union[tuple, list], blk_size: int = 25) -> List:
    trades = []

    if isinstance(cos, (tuple, list)) and (len(cos) == 2):
        c, o = cos
        trades.append(ib.placeOrder(c, o))

    else:
        cobs = {cos[i : i + blk_size] for i in range(0, len(cos), blk_size)}

        for b in tqdm(cobs):
            for c, o in b:
                td = ib.placeOrder(c, o)
                trades.append(td)
            ib.sleep(0.75)

    return trades

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
            {"lastTradeDateOrContractMonth": "expiry",
             "symbol": "ib_symbol"}, axis="columns", inplace=True
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

def process_volatility(df: pd.DataFrame) -> pd.DataFrame:
    # Vectorized calculation of implied volatility
    min_iv = np.minimum(df.und_iv, df.und_hv)
    weighted_avg_iv = (df.und_iv + df.und_hv) / 2 * 0.75
    df["iv"] = np.where(df.und_iv < df.und_hv, min_iv + weighted_avg_iv, df.und_iv)
    return df.dropna(subset=["iv"])

def us_repo_rate():
    return 5.51

def snp_marcom(df: pd.DataFrame) -> pd.DataFrame:
    df['itm'] = df.apply(lambda row: max(row['strike'] - row['undPrice'], 0) if row['right'] == 'P'
                         else max(row['undPrice'] - row['strike'], 0), axis=1)
    
    df['margin'] = df.apply(lambda row:
        row['price'] + max(0.20 * (2 * row['undPrice']) - row['itm'], 0.10 * row['strike'])
        if row['right'] == 'P'
        else row['price'] + max(0.15 * (3 * row['undPrice']) - row['itm'], 0.10 * row['strike']), axis=1)
    
    df['margin'] *= 100
    df['comm'] = 0.65
    
    return df.drop(columns='itm')

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

def make_chains(df_unds: pd.DataFrame, timeout: float=15, chunks: int=15, save: bool=False, msg: str='Getting chains') -> pd.DataFrame:
    MARKET = 'NSE' if df_unds.contract.iloc[0].exchange == 'NSE' else 'SNP'
    df_unds['undId'] = df_unds['contract'].apply(lambda x: x.conId)

    with get_ib(MARKET) as ib:
        chains = ib.run(get_option_chains(ib, df_unds.contract.to_list(), timeout=timeout, chunk_size=chunks, msg=msg))

    df_chains = util.df([c for c in chains if c is not None])
    df_chains.rename(columns={'underlyingConId': 'undId'}, inplace=True, errors="ignore")
    dfc = df_chains[['undId', 'expirations', 'strikes']]
    expirations = dfc.expirations.apply(lambda d: [convert_to_utc_datetime(val, eod=True) for val in d])
    dfc.loc[:, 'expirations'] = expirations

    data = []
    for i, row in dfc.iterrows():
        data.extend([(int(row['undId']), exp, strike, get_dte(exp), right) for exp, strike, right in itertools.product(row['expirations'], row['strikes'], ['P', 'C'])])

    df_chains = pd.DataFrame(data, columns=['undId', 'expiry', 'strike', 'dte', 'right'])
    df_chains = df_chains[df_chains.dte > 0]

    dfch = pd.merge(df_unds.drop(columns=['expiry', 'strike', 'right', 'contract'], errors='ignore'), df_chains, on=['undId'], how='left')
    dfch = dfch.rename(columns={'price': 'undPrice', 'iv': 'und_iv', 'hv': 'und_hv'}, errors='ignore')

    if save:
        ROOT = from_root()
        pickle_me(dfch, ROOT/'data'/str(MARKET.lower()+'_opts.pkl'))

    return dfch

def get_ib(MARKET: str, cid: int = 10, LIVE: bool = True) -> IB:
    port = get_port(MARKET=MARKET, LIVE=LIVE)
    connection = IB().connect(port=port, clientId=cid)
    return connection

async def get_option_chains(ib: IB, contracts: list, msg: str, chunk_size:int=20, timeout:float=4) -> list:
    option_chains = []
    total_contracts = len(contracts)

    with tqdm(total=total_contracts, unit="contract", desc=msg) as pbar:
        for i in range(0, total_contracts, chunk_size):
            chunk = contracts[i: i+chunk_size]
            tasks = [get_an_option_chain(ib, contract, timeout) for contract in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            option_chains.extend([chain for chain in results])
            pbar.update(len(chunk))

    return option_chains

async def get_an_option_chain(ib: IB, contract:Contract, timeout: int=2):
    try:
        chain = await asyncio.wait_for(ib.reqSecDefOptParamsAsync(
        underlyingSymbol=contract.symbol,
        futFopExchange="",
        underlyingSecType=contract.secType,
        underlyingConId=contract.conId,
        ), timeout=timeout)

        if chain:
            chain = chain[-1] if isinstance(chain, list) else chain
        return chain
    except asyncio.TimeoutError:
        logger.error(f"Timeout occurred while getting option chain for {contract.symbol}")
        return None
    
def split_symbol_price_iv(prices_dict: dict) -> pd.DataFrame:
    symbols, prices, ivs, hvs = zip(
        *((symbol, price, iv, hv) for symbol, (price, iv, hv) in prices_dict.items())
    )

    df_prices = pd.DataFrame({"ib_symbol": symbols, "price": prices, "iv": ivs, "hv": hvs})

    return df_prices

async def get_mkt_prices(ib:IB, data: list, chunk_size: int = 44, sleep: int = 7, gentick:str='106, 104') -> pd.DataFrame:
    data = to_list(data)
    chunks = chunk_me(data, chunk_size)
    results = dict()

    for cts in tqdm(chunks, desc="Mkt prices with IVs"):
        tasks = [get_a_price_iv(ib, c, sleep, gentick) for c in cts]
        res = await asyncio.gather(*tasks)

        for r in res:
            symbol, price, iv, hv = r.values()
            results[symbol] = (price, iv, hv)

    df_prices = split_symbol_price_iv(results)
    dfp = clean_ib_util_df(data).join(df_prices.drop(columns='ib_symbol'))

    return dfp

def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def chunk_me(data, size: int = 25) -> list:
    if isinstance(data, (list, pd.Series, pd.DataFrame)):
        d = list(flatten(data))
        return [d[i : i + size] for i in range(0, len(d), size)]
    elif isinstance(data, set):
        data_list = list(data)
        return [data_list[i : i + size] for i in range(0, len(data_list), size)]

    logger.error(
        f"Data type needs to be a list, pd.Series, pd.DataFrame, or set, not {type(data)}"
    )
    return None

def to_list(data):
    if isinstance(data, list):
        return list(flatten(data))

    try:
        return list(data)
    except TypeError:
        return [data]

def get_port(MARKET: str, LIVE: bool=True) -> int:
    config = load_config(market=MARKET.upper())

    if LIVE is True:
        port = config.get("PORT")
    else:
        port = config.get('PAPER')

    return port

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
    udf.rename(columns={"lastTradeDateOrContractMonth": "expiry",
                        "symbol": "ib_symbol"}, inplace=True)

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

if __name__ == "__main__":
    df = make_snp_naked_puts(save=True)
    print(f'{len(df)} contracts found!')
    print(df.head())