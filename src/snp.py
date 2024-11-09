import asyncio
import math
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from ib_async import IB, Contract, Index, LimitOrder, Stock, Option, util
from loguru import logger
from from_root import from_root
from src.utils import flatten, get_an_option_chain, get_ib, load_config, qualify_me

ROOT = from_root()
MARKET = 'SNP'
ACTIVESTATUS = os.getenv("ACTIVESTATUS", "")

CONFIG = load_config(MARKET)
MAXDTE = CONFIG.get("MAXDTE")

# Configure loguru
logger.remove()  # Remove the default logger
log_file = ROOT / "log" / "snp.log"

LOGLEVEL = os.getenv("LOGLEVEL", "ERROR")
logger.add(str(log_file), level=LOGLEVEL, rotation="1 MB", retention="7 days")
util.logToFile(log_file, level=LOGLEVEL)

def get_snp_unds() -> list:
    df = make_snp_weeklies(ROOT / "data" / "templates" / "snp_indexes.yml")
    df = make_unqualified_snp_underlyings(df)
    with get_ib(MARKET='SNP') as ib:
        qualified_contracts = ib.run(qualify_me(ib, df.contract.tolist(), desc='Qualifying SNP Unds'))
        return qualified_contracts
    
def make_snp_weeklies(indexes_path: Path):
    df_weekly_cboes = make_weekly_cboes()
    snps = get_snps()
    df_weekly_snps = df_weekly_cboes[df_weekly_cboes.symbol.isin(snps)].reset_index(drop=True)
    df_weeklies = add_snp_indexes(df_weekly_snps, indexes_path).pipe(split_snp_stocks_and_index)
    return df_weeklies

def make_unqualified_snp_underlyings(df: pd.DataFrame) -> pd.DataFrame:
    contracts = [
        Stock(symbol=symbol, exchange=exchange, currency="USD")
        if secType == "STK"
        else Index(symbol=symbol, exchange=exchange, currency="USD")
        for symbol, secType, exchange in zip(df.symbol, df.secType, df.exchange)
    ]
    df = df.assign(contract=contracts)
    return df

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

def arrange_orders(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    maxmargin = kwargs.get("maxmargin", 5000)
    how_many = kwargs.get("how_many", 2)
    puts_only = kwargs.get("puts_only", False)

    if not puts_only:
        gc = (
            df[df.right == "C"]
            .assign(ratio=df.safe_strike / df.strike)
            .sort_values("ratio")
            .groupby("symbol")
        )
        df_calls = gc.head(how_many).sort_values(
            ["symbol", "strike"], ascending=[True, False]
        )
        dfc = df_calls[df_calls.margin < maxmargin]
        dfc = dfc.assign(ratio=dfc.safe_strike / dfc.strike).sort_values("ratio")
    else:
        dfc = pd.DataFrame([])

    gc = (
        df[df.right == "P"]
        .assign(ratio=df.strike / df.safe_strike)
        .sort_values("ratio")
        .groupby("symbol")
    )
    df_puts = gc.head(how_many).sort_values(
        ["symbol", "strike"], ascending=[True, False]
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

def process_volatility(df: pd.DataFrame) -> pd.DataFrame:
    # Vectorized calculation of implied volatility
    min_iv = np.minimum(df.und_iv, df.und_hv)
    weighted_avg_iv = (df.und_iv + df.und_hv) / 2 * 0.75
    df["iv"] = np.where(df.und_iv < df.und_hv, min_iv + weighted_avg_iv, df.und_iv)
    return df.dropna(subset=["iv"])



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
    
def split_symbol_price_iv(prices_dict: dict) -> pd.DataFrame:
    symbols, prices, ivs, hvs = zip(
        *((symbol, price, iv, hv) for symbol, (price, iv, hv) in prices_dict.items())
    )

    df_prices = pd.DataFrame({"symbol": symbols, "price": prices, "iv": ivs, "hv": hvs})

    return df_prices


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

if __name__ == "__main__":
    pass
