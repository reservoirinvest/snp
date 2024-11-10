from dataclasses import dataclass
from datetime import datetime
from itertools import product
import os
import nest_asyncio
from ib_async import IB, Contract, Order, Stock
import asyncio
from from_root import from_root

from utils import do_in_chunks, clean_ib_util_df, get_dte, get_port, pickle_me, to_list
import pandas as pd
from ib_async import util
from typing import Union
from tqdm.asyncio import tqdm_asyncio
from loguru import logger

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

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

def get_ib(MARKET: str, cid: int = 10, LIVE: bool = True) -> IB:
    port = get_port(MARKET=MARKET, LIVE=LIVE)
    connection = IB().connect(port=port, clientId=cid)
    return connection

async def get_a_stock_price(item: str, ib: IB, sleep_time: int = 2) -> dict:
    stock_contract = Stock(item, 'SMART', 'USD') if isinstance(item, str) else item
    ticker = ib.reqMktData(stock_contract)  # Call without await
    
    await asyncio.sleep(sleep_time)  # Use asyncio.sleep instead of ib.sleep
    
    # Check if ticker.last is NaN and wait if true
    if pd.isna(ticker.last):
        await asyncio.sleep(2)

    ib.cancelMktData(stock_contract)

    # Return a dictionary with the symbol and its last price
    key = item if isinstance(item, str) else stock_contract
    value = ticker.last if not pd.isna(ticker.last) else ticker.close
    return {key: value}

async def prices(contracts: list, ib: IB, sleep_time: int = 2) -> dict:
    tasks = [get_a_stock_price(item=c, ib=ib, sleep_time=sleep_time) for c in contracts]
    results = await asyncio.gather(*tasks)
    return {k: v for d in results for k, v in d.items()}  # Combine results into a single dictionary

async def df_prices(ib: IB, stocks: list, sleep_time: int = 2, msg: str=None) -> pd.DataFrame:
    func_params = {'ib': ib, 'payload': stocks, 'sleep_time': sleep_time}
    results = do_in_chunks(func=prices, func_params=func_params, chunk_size=30, msg=msg)
    if isinstance(list(results)[0], str):
        df = pd.DataFrame.from_dict(results, orient='index', columns=['price']).rename_axis('symbol')
    else:
        df = clean_ib_util_df(list(results.keys()), ist=False)
        df['price'] = list(results.values())
    return df

async def get_an_iv(ib: IB, item: str, sleep_time: int = 3, gentick: str = '106, 104') -> dict:
    stock_contract = Stock(item, 'SMART', 'USD') if isinstance(item, str) else item
    ticker = ib.reqMktData(stock_contract, genericTickList=gentick)  # Request market data with gentick
    
    await asyncio.sleep(sleep_time)  # Use asyncio.sleep instead of ib.sleep
    
    # Check if ticker.impliedVolatility is NaN and wait if true
    if pd.isna(ticker.impliedVolatility):
        await asyncio.sleep(2)

    ib.cancelMktData(stock_contract)

    # Return a dictionary with the symbol, price, implied volatility, and historical volatility
    key = item if isinstance(item, str) else stock_contract
    price = ticker.last if not pd.isna(ticker.last) else ticker.close  # Get last price
    iv = ticker.impliedVolatility  # Get implied volatility from ticker
    hv = ticker.histVolatility  # Get historical volatility from ticker
    return {key: {'price': price, 'iv': iv, 'hv': hv}}  # Return structured data

async def volatilities(contracts: list, ib: IB, sleep_time: int = 3, gentick: str = '106, 104') -> dict:
    tasks = [get_an_iv(item=c, ib=ib, sleep_time=sleep_time, gentick=gentick) for c in contracts]
    results = await asyncio.gather(*tasks)
    return {k: v for d in results for k, v in d.items()}  # Combine results into a single dictionary

async def df_iv(ib: IB, stocks: list, sleep_time: int = 3, msg: str=None) -> pd.DataFrame:
    func_params = {'ib': ib, 'payload': stocks, 'sleep_time': sleep_time}
    results = do_in_chunks(func=volatilities, func_params=func_params, chunk_size=44, msg=msg)
    if isinstance(list(results)[0], str):
        df_out = pd.DataFrame.from_dict(results, orient='index').rename_axis('symbol').reset_index()
    else:
        df = pd.DataFrame.from_dict(results, orient='index', columns=['price', 'iv', 'hv'])
        df_out = clean_ib_util_df(df.index.to_list()).join(df.reset_index()[['hv', 'iv', 'price']])
    return df_out

async def get_an_option_chain(item:Contract, ib: IB, sleep_time: int=2):
    try:
        chain = await asyncio.wait_for(ib.reqSecDefOptParamsAsync(
        underlyingSymbol=item.symbol,
        futFopExchange="",
        underlyingSecType=item.secType,
        underlyingConId=item.conId,
        ), timeout=sleep_time)

        if chain:
            chain = chain[-1] if isinstance(chain, list) else chain
        return chain
    
    except asyncio.TimeoutError:
        logger.error(f"Timeout occurred while getting option chain for {item.symbol}")
        return None

async def chains(contracts: list, ib: IB, sleep_time: int=2) -> dict:
    tasks = [asyncio.create_task(get_an_option_chain(item=c, ib=ib, sleep_time=sleep_time), name=c.symbol) for c in contracts]
    results = await asyncio.gather(*tasks)
    out = {task.get_name(): result for task, result in zip(tasks, results)}
    return out

async def df_chains(ib: IB, stocks: list, sleep_time: int = 4, msg: str=None) -> list:
    func_params = {'ib': ib, 'payload': stocks, 'sleep_time': sleep_time}
    option_chain_data = do_in_chunks(func=chains, func_params=func_params, chunk_size=20, msg=msg)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(
        {key: {
            'exchange': value.exchange,
            'underlyingConId': value.underlyingConId,
            'tradingClass': value.tradingClass,
            'multiplier': value.multiplier,
            'expirations': value.expirations,
            'strikes': value.strikes
        } for key, value in option_chain_data.items() if value},
        orient='index'
    )

    # Create a list to hold the expanded rows
    expanded_rows = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Create Cartesian product of expirations and strikes
        for expiry, strike in product(row['expirations'], row['strikes']):
            expanded_rows.append({
                'symbol': row['tradingClass'],
                'expiry': expiry,
                'strike': strike
            })

    # Create a new DataFrame from the expanded rows
    df_out = pd.DataFrame(expanded_rows)

    # set expiry time to EST 4:00 PM
    # exp_time = pd.to_datetime(df_out.expiry).dt.tz_localize('US/Eastern').apply(lambda x: x.replace(hour=16, minute=0, second=0))
    df_out['dte'] = get_dte(df_out.expiry)

    return df_out

async def qualify_me(ib: IB, data: list, desc: str = "Qualifying contracts") -> list:
    data = to_list(data)
    tasks = [asyncio.create_task(ib.qualifyContractsAsync(c), name=c.localSymbol) for c in data]
    await tqdm_asyncio.gather(*tasks, desc=desc)
    result = [r for t in tasks for r in t.result()]
    return result

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

async def get_cushion(ib: IB) -> dict:
    """Gets account values

    Args:
        ib (IB): an active connection

    Returns:
        dict: current nlv, cash and margins
    """

    df_acc = util.df(ib.accountValues())

    d_map = {
        "TotalCashBalance": "cash",
        "Cushion": "cushion",
        "NetLiquidation": "nlv",
        "InitMarginReq": "init_margin",
        "EquityWithLoanValue": "equity_val",
        "MaintMarginReq": "maint_margin",
        "realizedPnL": "pnl_real",
        "UnrealizedPnL": "pnl_unreal",
        "LookAheadAvailableFunds": "funds_avlbl",
    }

    # get account values as a dictionary
    df_out = df_acc[df_acc.tag.isin(d_map.keys())]
    acc = df_out.set_index("tag").value.apply(float).to_dict()

    # sort account values based on d_map's order
    order = list(d_map.values())
    order_index = {key: index for index, key in enumerate(order)}
    sorted_keys = sorted(d_map.keys(), key=lambda x: order_index.get(x, float("inf")))
    sorted_dict = {d_map.get(key): acc.get(key) for key in sorted_keys}

    return sorted_dict

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

if __name__ == "__main__":

    ROOT = from_root()
    datapath = ROOT/'data'/'snp_unds.pkl'

    # from snp import get_snp_unds
    # stocks = get_snp_unds()
    # pickle_me(stocks, datapath)

    unds = pd.read_pickle(datapath)

    # Run the df_prices function
    with get_ib('SNP') as ib:
        # out = asyncio.run(df_prices(ib=ib, stocks=['ACN', 'GOOG', 'IEFA'], sleep_time=5))
        # out = asyncio.run(df_prices(ib=ib, stocks=stocks))
        
        # out = asyncio.run(df_iv(ib=ib, stocks=['ACN', 'GOOG', 'IEFA'], sleep_time=15))
        # out = asyncio.run(df_iv(ib=ib, stocks=stocks))

        out = asyncio.run(df_chains(ib, unds))
        pickle_me(out, ROOT/'data'/'chains.pkl')
        
    print(out)
    print(f'\nno of symbols with chains = {len(set(out.symbol))}')

