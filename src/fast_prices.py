import nest_asyncio
from ib_async import IB, Stock
import asyncio
from from_root import from_root

from snp import get_ib, make_snp_weeklies, make_unqualified_snp_underlyings, pickle_me, qualify_me, clean_ib_util_df
import pandas as pd

from tqdm import tqdm  # Import tqdm for progress bar

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def get_snp_unds() -> list:
    df = make_snp_weeklies(ROOT / "data" / "templates" / "snp_indexes.yml")
    df = make_unqualified_snp_underlyings(df)
    with get_ib(MARKET='SNP') as ib:
        qualified_contracts = ib.run(qualify_me(ib, df.contract.tolist(), desc='Qualifying SNP Unds'))
        return qualified_contracts
    
def process_in_chunks(func, func_params: dict, chunk_size: int = 44) -> dict:
    """Process payload in chunks using the provided function and its parameters."""
    
    if 'payload' not in func_params:
        raise ValueError("Missing 'payload' in func_params.")
    
    items = func_params.pop('payload')  # Extract items from func_params
    all_results = {}
    
    # Initialize tqdm progress bar
    with tqdm(total=len(items), desc=f"{func.__name__} in {chunk_size}s", unit="chunk") as pbar:
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            results = asyncio.run(func(chunk, **func_params))  # Call the function and collect results
            all_results.update(results)  # Combine results from each chunk
            
            # Update progress bar
            pbar.update(len(chunk))
    
    return all_results

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

async def stock_prices(contracts: list, ib: IB) -> dict:
    tasks = [get_a_stock_price(item=c, ib=ib) for c in contracts]
    results = await asyncio.gather(*tasks)  # Gather results directly
    return {k: v for d in results for k, v in d.items()}  # Combine results into a single dictionary

async def df_prices(ib: IB, stocks: list) -> pd.DataFrame:
    func_params = {'ib': ib, 'payload': stocks}  # Prepare parameters including ib
    results = process_in_chunks(func=stock_prices, func_params=func_params, chunk_size=30)
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
    
    # Check if ticker.last is NaN and wait if true
    if pd.isna(ticker.last):
        await asyncio.sleep(2)

    # Return a dictionary with the symbol, price, implied volatility, and historical volatility
    key = item if isinstance(item, str) else stock_contract
    price = ticker.last if not pd.isna(ticker.last) else ticker.close  # Get last price
    iv = ticker.impliedVolatility  # Get implied volatility from ticker
    hv = ticker.histVolatility  # Get historical volatility from ticker
    return {key: {'price': price, 'iv': iv, 'hv': hv}}  # Return structured data

async def volatilities(contracts: list, ib: IB, sleep_time: int = 3, gentick: str = '106, 104') -> dict:
    tasks = [get_an_iv(item=c, ib=ib, sleep_time=sleep_time, gentick=gentick) for c in contracts]
    results = await asyncio.gather(*tasks)  # Gather results directly
    return {k: v for d in results for k, v in d.items()}  # Combine results into a single dictionary

async def df_iv(ib: IB, stocks: list) -> pd.DataFrame:
    func_params = {'ib': ib, 'payload': stocks}  # Prepare parameters including ib
    results = process_in_chunks(func=volatilities, func_params=func_params, chunk_size=30)
    if isinstance(list(results)[0], str):
        df_out = pd.DataFrame.from_dict(results, orient='index').rename_axis('symbol')
    else:
        df = pd.DataFrame.from_dict(results, orient='index', columns=['price', 'iv', 'hv'])
        df_out = clean_ib_util_df(df.index.to_list()).join(df.reset_index()[['hv', 'iv', 'price']])
    return df_out

if __name__ == "__main__":

    ROOT = from_root()
    datapath = ROOT/'data'/'snp_unds.pkl'

    # stocks = get_snp_unds()
    # pickle_me(stocks, datapath)

    stocks = pd.read_pickle(datapath)


    # Run the df_prices function
    with get_ib('SNP') as ib:
        # out = asyncio.run(df_prices(ib=ib, stocks=['ABBV', 'ACN']))
        # out = asyncio.run(df_prices(ib=ib, stocks=stocks[:60]))
        
        out = asyncio.run(df_iv(ib=ib, stocks=stocks[:87]))
        # out = asyncio.run(df_iv(ib=ib, stocks=['ABBV', 'ACN', 'INTC', 'SBUX', 'DG', 'EMR']))
    print(out)
    print(f'\nlength of df without prices = {len(out[out.price.isnull()])}')