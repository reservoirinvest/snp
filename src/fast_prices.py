import nest_asyncio
from ib_async import IB, Stock
import asyncio
from from_root import from_root

from snp import get_ib, make_snp_weeklies, make_unqualified_snp_underlyings, qualify_me
import pandas as pd

from contextlib import asynccontextmanager
from tqdm import tqdm  # Import tqdm for progress bar

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def get_snp_unds() -> list:
    df = make_snp_weeklies(ROOT / "data" / "templates" / "snp_indexes.yml")
    df = make_unqualified_snp_underlyings(df)
    with get_ib(MARKET='SNP') as ib:
        qualified_contracts = ib.run(qualify_me(ib, df.contract.tolist(), desc='Qualifying SNP Unds'))
        return qualified_contracts


async def get_a_stock_price(item: str, ib: IB, sleep_time: int = 2) -> dict:
    if isinstance(item, str):
        stock_contract = Stock(item, 'SMART', 'USD')
    else:
        stock_contract = item

    ticker = ib.reqMktData(stock_contract)  # Call without await
    
    await asyncio.sleep(sleep_time)  # Use asyncio.sleep instead of ib.sleep
    
    # Check if ticker.last is NaN and wait for 2 more seconds if true
    await check_and_adjust_sleep(ticker, sleep_time)
    
    # Return a dictionary with the symbol and its last price
    key = item if isinstance(item, str) else stock_contract

    return {key: ticker.last}

async def check_and_adjust_sleep(ticker, sleep_time: int, add_time: int=2):
    if pd.isna(ticker.last):
        await asyncio.sleep(sleep_time + add_time)

@asynccontextmanager
async def ib_context():
    ib = IB()
    ib.connect('127.0.0.1', 1300, clientId=1)  # Connect without await
    try:
        yield ib
    finally:
        ib.disconnect()  # Ensure disconnection

async def stock_prices(contracts: list, ib: IB) -> dict:
    results = {}
    tasks = [get_a_stock_price(c, ib) for c in contracts]
    
    # Gather results and combine them into a single dictionary
    for result in await asyncio.gather(*tasks):
        results.update(result)
    
    return results

def process_in_chunks(func, func_params: dict, chunk_size: int = 50) -> dict:
    """Process items in chunks using the provided function and its parameters."""
    items = func_params.pop('paramList')  # Extract items from func_params
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

if __name__ == "__main__":
    # Example usage with a list of symbols
    ROOT = from_root()
    stocks = get_snp_unds()

    # Use the context manager to handle the IB connection
    async def main():
        async with ib_context() as ib:
            func_params = {'paramList': stocks, 'ib': ib}  # Prepare parameters including ib
            results = process_in_chunks(stock_prices, func_params)
            contract = list(results.keys())
            price = list(results.values())
            df = pd.DataFrame({'contract': contract, 'price': price})

            print(df)  # Print the collected results

    # Run the main function
    asyncio.run(main())