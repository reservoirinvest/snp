import os

import numpy as np
import pandas as pd
from from_root import from_root
from ib_async import util  # noqa: F811
from ib_async import Contract
from loguru import logger

from snp import (append_black_scholes, get_a_price_iv, get_ib, get_mkt_prices,
                 get_open_orders, get_xPrice, how_many_days_old, load_config,
                 make_chains, pickle_me, process_in_chunks, qualify_me,
                 quick_pf, us_repo_rate)

util.startLoop()

# Constants and Configuration
ROOT = from_root()
MARKET = 'SNP'
config = load_config(market=MARKET)

# Configure loguru
logger.remove()  # Remove the default logger
log_file = ROOT / "log" / "covers.log"

LOGLEVEL = os.getenv("LOGLEVEL", "DEBUG")
logger.add(str(log_file), level=LOGLEVEL, rotation="1 MB", retention="7 days")
util.logToFile(log_file, level=LOGLEVEL)

# Helper Functions
def get_positions(secType: str = None) -> pd.DataFrame:
    with get_ib(MARKET='SNP') as ib:
        df_pf = quick_pf(ib)

    columns = ['symbol', 'right', 'expiry', 'position', 'contract', 'mktPrice', 'avgCost', 'unPnL'] \
        if secType in ['OPT', None] \
            else ['symbol', 'position', 'contract', 'mktPrice', 'mktVal', 'avgCost', 'unPnL']
    return (df_pf[df_pf['secType'] == secType][columns]
            .pipe(lambda df: df if not df.empty else pd.DataFrame(columns=columns)))

def has_covered_option(row: pd.Series, option_positions: pd.DataFrame) -> bool:
    options = option_positions[option_positions['symbol'] == row['symbol']]
    if row['position'] > 0:  # Long stock position
        return any((options['right'] == 'C') & (options['position'] < 0))  # Short call
    elif row['position'] < 0:  # Short stock position
        return any((options['right'] == 'P') & (options['position'] < 0))  # Short put
    return False

def remove_covered_positions(stock_positions: pd.DataFrame, option_positions: pd.DataFrame) -> pd.DataFrame:
    return stock_positions[
        ~stock_positions.apply(lambda row: has_covered_option(row, option_positions), axis=1)
    ]

# Main Processing Functions
def process_options(positions: pd.DataFrame) -> pd.DataFrame:

    df_opts = make_chains(positions, msg='Covered chains')

    max_dte = config['CCCCP_MAX_DTE']
    df_options = df_opts[(df_opts['dte'] > 4) & (df_opts['dte'] <= max_dte)]
   
    df_options = df_options.rename(columns={'mktPrice': 'undPrice'})
    df_focus = calculate_safe_strike(df_options)
    df_focus = get_option_prices(df_focus)

    # Keep only the options with strike price closest to safe_strike
    df_focus['strike_diff'] = abs(df_focus['strike'] - df_focus['safe_strike'])
    df_focus = df_focus.loc[df_focus.groupby('ib_symbol')['strike_diff'].idxmin()]

    # Remove the temporary 'strike_diff' column
    df_focus = df_focus.drop('strike_diff', axis=1)

    # Remove any remaining duplicates based on 'ib_symbol'
    df_focus = df_focus.drop_duplicates(subset=['ib_symbol'], keep='first')

    # Reset the index to ensure it's continuous
    df_focus = df_focus.reset_index(drop=True)

    return df_focus

def filter_options(df_opts: pd.DataFrame, right: str) -> pd.DataFrame:
    max_dte = config['CCCCP_MAX_DTE']
    df_filtered = df_opts[(df_opts['right'] == right) &
                          (df_opts['dte'] > 4) &
                          (df_opts['dte'] <= max_dte)].copy()

    with get_ib('SNP') as ib:
        open_orders = get_open_orders(ib)

    open_sell_options = open_orders[(open_orders['secType'] == 'OPT') &
                                    (open_orders['action'] == 'SELL')]

    symbols_with_open_orders = set(open_sell_options['symbol'])

    df_filtered = df_filtered[~df_filtered['ib_symbol'].isin(symbols_with_open_orders)]

    return df_filtered

def calculate_safe_strike(df_options: pd.DataFrame) -> pd.DataFrame:
    
    # Filter out df_options whose dte is between CCCCP_MIN_DTE and CCCCP_MAX_DTE
    df_options = df_options[(df_options['dte'] >= config['CCCCP_MIN_DTE']) & (df_options['dte'] <= config['CCCCP_MAX_DTE'])]
    
    # Compute stdmult based on the 'right' column in df_options
    stdmult = np.where(df_options['right'] == 'C', config['CALLSTDMULT'], config['PUTSTDMULT'])
    df_options.loc[:, 'sdev'] = df_options['und_iv'] * df_options['undPrice'] * np.sqrt(df_options['dte'] / 365)

    # Calculate safe strike based on the 'right' column
    df_options['safe_strike'] = np.where(
        df_options['right'] == 'C',
        np.ceil(df_options['undPrice'] + df_options['sdev'] * stdmult),
        np.floor(df_options['undPrice'] - df_options['sdev'] * stdmult)
    )

    # Filter options based on the calculated safe strike and position
    df_opts = df_options[
        ((df_options['right'] == 'C') & (df_options['strike'] > df_options['safe_strike']) & (df_options['position'] > 0)) |
        ((df_options['right'] == 'P') & (df_options['strike'] < df_options['safe_strike']) & (df_options['position'] < 0))
    ]

    # Group and filter the options to select two adjacent strikes with minimum DTE for each symbol
    filtered = df_opts.groupby('ib_symbol').apply(
        lambda x: x[x['strike'] >= x['safe_strike'].iloc[0]].sort_values(['strike', 'dte']).head(2) if x['right'].iloc[0] == 'C' 
        else x[x['strike'] <= x['safe_strike'].iloc[0]].sort_values(['strike', 'dte'], ascending=[False, True]).head(2),
        include_groups=False
    )

    output = filtered.reset_index(level='ib_symbol')
    output = output[output['dte'] == output.groupby('ib_symbol')['dte'].transform('min')]

    return output

def get_option_prices(df_options: pd.DataFrame) -> pd.DataFrame:
    contracts = [Contract(secType='OPT', symbol=s, lastTradeDateOrContractMonth=util.formatIBDatetime(e)[:8],
                          strike=k, right=r, exchange='SMART', currency='USD')
                 for s, e, k, r in zip(df_options.ib_symbol, df_options.expiry, df_options.strike, df_options.right)]

    with get_ib(MARKET) as ib:
        qualified_contracts = ib.run(process_in_chunks(ib, contracts, func=qualify_me,
                                                       func_args={'desc': 'qualified'}, chunk_size=200))
        prices = ib.run(process_in_chunks(ib, qualified_contracts, func=get_a_price_iv,
                                          func_args={'sleep': 15, 'gentick':''}, chunk_desc='Pricing'))

    df_price = pd.concat(prices, ignore_index=True)
    df_options = df_options.merge(df_price, on=['ib_symbol', 'expiry', 'strike', 'right'])

    # Find the strike closest to undPrice for each ib_symbol
    df_options['strike_diff'] = abs(df_options['strike'] - df_options['undPrice'])
    df_options = df_options.loc[df_options.groupby('ib_symbol')['strike_diff'].idxmin()]
    df_options = df_options.drop('strike_diff', axis=1)

    risk_free_rate = us_repo_rate() / 100
    df_options = append_black_scholes(df_options, risk_free_rate)
    df_options = get_xPrice(df_options)
    df_options['xPrice'] = df_options['xPrice'].clip(lower=config['MINOPTPRICE'])
    return df_options

def generate_option_recommendations(df_options: pd.DataFrame) -> pd.DataFrame:
    df_options['maxProfit'] = (abs(df_options['undPrice'] - df_options['strike']) + df_options['xPrice'])*abs(df_options['position'])
    return df_options

# Main Functions
def get_covered_calls(positions: pd.DataFrame) -> pd.DataFrame:
    df_options = process_options(positions)
    return generate_option_recommendations(df_options)

def process_covered_positions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_positions = get_positions('STK')
    option_positions = get_positions('OPT')
    stock_positions_without_covers = remove_covered_positions(stock_positions, option_positions)
    stock_positions_without_covers['right'] = stock_positions_without_covers['position'].apply(lambda x: 'P' if x < 0 else 'C')
    und_iv_df = get_und_iv(stock_positions_without_covers)
    stock_positions_without_covers = stock_positions_without_covers.join(und_iv_df[['und_iv', 'und_hv']], how='left').rename(columns={'symbol': 'ib_symbol'})
    covered_calls = get_covered_calls(stock_positions_without_covers)
    return stock_positions, stock_positions_without_covers, covered_calls

def get_und_iv(df: pd.DataFrame) -> pd.DataFrame:
    with get_ib(MARKET='SNP') as ib:
        qualified_contracts = ib.run(qualify_me(ib, df.contract.tolist()))
        df = ib.run(get_mkt_prices(ib, qualified_contracts, sleep=15, chunk_size=39))
    df.rename(columns={'conId': 'undId', 'iv': 'und_iv', 'hv': 'und_hv'}, inplace=True, errors='ignore')
    return df

if __name__ == "__main__":
    stock_positions, _, covered_calls = process_covered_positions()

    print(f"\nRecommended Covered Calls with total maxProfit of {covered_calls.maxProfit.sum():.0f}\n")
    pickle_me(covered_calls, ROOT/'data'/'snp_covers.pkl' )

    print(covered_calls.drop(columns=['contract', 'iv', 'hv']))

