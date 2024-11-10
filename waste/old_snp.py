from ib_async import Index, Stock
import pandas as pd
from snp import make_snp_weeklies
from utils import get_ib, qualify_me


def us_repo_rate():
    return 5.51

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

def assemble_snp_underlyings(live: bool = True, fresh: bool = True) -> pd.DataFrame:
    und_path = ROOT / 'data' / 'snp_unds.pkl'
    if not fresh:
        df = get_pickle(und_path)
        if df is not None and not df.empty:
            return df

    qualified_contracts = get_snp_unds()
    with get_ib(MARKET='SNP') as ib:
        df = df_iv(ib=ib, stocks=qualified_contracts)
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