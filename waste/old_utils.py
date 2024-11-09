


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

CONFIG = load_config(MARKET)

PUTSTDMULT = CONFIG.get("PUTSTDMULT")
CALLSTDMULT = CONFIG.get("CALLSTDMULT")
MINEXPROM = CONFIG.get("MINEXPROM")




def append_black_scholes(df: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    # Vectorized calculations for Black-Scholes pricing
    S, K, T, r = df["undPrice"].values, df["strike"].values, df["dte"].values / 365, risk_free_rate
    
    # Build the sigma series by checking each column in the specified order
    sigma = df.apply(lambda row: row['iv'] if not pd.isna(row['iv']) 
                     else row['hv'] if not pd.isna(row['hv']) 
                     else row['und_iv'] if not pd.isna(row['und_iv']) 
                     else row['und_hv'], axis=1)
    
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

def get_prec(v: float, base: float) -> float:
    try:
        output = round(round((v) / base) * base, -int(math.floor(math.log10(base))))
    except Exception:
        output = None

    return output



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

def append_xPrice(df: pd.DataFrame, MINEXPROM: float) -> pd.DataFrame:
    df = df.drop(columns=["order"], errors="ignore")
    df = get_xPrice(df)

    margin = np.where(df.margin <= 0, np.nan, df.margin)

    rom = df.xPrice * df.lot / margin * 365 / df.dte
    df = df.assign(rom=rom)

    df = df[df.rom > MINEXPROM].reset_index(drop=True)

    df = df.loc[(df.xPrice / df.price).sort_values().index]

    return df

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