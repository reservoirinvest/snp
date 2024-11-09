# Objectives

1. Perpare for morning naked trades
2. Prepare set of utilities that could be common to NSE and SNP
3. Fully independent of IBKR, with ability to hook to IBKR when needed
4. Class (OOP) based with stock and option bots
5. Self-sufficient continuous-monitoring and autonomous option bots
6. Integration to TradingView graph

---

# Rules

## Symbol rules
1. Every valid symbol should have at least one order or position in the portfolio
2. A Symbol without an underlying position should have one naked order
3. An Underlying position should have two options:
   - For Underlying Put shorts: a Covered Call sell and a Protective Put.
   - For Underlying Call Shorts:  a Covered Put sell and a Protective Call buy position
4. Put and Call buys without underlying positions are `orphaned`. They should have closing orders.

## Symbol `States` with colours

- **tbd** : Unknown status of the position/order. [grey]
- **cwd** : Covered with protection. [green]
<br/>

- **reaped** : An naked option position with a closing order. [green]
- **unreaped** : A naked call or put position that doesn't have an open order/position to reap. [yellow]
<br/>

- **uncovered**: A (long/short) stock with no covers, i.e. (call/put) buy positions/orders  [yellow]
<br/>

- **unsowed**: A symbol with no existing positions/orders [white]
- **synthetic**: A symbol without stock (synthetic / straddle), that has more than one option longs/shorts [black]
- **orphaned** : A single long call or put option (positive) position that doesn't have an underlying position. [blue]
<br/>

- **unprotected**: Stock position present with cover but with no protective call or put order or position [light-red]
- **lonely**: Stock position is present but without cover, protective options/orders[light-brown]
</br> 

- **sowing** : Has naked Option orders that neither covers or protects. [blue]
- **covering** : Symbol position is protected with option position and being covered with an order [cream]
- **protecting** : Symbol position is covered with option but not and being protected with an order [pink]
- **perfecting** : Symbol position with open orders for covering and protecting [light-green]

# Dataclass

There are only two things required:

## Positions

```python 
@dataclass
class Portfolio:
    conId: int = 0
    symbol: str = None
    secType: str = "STK" # or "OPT" for options
    expiry: datetime = datetime.now()
    strike: float = 0.0
    right: str = None
    position: float = 0.0
    mktPrice: float = 0.0
    mktVal: float = 0.0
    avgCost: float = 0.0
    unPnL: float = 0.0
    rePnL: float = 0.0
```

## Open Orders
```python
@dataclass
class OpenOrder:
    conId: int = 0
    symbol: str = None
    secType: str = "STK" # or "OPT" for options
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
```

# Main Programs (sequenced)
1. `make_snp_naked_puts()`
   - `fnos()` ... list of fnos (weekly preferred. includes both stocks and index)
   - `bans()` ... banned stocks of the exchange
   - `underlyings()` ... `price()`, `iv()`, `closest_opt_price()` and `closest_margin()`
   - `chains()` ... all option chains limited by a `MAXDTE` that is typically 50 days.
   - `targets()` ... `target_calls()` based on `CALLSTDMULT` and `target_puts()` based on `PUTSTDMULT` with `xPrice`

2. `opt_closures()` - create closing orders based on profitability scaled to dte from `fill_date()`. 

3. `cover_orders()` ... for stock positions with `CCCCP_STDMULT` that is typically 1 SD

4. `protect_orders()` ... for stock positions with `PROTECTSTD` that is tyically 1 SD

5. `placements`:
   - `place_nakeds()` ... place targets, after checking `get_portfolio()` and `get_open_orders()`
   - `place_colsures()` ... for placing closures to existing open naked positions
   - `place_covers()` ... to place cover orders
   - `place_protects()` ... to place protections

## ---- ON DEMAND ----

1. `get_portfolio()` ... with `cushion()`, `pnl()` and `risk()`

2. `get_openorders()` 

3. `fill_date()` ... gets the order fill date from /data/xn_history (or) IB report 

4. `und_history(delta=False)` ... OHLCs of underlyings. Updated in `delta` mode for missing days.

5. `opt_history(delta=False)` ... OHLCs of options. Updated in `delta` mode for missing days.

6. `offline_margin()`: Option to pick up margins from offline

7. `modify_order_price()`: modify an order price - for nakeds

8. `cancel_order()`: cancels an order function for nakeds for an orderId or for a symbol, if it is ACTIVE

## ---- UTILITY PROGRAMS ----

1. `load_config()` which loads configuration from `snp_config.yml`
2. `pickle_me()` to pickle any file
3. `get_pickle()` to get any pickled file
4. `get_ib()` to get the IB connection
5. `get_closest_strike()` to get the closest `strike` price to `undPrice`
6. `safe_strike()` calculate safe nakeds for PUTSTDMULT and CALLSTDMULT
7. `black_scholes()` gets black scholes price for an option with risk-free rate provided
8. `process_in_chunks()` to process ny callable function in chunks with tqdm and func args provided in dict
9. `get_xPrice()` gets expected price for nakeds.

10. `quick_pf()` gets the current portfolio positions.
11. `make_ib_orders()` for making "SELL" or "BUY" orders from a dataframe with `qty` and `xPrice`.
12. `us_repo_rate()` for repo rate
13. `convert_to_utc_datetime` for getting option expiry dte correct
14. `get_dte()` for getting the dte
15. `get_an_option_chain()` for getting option chains
16. `flatten()` for flattening nested lists
17. `chunk_me()` for manipulating ib live requests in chunks and optimize
18. `to_list()` converts any iterble to a list
19. `get_ib()` gets an IB connection for a market
20. `clean_ib_util_df()` cleans util.df to keep only relevant fields. shortens 'expiry' column name 
21. `yes_or_no()` proceed or not
23. `how_many_days_old()` gets file age
24. `get_prec()` sets precision for placing orders. 0.01 for SNP, 0.05 for NSE

<br/>

a. `qualify_me()` to qualify a list of contracts
b. `get_a_price_iv()` gets a price and iv for a contract.
c. `atm_margin_comm()` gets the atm margin and commission

## ---- CONTINUOUS MONITORING ----

### Orchestrator
An orchestrator will be continuously running to check for the following events.
 - ****EVENT****: MARGIN_BREACH (DANGEROUS)
 - ****EVENT****: ORDER_FILL

1. If the margin cushion is lower than 10% all open shorts for non-poisitions will be cancelled
2. If there is an order fill
   - selling price of all open shorts for non-positions will be bumped up with `bump_price()` by about 10%
   - the order fill will be journaled
   - algo will invoke `recalculate()` on xPrice of open naked orders 
   - selling price of all open shorts for non-positions will be modified per re-calculation
   - algo will go to monitor (listening) mode
3. Will schedule requests for information, like `get_portfolio()` in a separate thread.

# Installation notes

## Folder preparation
- make a `project` folder (e.g. `snp`)
- install git with pip in it

## Virtual enviornment management
- Use `pdm` to manage virtual environment
   - use `pyproject.toml`
   - add an empty `.project-root` file at the root for relative imports / paths

### Note:
- For every package to be installed use `pdm add \<package-name> -d` 
   - the `-d` is for development environment

## Run after installation
- First activate venv with `pdm venv activate`

## Using Jupyterlab IDE
- `pdm run jupyter lab .`
    - if browser doesn't load jupyter close cli and run `jupyter lab build `

-  install jupyter extensions
    - `jupyterlab-code-formatter` <i> for `black` and `isort` of imports </i>
    - `jupyterlab-code-snippets` <i> for auto codes like if \__name__ == ...</i>
    - `jupyterlab-execute-time`  <i> for execution times in cells </i>
    - `jupyterlab-git` <i> for controlling git within jupyterlab </i>
    - `jupyterlab-jupytext` <i> for saving notebook to srcipts, pdfs, etc </i>
    - `jupyterlab-plotly` <i> for graphing (alternative to matplotlib) </i>