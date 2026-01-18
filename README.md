# quant_bot (IG-style research backtester)

This repository contains a research backtester that simulates an IG-style spread-betting account using daily OHLC data.

## What it is (and what it is not)

- **Default data source:** `yfinance` (tickers like `SPY`, `QQQ`, `GLD`). These are **ETFs**.
- **Simulation model:** a simplified IG-style spread model + margin locks/releases + overnight financing.

Because ETFs are not the same as IG index/DFB markets, you should treat the defaults as an **ETF-proxy research setup** unless you replace the price source with true market data aligned to the IG contract you trade.

## Quickstart

1) Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2) Run the backtest from the repo root:

```bash
python -m ig_quant_bot.main
```

3) Output artifacts are written to `vault/runs/<run_id>/` (ledger, fills, trades and metadata).

## Configuration

Edit `config.yaml` to control:
- Universe tickers
- RSI and regime filter
- Costs: adaptive spread, financing and optional fixed commission
- Risk: position sizing, ATR stop/TP and time-stop

## Live paper trading (IG UK DEMO)

This repo includes a "ready enough" live runner that can place orders on IG **DEMO** using the `trading-ig` library.

1) Copy `.env.example` to `.env` and set your IG **DEMO** credentials.

2) Run an **offline smoke test** (does not connect to IG, validates config and logging):

```bash
python scripts/live_papertrade.py --smoke
```

3) Run a **dry-run** live cycle (connects to IG but does not place orders):

```bash
python scripts/live_papertrade.py --config config.yaml
```

4) When you are comfortable with the decisions and logs, set `live.dry_run: false` in `config.yaml`.

Each live run writes:
- `vault/runs/<run_id>/run.log`
- `vault/runs/<run_id>/decision_log.csv` (the full "why did we trade or skip" reasoning)

Compatibility note: different `trading-ig` versions use different parameter names for historical data (for example `numpoints` vs `num_points`). The live adapter handles both.

Optional: validate login and data access before running the live cycle:

```bash
python scripts/ig_connectivity_check.py --config config.yaml --ticker QQQ
```

### Avoiding IG API quota exhaustion (ApiExceededException)

IG REST endpoints are quota-limited. Repeated history fetches across multiple symbols can trigger `ApiExceededException` and, in some cases, you may need to wait for quota to replenish.

This repo protects your quota in three ways:
- **Rate limiting** between IG calls
- **Local history cache** under `vault/cache/ig_history/`
- **Per-run API budget** (`live.api_budget.max_history_calls_per_run`) so a single run cannot spam the history endpoint

Recommended testing workflow:

1) Fetch once and cache for each symbol (run these with a few seconds gap):

```bash
python scripts/ig_connectivity_check.py --config config.yaml --ticker QQQ --write-cache
python scripts/ig_connectivity_check.py --config config.yaml --ticker SPY --write-cache
python scripts/ig_connectivity_check.py --config config.yaml --ticker GLD --write-cache
```

2) After caching, use cached data (no IG history call):

```bash
python scripts/ig_connectivity_check.py --config config.yaml --ticker QQQ --use-cache
```

3) Run the live runner. It will prefer the cache and only refresh a small tail when needed.

If you still hit `ApiExceededException`, reduce `live.api_budget.max_history_calls_per_run`, increase `live.rate_limit.min_interval_seconds` and avoid repeated rapid runs.

## Notes on technique defaults

The updated defaults are intentionally more conservative for daily bars:
- Exit RSI uses a lower threshold (mean-reversion exit)
- Optional ATR stop-loss and take-profit
- Optional max holding period
- Optional rule to avoid holding two instruments from the same group (e.g., `SPY` and `QQQ`)

## Disclaimer

This is research software. It is not financial advice. Validate with your own data, slippage model and brokerage rules before using any strategy live.
