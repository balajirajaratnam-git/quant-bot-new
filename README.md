# IG Quant Bot (UK IG.com) – Paper Trading Framework

This repository is a learning friendly framework for running a simple systematic strategy with **IG.com UK**.

Key goals
- Run once and exit (Task Scheduler or cron friendly)
- Crash safe by design (reconciles positions from IG)
- Low API usage (history caching, per run budgets, backfill fallback)
- Traceability (decision logs explain every skip, entry, and exit)

## Safety modes

The bot supports three layers of safety:
1. **Offline smoke mode**: no IG login, no orders
2. **Dry run live mode**: connects to IG but does not place orders
3. **DEMO live paper mode**: places trades on your DEMO account

## Quick start

1) Create a virtualenv and install dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Set IG credentials in environment variables

```bash
set IG_API_KEY=...
set IG_USERNAME=...
set IG_PASSWORD=...
set IG_ACCOUNT_ID=...
```

3) Run the offline smoke test

```bash
python scripts/live_papertrade.py --smoke --config config.yaml
```

4) Run the IG connectivity check

```bash
python scripts/ig_connectivity_check.py --config config.yaml --ticker QQQ --debug
```

5) Run a dry run live cycle (no orders)

```bash
python scripts/live_papertrade.py --config config.yaml
```

6) Enable DEMO paper trading
- In `config.yaml` set `live.allow_live: true`
- In `config.yaml` set `live.dry_run: false`

Then run the same command again

```bash
python scripts/live_papertrade.py --config config.yaml
```

## Config files

- `config.yaml` is the default single instrument config (QQQ only)
- `config_multi.yaml` is the multi instrument version (QQQ, SPY, GLD)
- `config_A1.yaml` and `config_A2.yaml` are starter configs for backtests and learning

## Avoiding IG ApiExceeded

IG can throttle or block API usage if you call endpoints too often.

This repo reduces API usage using:
- A history cache (`live.history_cache`) stored under `vault/cache/ig_history`
- A per run history budget (`live.api_budget.max_history_calls_per_run`)
- A rate limiter (`live.rate_limit.min_interval_seconds`) and cooldown handling
- Optional yfinance fallback for history when your IG budget is exhausted

Practical advice
- Start with one instrument (QQQ)
- Keep `max_history_calls_per_run` at 1 while learning
- Run the bot on a schedule (hourly or daily), not in a tight loop
- Prefer cached history for repeated experiments


## Margin factors and dynamic margin

`InstrumentCatalog` contains default margin factors for sizing and pre-trade gating.
IG can change margin requirements dynamically in volatile periods so treat these defaults as conservative hints, not guarantees.

Practical guidance

You can override margin factors in config without editing Python:

```yaml
risk:
  margin_factor_override:
    QQQ: 0.50
```
- Keep `risk.per_slot_margin_fraction` modest while learning
- Keep `risk.min_free_margin_buffer` at 0.10 or higher
- If IG rejects an order due to margin, reduce stake and re-run

## Order parameter compatibility

The live adapter uses defensive order placement because different `trading-ig` versions accept slightly different parameter shapes.
This is intentional so you can upgrade dependencies without rewriting the bot.

## Strategy overview (simple by intention)

Entry (long only)
- Regime filter: price above long SMA (bull stable)
- RSI oversold trigger
- Optional minimum signal strength filter

Risk and position sizing
- Margin based cap per slot
- Optional risk per trade cap using ATR stop distance
- Free margin buffer gate to prevent over allocation

Exit
- Base exit: RSI mean reversion threshold
- Time stop (max hold days)
- Optional adverse exit (if oversold trade fails to recover)
- Optional trailing stop using ATR

## Outputs

- `vault/live_state/state.json` stores idempotency keys and position metadata
- `vault/decisions/decision_log.csv` explains every decision
- `vault/runs/<run_id>/` stores trade ledgers and performance artifacts



## Quick parameter sweep (offline backtest)

Run a small sweep of RSI entry thresholds and regime filter to get enough trades for learning:

```bash
python scripts/sweep_rsi_thresholds.py --base-config config_A1.yaml --start 2018-01-01 --rsi-entries 30,35,40,45 --regimes BULL_STABLE,ANY
```


## Entry quality filters (optional)

- `strategy.require_trend_up`: only enter when Close > SMA and SMA_Slope > 0
- `strategy.require_rsi_turn`: only enter when RSI_Slope > 0 (RSI turning up)
- `strategy.min_signal_strength`: minimum composite Signal_Strength score

See `config_A1_edge.yaml` and `config_A2_edge.yaml`.
