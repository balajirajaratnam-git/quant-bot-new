# Professional Upgrade Modifications (V8.0)

This document describes the changes made to upgrade the trading bot to professional-grade exit logic and improved entry scoring.

## Summary of Changes

| File | Change Type | Purpose |
|------|-------------|---------|
| `feature_engine.py` | **REPLACE** | Auto-generate `is_real_bar`, add Signal_Strength scoring |
| `config.yaml` | **REPLACE** | Add trailing stop config, adjust risk parameters |
| `state_store.py` | **REPLACE** | Add trailing stop state persistence |
| `exit_logic.py` | **NEW** | Modular exit evaluation (trailing, adverse, etc.) |
| `live_runner.py` | **REPLACE** | Integrate new exit logic (complete file, ready to use) |

**All files are complete replacements** — just copy them over the originals. No manual patching required.

---

## What Was Fixed

### 1. The `is_real_bar` Bug
**Problem**: FeatureEngine required `is_real_bar` column but live_runner didn't add it when fetching from IG or yfinance.

**Solution**: FeatureEngine now auto-generates `is_real_bar=True` if the column is missing. This is safe because live data is always "real" (no forward-filled calendar holes).

### 2. Exit Logic Improvements
**Problem**: Original exit was only RSI >= 50 or time stop. This left money on the table and didn't cut losing trades early.

**Solution**: New exit logic module with:
- **Trailing stops**: After price moves 1 ATR in your favor, stop trails 1.2 ATR behind the high
- **Adverse exit**: If held 3+ days, RSI has recovered to 40+, but price is still underwater → cut the loser
- **Tighter RSI exit**: Changed from 50 to 45 to capture gains earlier

### 3. Entry Quality Scoring
**Problem**: Original scoring was just `(30 - RSI)`. A trade with RSI=29 in a weak trend got the same weight as RSI=15 in a strong trend.

**Solution**: New `Signal_Strength` composite (0-100) that combines:
- How oversold (RSI contribution: 40 pts max)
- Trend quality (30 pts max) 
- RSI momentum recovering (20 pts max)
- Distance from SMA support (10 pts max)

### 4. Risk Parameter Adjustments
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `rsi_entry_threshold` | 30 | 28 | Require deeper oversold for higher conviction |
| `rsi_exit_threshold` | 50 | 45 | Exit earlier to lock in gains |
| `atr_stop_mult` | 2.0 | 1.5 | Tighter stop limits max loss |
| `atr_take_profit_mult` | 2.5 | 2.0 | Take profits more reliably |

---

## How to Apply Changes

### Step 1: Copy All Files to Their Destinations

```
modified_quant_bot/feature_engine.py  →  ig_quant_bot/core/feature_engine.py
modified_quant_bot/config.yaml        →  config.yaml
modified_quant_bot/state_store.py     →  ig_quant_bot/live/state_store.py
modified_quant_bot/exit_logic.py      →  ig_quant_bot/live/exit_logic.py  (NEW)
modified_quant_bot/live_runner.py     →  ig_quant_bot/live/live_runner.py
```

That's it. All files are complete and ready to use as direct replacements.

---

## Understanding the Leverage Question

Your system uses **IG Spread Betting**, which is leveraged by nature. However, your position sizing deliberately limits actual exposure:

| Setting | Effect |
|---------|--------|
| `margin_factor: 0.20` | Instrument allows 5x leverage |
| `per_slot_margin_fraction: 0.15` | You use only 15% of equity as margin per slot |
| `max_slots: 2` | Maximum 2 concurrent positions |

**Net effect**: Each position has notional exposure of about 75% of equity (0.15 / 0.20). With 2 slots, maximum total exposure is ~150% of equity — similar to a 1.5x leveraged ETF, not aggressive 5x leverage.

This is **conservative for spread betting** and appropriate for learning. The risk-per-trade sizing (0.5% of equity) further limits any single trade's impact.

---

## Expected Behavior Changes

### Entries
- Fewer entries (higher bar: RSI < 28 AND Signal_Strength > 50)
- Higher quality entries (composite scoring ranks by conviction)
- Better regime filtering (Trend_Strength provides nuance)

### Exits
- Faster profit-taking (RSI exit at 45 vs 50)
- Protected gains (trailing stop activates after 1 ATR profit)
- Cut losers earlier (adverse exit on failed reversals)
- Tighter max loss (1.5 ATR stop vs 2.0 ATR)

### Risk/Reward Profile
- **Expected improvement**: Smaller losses, slightly smaller wins, better win rate
- **Trade frequency**: Likely 20-30% fewer trades (quality filter)
- **Sharpe improvement**: Estimated 0.1-0.3 increase (less variance in outcomes)

---

## Testing Checklist

1. ✅ Run smoke test: `python scripts/live_papertrade.py --smoke`
2. ✅ Run dry mode during market hours and check decision_log.csv
3. ✅ Verify Signal_Strength appears in logs
4. ✅ Verify trailing stop updates appear for winning positions
5. ✅ Verify adverse exit triggers on failed reversals
6. ✅ Run for 1 week in dry mode before enabling wet mode

---

## Rollback

If issues arise, simply restore the original files from your v13 backup. The state.json format is backward compatible (v2 positions work with v1 code, extra fields are ignored).
