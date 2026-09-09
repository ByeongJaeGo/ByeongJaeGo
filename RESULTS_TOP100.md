# Top100 Backtest Results (2019-01-01 → run date)

## Rules
- Universe: KOSPI+KOSDAQ trading-value top 100 (Naver, ETF/preferred excluded)
- Entry: daily `MFI(6)<20` & `MFI(13)<20`, weekly `MFI(6)<60`, ≥2 of %b(8/14/20) `<0`
- Exit: weekly MFI≥60 / daily overheat rollover / setup-low stop / 60d max
- Note: request text `MFI<0` is infeasible on standard MFI (0–100). Literal `≤0` → 3 trades only.

## Aggregate (MFI max = 20)

| metric | value |
|--------|------:|
| tickers tested | 100 |
| tickers with trades | 94 |
| total trades | 420 |
| wins / losses | 133 / 287 |
| **win rate** | **31.7%** |
| avg return / trade | +1.98% |
| avg ticker total return (compounded) | +9.68% |
| median ticker total return | -2.48% |
| best / worst ticker total | +171.1% / -26.4% |

## Literal MFI ≤ 0 (request text)

| metric | value |
|--------|------:|
| trades | 3 |
| win rate | 33.3% |
| avg return / trade | -2.62% |

## How to reproduce
```bash
python3 run_top100_backtest.py --start 2019-01-01 --also-literal-zero
```
