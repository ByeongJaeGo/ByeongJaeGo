# ByeongJaeGo — MFI + Bollinger Swing

## 추천: BB/%b/MFI/Bandwidth만으로 가장 확실한 매수·매도

상세: **[BEST_BB_UPTREND_METHOD.md](BEST_BB_UPTREND_METHOD.md)**

```bash
python3 run_bb_uptrend_best.py --mode certain   # 가장 확실
python3 run_bb_uptrend_best.py --mode balanced  # 신호 더 많음
```

- **매수:** `%b≤0.25` + `MFI(14)<20` + MFI 상승 전환  
- **매도:** `%b≥0.85 & MFI≥80` / %b 트레일 / -10% / 최대 3주  

## 추가 필터 (거래량·RSI 등)

**[BEST_CONFIRMED_METHOD.md](BEST_CONFIRMED_METHOD.md)** — `run_friday_confirmed.py --grade A|B`

## 일봉·주1회 기본판

**[BEST_DAILY_WEEKLY_METHOD.md](BEST_DAILY_WEEKLY_METHOD.md)**

## 레거시 단일 종목 전략

| 단계 | 조건 |
|------|------|
| 매수 | `%b ≤ 0` & `MFI ≤ 20` + 반전 확인 + 추세 필터 |
| 매도 | `%b ≥ 0.85` & `MFI ≥ 80` 롤오버 / 손절 / 60일 |

```bash
python3 run_backtest.py 005930 --relaxed --trades
```

## 패키지

```
backtest/
  universe.py     # 거래대금 상위 종목
  indicators.py   # %b(8/14/20), MFI(6/13), weekly MFI
  strategy.py     # 시그널
  engine.py       # 백테스트·승률/수익 집계
  data.py         # Yahoo OHLCV
run_top100_backtest.py
run_backtest.py
```

과거 성과 ≠ 미래 수익. 수수료·슬리피지는 미반영입니다.
