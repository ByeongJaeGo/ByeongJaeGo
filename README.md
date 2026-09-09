# ByeongJaeGo — MFI + Bollinger Swing

## 추천: 일봉으로 보고 주 1회만 매매 (수익 최대)

만쥬·수급단타왕처럼 **일봉**을 보되, 매매는 **금요일 1회**.  
상세: **[BEST_DAILY_WEEKLY_METHOD.md](BEST_DAILY_WEEKLY_METHOD.md)**

```bash
python3 run_friday_daily_best.py --positions 1
```

규칙 요약: `%b≤0.15` + `MFI(14)<30` → 다음 주 시가 / -10% / 최대 2주  
백테스트 월평균 **~+5%**, 월 최대 **~+63%** (1종목).

## 주봉만 볼 때

스퀴즈 돌파: **[BEST_WEEKLY_METHOD.md](BEST_WEEKLY_METHOD.md)**

## 매일 볼 수 있는 일봉 스윙

**[BEST_SWING_METHOD.md](BEST_SWING_METHOD.md)**

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
