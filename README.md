# ByeongJaeGo — MFI + Bollinger Swing

## 추천: 주 1회 + 확정조건 (변수↓)

BB/%b/MFI만 쓰면 흔들립니다. **거래량·RSI·주봉MFI·추세필터**를 붙이세요.  
상세: **[BEST_CONFIRMED_METHOD.md](BEST_CONFIRMED_METHOD.md)**

```bash
python3 run_friday_confirmed.py --grade A --compare
```

- **A급**: 기본 + 거래량1.5 + RSI≤30 + 주MFI\<60 + 하락배열 제외 → 확실  
- **B급**: 기본 + 거래량1.5만 → 수익 유지하면서 최악월 완화  
- **C급**: 기본만 → 비추천(변수 큼)

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
