# ByeongJaeGo — MFI + Bollinger Swing

## 최선 방법 (월 수익 최적화 결과)

자세한 결론·규칙·성적: **[BEST_SWING_METHOD.md](BEST_SWING_METHOD.md)**

한 줄 요약:
- **Bandwidth 스퀴즈** + `MFI(6)<20` + `%b≤0.15` → 중심선 익절 / -7% 손절
- 백테스트 최선 **월평균 ~+5%** (1종목 집중). **평균 월 +20%는 미달성** (640규칙 탐색)

```bash
pip install -r requirements.txt
python3 run_optimize_monthly.py --start 2019-01-01
python3 run_best_swing.py --positions 1
```

---

## Top100 이전 규칙 (참고)

유니버스: 네이버 기준 **코스피+코스닥 거래대금 상위 100** (ETF/우선주 제외).

| 조건 | 규칙 |
|------|------|
| 일봉 MFI | `MFI(6) < 20` **그리고** `MFI(13) < 20` |
| 주봉 MFI | `MFI(6) < 60` |
| %b | 8/14/20 중 **2개 이상** `< 0` |

```bash
python3 run_top100_backtest.py --also-literal-zero
```

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
