# ByeongJaeGo — MFI + Bollinger %b Swing Backtest

일봉 스윙: **저평가(과매도) 매수 → 과열 매도**.

## Top100 전략 (권장)

유니버스: 네이버 기준 **코스피+코스닥 거래대금 상위 100** (ETF/우선주 제외, 거래대금≈가격×거래량).

| 조건 | 규칙 |
|------|------|
| 일봉 MFI | `MFI(6) < 20` **그리고** `MFI(13) < 20` |
| 주봉 MFI | `MFI(6) < 60` (60 이상 과열권 아님) |
| %b | 기간 **8 / 14 / 20** 중 **2개 이상** `%b < 0` |
| 매도 | 주봉 MFI(6)≥60, 또는 일봉 과열 롤오버, 또는 셋업 저점 종가 손절, 또는 60일 |

> 요청 원문 `(일) MFI(6)(13) < 0` 은 표준 MFI(0~100)에서 **신호가 0건**입니다.  
> 바닥 과매도권으로 **`< 20`** 을 기본 운용합니다. (`--mfi-max 0` 으로 원문 검증 가능)

```bash
pip install -r requirements.txt
python3 run_top100_backtest.py
python3 run_top100_backtest.py --also-literal-zero
python3 run_top100_backtest.py --mfi-max 0          # 원문 <0 (통상 0건)
python3 run_top100_backtest.py --limit 20 --trades
```

결과: `out_top100_universe.csv`, `out_top100_results.csv`

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
