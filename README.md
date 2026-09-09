# ByeongJaeGo — MFI + Bollinger %b Swing Backtest

일봉 스윙: **저평가(과매도) 매수 → 과열 매도**.  
볼린저 **%b** + **MFI** 동시 극단 + 반전 확인.

## 전략 규칙

| 단계 | 조건 |
|------|------|
| 매수 준비 | `%b ≤ 0` **그리고** `MFI ≤ 20` |
| 매수 실행 | 위 + (`%b` 상승 전환 **또는** `MFI` 20 상향 돌파) + 추세 필터 |
| 추세 필터 | 종가 ≥ SMA50 **또는** SMA50 ≥ SMA200 (강한 하락장 매수 회피) |
| 매도 존 | `%b ≥ 0.85` **그리고** `MFI ≥ 80` |
| 매도 실행 | MFI 80 하향 이탈, 또는 `%b` 1→0.8 하락, 또는 매수봉 저점 손절 |
| 타임아웃 | 최대 보유 60거래일 |

지표 기본값: 볼린저(20, 2σ), MFI(14).

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 기본 샘플 (삼성전자, SK하이닉스, 네이버, 현대차, AAPL)
python3 run_backtest.py

# 종목 지정 (한국 6자리는 자동으로 .KS)
python3 run_backtest.py 005930 000660 035420.KQ

# 기간 / 옵션
python3 run_backtest.py 005930 --start 2020-01-01 --trades
python3 run_backtest.py 005930 --relaxed --trades
python3 run_backtest.py 005930 --no-trend-filter --export out_signals
```

`--relaxed`: 매수 `%b≤0.2 & MFI≤30`, 매도 `%b≥0.8 & MFI≥70` (신호 더 많음).  
손절은 **셋업 봉 저점 종가 이탈** 기준입니다.

## 패키지 구조

```
backtest/
  indicators.py   # %b, MFI, SMA
  strategy.py     # 매수/매도 시그널
  engine.py       # 백테스트·성과 요약
  data.py         # OHLCV 다운로드
run_backtest.py   # CLI
```

## 해석 팁

- `total_return_pct`: 신호를 따라 **전액 순차 진입**했을 때 복리 수익 (포지션 사이징 없음).
- `vs_buy_hold_pct`: 동일 기간 Buy&Hold 대비.
- 거래 수가 적으면 통계가 불안정합니다. 여러 종목·긴 기간으로 보세요.
- 과거 성과 ≠ 미래 수익. 실전 전 수수료·슬리피지·유동성을 반영하세요.
