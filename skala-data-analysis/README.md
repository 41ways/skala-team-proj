# 성인 소득(Adult Income) 데이터 분석 파이프라인

미국 인구조사 기반 성인 소득 데이터로 **소득 >50K 여부를 예측**하는 엔드투엔드 분석 파이프라인입니다.
탐색적 분석(EDA) → 통계 가설 검정(CDA) → 분류 모델링 → 인터랙티브 대시보드 → 자동 리포트까지 한 번에 수행합니다.

> 이 저장소는 Kaggle 노트북(`data/adult-income-dataset-with-knn.ipynb`)을 분석한 뒤,
> 기존 실습 스크립트(`skala-data-prac4/광주_4반_길다인.py`)의 코드 컨벤션에 맞춰
> **단일 실행 스크립트로 모듈화**한 결과물입니다.

---

## 목차
1. [프로젝트 구조](#1-프로젝트-구조)
2. [데이터셋 설명](#2-데이터셋-설명)
3. [원본 노트북 분석 — 무엇을 했나](#3-원본-노트북-분석--무엇을-했나)
4. [발견된 개선점](#4-발견된-개선점)
5. [모듈화 설계 — 코드 컨벤션](#5-모듈화-설계--코드-컨벤션)
6. [파이프라인 5단계](#6-파이프라인-5단계)
7. [실행 방법](#7-실행-방법)
8. [실행 결과 — 원본 대비 개선](#8-실행-결과--원본-대비-개선)
9. [작업 이력](#9-작업-이력)

---

## 1. 프로젝트 구조

```
skala-data-final/
├── test.py                              # 메인 파이프라인 (단일 실행 스크립트)  ※ 원래 이름: 광주_4반_길다인.py
├── report_template.j2                   # Jinja2 HTML 리포트 템플릿
├── requirements.txt                     # 의존성 목록
├── README.md                            # 이 문서
├── data/
│   ├── adult.data                       # UCI Adult 원본 (헤더 없음, 32,561행)
│   └── adult-income-dataset-with-knn.ipynb   # 참고용 원본 Kaggle 노트북
└── outputs/                             # 실행 산출물 (자동 생성)
    ├── eda_2x2.png                      # EDA 2×2 대시보드
    ├── income_by_education_gender.html  # Plotly 인터랙티브 차트
    ├── model.joblib                     # 직렬화된 학습 파이프라인
    └── report.html                      # 자동 생성 분석 리포트
```

---

## 2. 데이터셋 설명

- **파일**: `data/adult.data` (UCI Adult / Census Income)
- **크기**: 32,561행, 헤더 없는 CSV (콤마+공백 구분, 결측은 `?`로 표기)
- **목적**: 인구통계 정보로 연소득이 **`>50K`인지 `<=50K`인지** 예측하는 이진 분류
- **컬럼 15개**:

| # | 컬럼 | 설명 | # | 컬럼 | 설명 |
|---|------|------|---|------|------|
| 1 | age | 나이 | 9 | race | 인종 |
| 2 | workclass | 고용 형태 | 10 | gender | 성별 |
| 3 | fnlwgt | 인구 가중치 | 11 | capital-gain | 자본 이득 |
| 4 | education | 학력(문자열) | 12 | capital-loss | 자본 손실 |
| 5 | educational-num | 학력 연수(숫자) | 13 | hours-per-week | 주당 근로시간 |
| 6 | marital-status | 결혼 상태 | 14 | native-country | 출신 국가 |
| 7 | occupation | 직업 | 15 | **income** | **타깃** (`<=50K`/`>50K`) |
| 8 | relationship | 가족 내 관계 | | | |

- 결측치는 `workclass`, `occupation`, `native-country` 세 컬럼에 `?`로 존재.

---

## 3. 원본 노트북 분석 — 무엇을 했나

`data/adult-income-dataset-with-knn.ipynb` (168셀)을 분석한 결과, **사실상 100% EDA + 예측 모델링**이었고 **CDA(확증적 분석)는 전무**했습니다.

### ✅ 했던 것 — EDA
- **데이터 품질 점검**: 결측치(`?`→NaN, 컬럼별 결측 비율, 히트맵), 중복 제거, `df.info()`/`describe()`
- **단변량 분석**: 범주형 11개 컬럼 `value_counts()`, 연령 분포(boxplot·histplot+KDE), income 분포
- **이변량 분석**: income vs age, education vs age, `pairplot`, groupby(workclass/gender)별 소득 평균, 상관 히트맵
- **예측 모델링**: Label Encoding → KNN(k=15) → 정확도 약 75%

### ❌ 안 한 것 — CDA (확증적 분석)
- t-검정, 카이제곱, ANOVA 등 **가설 검정 없음**
- p-value, 신뢰구간, 효과크기 **없음**
- "성별에 따라 소득 차이가 유의한가?"를 **눈으로만** 확인, 통계 검정 없음

> 통계 검정 관련 키워드(scipy/stats/ttest/chi2/anova/p_value 등)를 노트북 전체에서 탐색한 결과 **0건**.

---

## 4. 발견된 개선점

원본 노트북 검토에서 나온 개선 항목 (이번 모듈에 반영):

| 항목 | 문제 | 이번 모듈 반영 |
|------|------|----------------|
| **CDA 부재** | 집단 차이를 눈으로만 봄 | t검정·카이제곱·ANOVA 3종 추가 |
| **스케일링 누락** | KNN(거리 기반)인데 표준화 없음 → 성능 저하 | `StandardScaler` 파이프라인 |
| **클래스 불균형** | 고소득 재현율 6% (정확도에 속기 쉬움) | 층화 분할(stratify) + 소수 클래스 지표 별도 보고 |
| **컬럼 삭제 재검토** | `capital-gain/loss`를 "75%가 0"이라며 삭제 | 고소득과 연관되므로 **피처로 유지** |
| **부수효과** | `inplace=True` 남발 → 셀 순서 의존 | 순수 함수(입력 복사·반환)로 재작성 |

추가로 `educational-num`(수치)이 `education`(문자열)과 동일 정보를 담으므로 **모델에는 수치형만 사용**하고, `education` 원본은 EDA·검정용으로 보존. `fnlwgt`는 표본 가중치라 예측력이 없어 피처에서 제외.

---

## 5. 모듈화 설계 — 코드 컨벤션

`skala-data-prac4/광주_4반_길다인.py`의 컨벤션을 그대로 적용했습니다. 핵심은 **파일을 여러 개로 쪼개는 게 아니라, 단일 `.py` 스크립트를 배너로 구분된 논리적 단계로 나누는 방식**입니다.

- **리치 module docstring**: 목적 / 처리단계 / 아키텍처 / 산출물 / 실행환경 / 실행방법 / 변경이력 / 작성자
- **import**: stdlib 먼저 → 서드파티는 `try/except ImportError`로 감싸 누락 시 설치 안내 후 `sys.exit`, `matplotlib.use("Agg")`
- **전역 설정**: `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` 기준 경로, `TARGET`/피처 리스트 상수
- **커스텀 예외 `DataError`**: 함수는 `raise`만, 종료 판단은 `__main__`에서만
- **헬퍼**: `section()`(로그 배너), `set_korean_font()`(한글 폰트 자동 선택)
- **순수 함수** + 전체 타입힌트 + 한글 docstring(Args/Returns/Raises), 결과는 `dict` 반환하며 `msg`에 해석 문장 포함
- `.copy()`로 SettingWithCopy 방지, 주석은 "왜"를 설명
- **`main()` 오케스트레이터** + `section()` 로깅
- **`__main__` 가드**: `DataError` / `KeyboardInterrupt` / 그 외 `Exception` 3단 방어

### 설계 결정 (prac4와 다른 점)
- **자체 정제, 독립 실행**: prac4는 상류(prac3) 정제 모듈을 importlib로 동적 로드했지만, 이번엔 `adult.data`를 직접 로드·정제 (외부 의존 없음)
- **회귀 → 분류**: prac4는 매출 회귀(LinearRegression), 이번엔 소득 이진 분류(KNeighborsClassifier)
- **prac4와 동일한 5단계 구조** 유지 (EDA·검정·모델·Plotly·Jinja2 리포트)

---

## 6. 파이프라인 5단계

`test.py` 내부 구성 (배너로 구분된 단계):

| 단계 | 함수 | 내용 |
|------|------|------|
| **[0] 적재·정제** | `load_data()`, `clean_data()` | 헤더 없는 원본 로드+컬럼 부여, `?`→NaN, 결측·중복 제거, income 0/1 이진화 |
| **[1] EDA** | `eda_plot()` | 2×2 대시보드 — ① 연령 분포 ② 성별 고소득률 ③ 학력별 고소득률 ④ 수치형 상관 히트맵 |
| **[2] CDA** | `run_ttest()`, `run_chi2()`, `run_anova()` | 소득집단 간 연령차(t검정) · 성별×소득(카이제곱) · 직군별 근로시간(ANOVA) |
| **[3] 모델링** | `build_pipeline()`, `train_eval_save()` | `StandardScaler`+`OneHotEncoder`+`KNN` 파이프라인, 층화 분할, 학습·평가·직렬화·재적재 검증 |
| **[4] 대시보드** | `build_agg()`, `plotly_chart()` | 학력·성별별 고소득률 Plotly 인터랙티브 HTML |
| **[5] 리포트** | `render_report()` | 위 결과를 Jinja2 템플릿(`report_template.j2`)으로 렌더링한 HTML 리포트 |

---

## 7. 실행 방법

```bash
# 1) 의존성 설치 (venv 권장)
pip install -r requirements.txt
# 또는: pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly joblib jinja2

# 2) 실행
python3 test.py
```

실행하면 `outputs/` 아래에 EDA 이미지·직렬화 모델·인터랙티브 HTML·분석 리포트 4종이 생성됩니다.
경로는 실행 위치가 아니라 스크립트 파일(`__file__`) 기준으로 해석되므로 어느 디렉터리에서 실행해도 동작합니다.

- **환경**: Python 3.9+
- **데이터**: `data/adult.data` 필요

---

## 8. 실행 결과 — 원본 대비 개선

스케일링·층화 분할 적용만으로 소수 클래스(고소득) 포착력이 크게 향상되었습니다.

| 지표 | 원본 노트북 (스케일링 X) | 이번 모듈 (스케일링 O) |
|------|:---:|:---:|
| 정확도 | 74.75% | **83.64%** |
| 고소득(>50K) 재현율 | 0.06 | **0.61** |
| 고소득 F1 | 0.11 | **0.65** |
| 고소득 정밀도 | 0.38 | **0.70** |

**통계 검정 결과 (CDA, 모두 p<0.05 유의)**
- **t검정**: 고소득 집단 평균 44.0세 vs 저소득 36.6세 → 연령 차이 통계적으로 유의미
- **카이제곱**: 성별 × 소득 → 독립 아님(연관 있음), chi2=1413.80
- **ANOVA**: 직군(workclass)별 근로시간 차이 유의미, F=134.69

**정제 통계**: 원본 32,561행 → 정제 30,139행 (결측 2,399 · 중복 23 제거), 고소득 비율 24.9%
모델 재로딩 예측 일치 검증: OK

---

## 9. 작업 이력

이 저장소가 만들어진 과정 (대화 순서):

1. **데이터 확인·요약**: `data/adult.data`(원본)와 Kaggle 노트북을 읽고 데이터셋·분석 흐름 요약
2. **EDA/CDA 점검**: 노트북이 한 것(EDA)과 안 한 것(CDA)을 구분, 통계 검정 부재 확인 및 추가 항목 추천
3. **모듈화 계획 수립**: 재사용·테스트 가능한 구조로 재구성 계획 작성
4. **컨벤션 채택**: `skala-data-prac4/광주_4반_길다인.py`의 코드 컨벤션 분석 → 동일 스타일로 단일 스크립트 모듈화
   - 결정: prac4와 동일한 5단계 / 자체 정제(독립 실행) / 동일 명명 컨벤션
5. **구현·검증**: `광주_4반_길다인.py` + `report_template.j2` 작성, 엔드투엔드 실행 성공 및 결과 검증
6. **파일명 변경**: `광주_4반_길다인.py` → `test.py`

> **참고**: 파일명을 `test.py`로 바꿨지만 스크립트 내부 docstring의 "실행 방법" 문구와 변경 이력에는
> 원래 파일명(`광주_4반_길다인.py`)이 남아 있습니다. 실행은 `__file__` 기준이라 `python3 test.py`로 정상 동작합니다.

---

_작성자: 광주캠퍼스 4반 길다인_
