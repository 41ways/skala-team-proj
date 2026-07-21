# -*- coding: utf-8 -*-
"""성인 소득 데이터 분석 파이프라인 — 시각화 · 통계검정 · 소득 예측 모델링.

■ 프로젝트 목적
    미국 인구조사 기반 성인 소득 데이터(adult.data, 약 3.2만 건)를 입력으로 받아,
    데이터 이해부터 모델 학습·배포·리포팅까지 한 번에 수행하는 엔드투엔드 분석
    파이프라인이다. 비즈니스 관점의 질문 세 가지에 답하는 것을 목표로 한다.
        · 소득·연령·근로시간은 어떻게 분포하며 집단 간 차이는 어떤가?    → 탐색적 분석
        · 성별·학력에 따라 고소득 비율이 실제로 다른가?                → 통계적 가설 검정
        · 인구통계 속성으로 고소득(>50K) 여부를 얼마나 예측할 수 있는가?  → 분류 모델링

■ 처리 단계
    1) 탐색적 분석(EDA)   : 연령 분포·성별/학력별 고소득률·수치형 상관을 2×2 대시보드로 시각화
    2) 가설 검정          : 소득집단 간 연령차(t검정), 성별·소득 연관성(카이제곱),
                          직군 간 근로시간차(ANOVA)
    3) 소득 예측 모델      : 전처리~분류를 단일 파이프라인으로 학습·평가하고 배포용으로 직렬화
    4) 인터랙티브 대시보드 : 학력·성별별 고소득률을 Plotly로 시각화해 공유용 HTML로 배포
    5) 자동 리포트        : 위 결과를 Jinja2 템플릿으로 렌더링한 HTML 리포트로 생성

■ 아키텍처
    적재·정제는 이 스크립트가 자체적으로 수행한다(외부 정제 모듈 비의존, 단독 실행).
    load_data → clean_data 로 '검증된 원본 → 결측·중복 정제 → 타깃 이진화'를 거친 단일
    정제 데이터셋을 만들고, 이후 시각화·검정·모델링이 모두 '동일하게 정제된 데이터셋'
    위에서 수행되어 결과 간 정합성이 보장된다. 각 처리 단계는 부수효과가 적은 순수
    함수로 분리해 독립 검증이 쉽도록 했다. 거리 기반 모델(KNN)이므로 전처리 단계에서
    수치형을 표준화(StandardScaler)해 스케일 편향을 제거한다. 무거운 I/O는 함수 내부
    에서만 일어나므로 모듈을 임포트하는 것만으로는 비용이 발생하지 않는다.

■ 산출물 (outputs/)
    eda_2x2.png · income_by_education_gender.html · model.joblib · report.html

■ 실행 환경
    Python 3.9+
    설치 : pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly joblib jinja2
    데이터 : data/adult.data (헤더 없는 UCI Adult 원본, 콤마+공백 구분)

■ 실행 방법
    $ python3 광주_4반_길다인.py
    실행하면 outputs/ 아래에 대시보드 이미지·직렬화 모델·인터랙티브 HTML·분석 리포트가 생성된다.

■ 변경 이력
    2026-07-21  길다인  파이프라인 초기 구축
                         - EDA 2×2 대시보드 / t검정·카이제곱·ANOVA / KNN 분류 파이프라인 구성
                         - Plotly 대시보드 및 Jinja2 리포트 자동화
                         - 자체 적재·정제, 표준화 전처리, 예외 처리 정비

작성자 : 광주캠퍼스 4반 길다인
"""

import os
import sys
import base64
from datetime import datetime

# 필수 서드파티 의존성. 미설치 시 스택트레이스 대신 설치 안내를 남기고 종료한다.
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")                    # 파일 저장 전용 백엔드(디스플레이 불필요)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    import joblib
    import plotly.express as px
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError as e:  # noqa: BLE001 - 시작 전 의존성 확인
    sys.exit(
        f"[치명] 필수 라이브러리 누락: {e.name}  →  "
        "pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly joblib jinja2"
    )

# ---- 전역 설정 --------------------------------------------------------------
# 모든 경로는 '실행 위치'가 아니라 '이 파일의 위치'를 기준으로 해석한다.
# 어느 디렉터리에서 실행하든 데이터·산출물·템플릿을 안정적으로 찾기 위함이다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "adult.data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMPLATE_PATH = os.path.join(BASE_DIR, "report_template.j2")

EDA_PNG = os.path.join(OUT_DIR, "eda_2x2.png")
PLOTLY_HTML = os.path.join(OUT_DIR, "income_by_education_gender.html")
MODEL_PATH = os.path.join(OUT_DIR, "model.joblib")
REPORT_HTML = os.path.join(OUT_DIR, "report.html")

# adult.data는 헤더가 없으므로 UCI 규격 순서대로 컬럼명을 직접 부여한다.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "educational-num",
           "marital-status", "occupation", "relationship", "race", "gender",
           "capital-gain", "capital-loss", "hours-per-week", "native-country",
           "income"]
RAW_NA = "?"                                  # 원본에서 결측을 표기하는 문자

# 예측 대상은 소득 구간(>50K 여부)이다. income을 0/1로 이진화해 사용한다.
TARGET = "income"
POSITIVE_LABEL = ">50K"                        # 관심 클래스(고소득)

# 모델 피처 구성. educational-num(수치)이 education(문자열)과 동일 정보를 담으므로
# 모델에는 수치형만 넣어 중복을 피하고, education 원본 컬럼은 EDA·검정용으로 보존한다.
# fnlwgt는 표본 가중치라 소득 예측력이 없어 피처에서 제외한다.
NUM_FEATURES = ["age", "educational-num", "capital-gain", "capital-loss",
                "hours-per-week"]
CAT_FEATURES = ["workclass", "marital-status", "occupation", "relationship",
                "race", "gender", "native-country"]

K_NEIGHBORS = 15                              # KNN 이웃 수


class DataError(Exception):
    """데이터 적재·정제·검증 단계에서 발생하는 '복구 불가' 오류.

    함수는 이 예외를 raise만 하고, 프로세스 종료 여부는 진입점(__main__)에서만
    판단한다. 덕분에 각 함수를 다른 모듈에서 재사용하거나 단위 테스트하기 쉽다.
    """


def section(title: str) -> None:
    """구분선과 함께 섹션 제목을 표준출력에 남긴다(진행 로그용, 반환값 없음)."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def set_korean_font() -> None:
    """실행 환경(OS)에 설치된 한글 폰트를 자동 선택해 차트의 한글 깨짐을 방지한다.

    후보 폰트를 우선순위대로 확인해 처음 발견한 것을 적용한다. 사용 가능한 폰트가
    하나도 없으면 경고만 남기고 진행한다(차트는 그려지되 한글이 깨질 수 있음).
    음수 축 라벨(−)이 네모로 깨지는 문제도 함께 비활성화한다.
    """
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("AppleGothic", "Malgun Gothic", "NanumGothic",
                 "AppleSDGothicNeo", "Noto Sans CJK KR"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[경고] 한글 폰트를 찾지 못함 — 그래프의 한글이 깨질 수 있음")
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================================ #
# 0) 데이터 적재 및 정제 — 검증된 원본 → 결측·중복 제거 → 타깃 이진화
# ============================================================================ #
def load_data(csv_path: str) -> pd.DataFrame:
    """헤더 없는 adult.data를 읽어 규격 컬럼명을 부여한 DataFrame으로 반환한다.

    원본은 콤마 뒤에 공백이 붙는 형식(skipinitialspace)이며 결측은 '?'로 표기된다.
    적재 시점에 공백을 제거하고 '?'를 NaN으로 변환해, 이후 단계가 표준 결측 처리를
    그대로 쓸 수 있게 한다.

    Args:
        csv_path: adult.data 파일 경로.

    Returns:
        15개 컬럼이 부여된 원본 DataFrame.

    Raises:
        DataError: 파일이 없거나, 컬럼 수가 규격과 다르거나, 빈 데이터일 때.
    """
    if not os.path.isfile(csv_path):
        raise DataError(f"데이터 파일을 찾을 수 없음: {csv_path}")

    df = pd.read_csv(csv_path, header=None, names=COLUMNS,
                     skipinitialspace=True, na_values=RAW_NA)
    if df.empty:
        raise DataError("적재 결과가 비어 있음(0행)")
    if df.shape[1] != len(COLUMNS):
        raise DataError(f"컬럼 수 불일치: 기대 {len(COLUMNS)}, 실제 {df.shape[1]}")
    return df


def clean_data(df: pd.DataFrame):
    """결측·중복을 제거하고 타깃(income)을 0/1로 이진화한 분석용 데이터셋을 만든다.

    처리 순서: 결측 행 제거 → 중복 행 제거 → income을 {<=50K:0, >50K:1}로 이진화.
    이후 시각화·검정·모델링의 공통 입력이 되며, 정제 통계는 리포트에 활용한다.

    Args:
        df: load_data가 반환한 원본 DataFrame.

    Returns:
        (df_clean, report): 정제 DataFrame과 정제 통계 dict
        (n_before, n_after, n_missing_dropped, n_dup_dropped, missing_by_col, pos_rate).

    Raises:
        DataError: 정제 후 남은 행이 없을 때.
    """
    df = df.copy()                                       # 원본 뷰 수정 경고(SettingWithCopy) 방지
    n_before = len(df)

    missing_by_col = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    df = df.dropna(how="any")
    n_after_na = len(df)

    df = df.drop_duplicates()
    n_after = len(df)

    if n_after == 0:
        raise DataError("정제 후 학습·분석에 쓸 표본이 없음(0행)")

    # 타깃 이진화: 관심 클래스(>50K)를 1로 둔다. 매핑 밖 값이 있으면 NaN이 되므로 방어적으로 제거.
    df[TARGET] = (df[TARGET] == POSITIVE_LABEL).astype(int)

    report = {
        "n_before": n_before,
        "n_after": n_after,
        "n_missing_dropped": n_before - n_after_na,
        "n_dup_dropped": n_after_na - n_after,
        "missing_by_col": missing_by_col,
        "pos_rate": float(df[TARGET].mean()),            # 고소득(>50K) 비율
    }
    return df, report


# ============================================================================ #
# 1) 탐색적 데이터 분석(EDA) — 2×2 대시보드
# ============================================================================ #
def eda_plot(df: pd.DataFrame, out_png: str) -> str:
    """탐색적 분석 4종을 하나의 2×2 대시보드로 그려 이미지로 저장한다.

    ① 연령 분포(히스토그램+KDE)       ② 성별 고소득률(막대)
    ③ 학력별 고소득률(정렬 막대)      ④ 수치형 변수 상관(히트맵)
    네 지표를 한 figure에 모아 데이터의 규모·집단편차·상관을 한눈에 비교한다.

    Args:
        df: 정제된(income 0/1) DataFrame.
        out_png: 저장할 이미지 경로.

    Returns:
        저장된 이미지 경로.
    """
    set_korean_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(df["age"], kde=True, bins=40, ax=axes[0, 0])
    axes[0, 0].set_title("① 연령(age) 히스토그램 + KDE")

    gender_rate = df.groupby("gender")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=gender_rate.index, y=gender_rate.values, ax=axes[0, 1])
    axes[0, 1].set_title("② 성별 고소득(>50K) 비율")
    axes[0, 1].set_ylabel("고소득 비율")

    edu_rate = df.groupby("education")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=edu_rate.index, y=edu_rate.values, ax=axes[1, 0])
    axes[1, 0].set_title("③ 학력별 고소득(>50K) 비율")
    axes[1, 0].set_ylabel("고소득 비율")
    axes[1, 0].tick_params(axis="x", rotation=90)

    sns.heatmap(df[NUM_FEATURES + [TARGET]].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title("④ 수치형 상관 히트맵")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)                                       # figure 자원 즉시 해제(반복 호출 시 메모리 누수 방지)
    return out_png


# ============================================================================ #
# 2) 가설 검정 — t검정 · 카이제곱 · ANOVA
# ============================================================================ #
def run_ttest(df: pd.DataFrame, group_col: str, value_col: str,
              group_a, group_b, alpha: float = 0.05) -> dict:
    """두 그룹의 평균 차이를 Welch t검정(equal_var=False)으로 검정한다.

    두 집단의 분산이 다를 수 있으므로 Welch 방식을 사용한다. 반환 dict의 msg에는
    유의수준 대비 판단(유의미 여부) 문장을 함께 담아, 수치와 해석을 같이 제공한다.

    Args:
        group_col: 그룹을 나누는 컬럼(예: income).
        value_col: 비교할 수치 컬럼(예: age).
        group_a, group_b: 비교할 두 그룹 값(예: 0, 1).
        alpha: 유의수준(기본 0.05).

    Returns:
        dict(t, p, mean_a, mean_b, n_a, n_b, significant, msg)

    Raises:
        DataError: 두 그룹 중 하나라도 표본이 비어 검정이 불가능할 때.
    """
    a = df.loc[df[group_col] == group_a, value_col].dropna()
    b = df.loc[df[group_col] == group_b, value_col].dropna()
    if a.empty or b.empty:
        raise DataError(f"t검정 표본 부족: {group_a}={a.size}, {group_b}={b.size}")

    t, p = stats.ttest_ind(a, b, equal_var=False)
    significant = bool(p < alpha)
    msg = (f"p={p:.4g} < {alpha} → 두 그룹의 평균 {value_col} 차이가 통계적으로 유의미"
           if significant else
           f"p={p:.4g} ≥ {alpha} → 평균 {value_col} 차이가 유의미하지 않음(우연 수준)")
    return {"t": float(t), "p": float(p),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "n_a": int(a.size), "n_b": int(b.size),
            "significant": significant, "msg": msg}


def run_chi2(df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> dict:
    """두 범주형 변수의 독립성을 카이제곱 검정으로 확인한다.

    카이제곱 검정은 합계가 아니라 빈도를 입력으로 하므로 교차표(crosstab)의 빈도를
    사용한다. 반환 dict의 msg에 유의미 여부 해석을 함께 담는다.

    Args:
        col1, col2: 연관성을 볼 두 범주형 컬럼(예: gender, income).
        alpha: 유의수준(기본 0.05).

    Returns:
        dict(chi2, p, dof, significant, msg)

    Raises:
        DataError: 교차표가 비어 검정이 불가능할 때.
    """
    table = pd.crosstab(df[col1], df[col2])
    if table.size == 0 or int(table.values.sum()) == 0:
        raise DataError(f"카이제곱 분할표가 비어 검정 불가: {col1} × {col2}")

    chi2, p, dof, _ = stats.chi2_contingency(table)
    significant = bool(p < alpha)
    msg = (f"p={p:.4g} < {alpha} → 두 변수는 독립이 아님(연관 있음)"
           if significant else
           f"p={p:.4g} ≥ {alpha} → 두 변수는 독립(연관 없음)")
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof),
            "significant": significant, "msg": msg}


def run_anova(df: pd.DataFrame, group_col: str, value_col: str,
              alpha: float = 0.05) -> dict:
    """세 개 이상 그룹의 평균 차이를 일원배치 분산분석(ANOVA)으로 검정한다.

    범주(group_col)별로 수치(value_col) 표본을 나눠 f_oneway에 투입한다. 표본이 하나뿐인
    그룹은 분산 계산에서 의미가 없어 제외한다. msg에 유의미 여부 해석을 함께 담는다.

    Args:
        group_col: 그룹을 나누는 범주 컬럼(예: workclass).
        value_col: 비교할 수치 컬럼(예: hours-per-week).
        alpha: 유의수준(기본 0.05).

    Returns:
        dict(f, p, k_groups, significant, msg)

    Raises:
        DataError: 검정에 쓸 유효 그룹이 2개 미만일 때.
    """
    groups = [g[value_col].dropna().values
              for _, g in df.groupby(group_col) if g[value_col].dropna().size > 1]
    if len(groups) < 2:
        raise DataError(f"ANOVA 유효 그룹 부족: {group_col} 기준 {len(groups)}개")

    f, p = stats.f_oneway(*groups)
    significant = bool(p < alpha)
    msg = (f"p={p:.4g} < {alpha} → 그룹 간 평균 {value_col} 차이가 통계적으로 유의미"
           if significant else
           f"p={p:.4g} ≥ {alpha} → 그룹 간 평균 {value_col} 차이가 유의미하지 않음")
    return {"f": float(f), "p": float(p), "k_groups": len(groups),
            "significant": significant, "msg": msg}


# ============================================================================ #
# 3) 소득 예측 모델 — 파이프라인 구성 · 학습 · 평가 · 직렬화 · 재적재 검증
# ============================================================================ #
def build_pipeline(num_features: list, cat_features: list,
                   n_neighbors: int = K_NEIGHBORS) -> Pipeline:
    """전처리와 KNN 분류기를 하나로 묶은 학습 파이프라인을 구성한다.

    ColumnTransformer로 수치형은 표준화(StandardScaler), 범주형은 원-핫 인코딩을
    병렬 적용하고, 그 뒤에 KNN 분류기를 연결한다. KNN은 거리 기반이라 표준화가
    성능에 결정적이며, 전처리와 모델이 한 객체이므로 학습·예측에 동일 변환이 일관되게
    적용되고 직렬화하면 전처리까지 함께 저장된다.
    """
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])
    return Pipeline([
        ("pre", preprocessor),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])


def train_eval_save(df: pd.DataFrame, pipeline: Pipeline,
                    num_features: list, cat_features: list, target: str,
                    model_path: str, test_size: float = 0.2,
                    random_state: int = 42) -> dict:
    """파이프라인을 학습·평가하고 직렬화한 뒤 재적재 결과까지 검증한다.

    처리 순서: 결측 제거 → 층화 분할(클래스 불균형 대응) → 학습 → 예측 → 지표 산출
             (정확도·정밀도·재현율·F1) → joblib 직렬화 → 재적재 → 예측 일치 검증.
    소득 데이터는 고소득 비율이 낮은 불균형 데이터라, 정확도만 보면 오해가 생긴다.
    따라서 관심 클래스(>50K)의 정밀도·재현율·F1을 함께 반환한다.

    Returns:
        dict(accuracy, precision_pos, recall_pos, f1_pos, f1_macro,
             confusion, n_train, n_test, reload_ok, model_path, msg)

    Raises:
        DataError: 결측 제거 후 학습에 쓸 표본이 남지 않을 때.
    """
    features = num_features + cat_features
    data = df.dropna(subset=features + [target])
    if data.empty:
        raise DataError("학습 표본이 비어 있음(결측 제거 후 0행)")

    X, y = data[features], data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    # 관심 클래스(양성=1)의 정밀도·재현율·F1을 따로 뽑는다.
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, labels=[1], average=None, zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)                    # 전처리+모델을 하나의 아티팩트로 직렬화

    reloaded = joblib.load(model_path)                   # 저장본을 다시 읽어
    reload_ok = bool(np.array_equal(                     # 예측 일치 검증
        reloaded.predict(X_test.head(100)), pred[:100]))

    msg = (f"정확도 {accuracy:.3f}이지만 고소득(>50K) 재현율은 {rec[0]:.3f} — "
           "불균형 데이터라 소수 클래스 포착력을 함께 봐야 한다")
    return {"accuracy": float(accuracy),
            "precision_pos": float(prec[0]), "recall_pos": float(rec[0]),
            "f1_pos": float(f1[0]), "f1_macro": float(f1_macro),
            "confusion": cm.tolist(),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "reload_ok": reload_ok, "model_path": model_path, "msg": msg}


# ============================================================================ #
# 4) 인터랙티브 대시보드 — Plotly HTML 배포
# ============================================================================ #
def build_agg(df: pd.DataFrame) -> pd.DataFrame:
    """학력·성별별 고소득(>50K) 비율을 집계한다(대시보드·리포트 공용 입력).

    Returns:
        컬럼(education, gender, income_rate)을 가진 집계 DataFrame.
    """
    agg = (df.groupby(["education", "gender"], as_index=False)[TARGET]
             .mean()
             .rename(columns={TARGET: "income_rate"}))
    return agg


def plotly_chart(agg: pd.DataFrame, out_html: str) -> str:
    """학력·성별별 고소득률을 Plotly 막대 차트로 그려 공유용 HTML로 저장한다.

    결과는 브라우저에서 바로 열어 확대·호버·범례 토글이 가능한 인터랙티브 HTML로
    저장한다.

    Returns:
        저장된 HTML 경로.
    """
    fig = px.bar(
        agg, x="education", y="income_rate", color="gender", barmode="group",
        title="학력·성별별 고소득(>50K) 비율",
        labels={"income_rate": "고소득 비율", "education": "학력", "gender": "성별"},
    )
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html)
    return out_html


# ============================================================================ #
# 5) 자동 리포트 — Jinja2 렌더링
# ============================================================================ #
def _img_data_uri(png_path: str) -> str:
    """이미지를 base64 data URI로 인코딩한다(리포트에 임베드해 단일 파일로 자기완결화)."""
    with open(png_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def render_report(context: dict, template_path: str, out_html: str) -> str:
    """분석 결과(context)를 Jinja2 템플릿에 채워 HTML 리포트를 생성한다.

    렌더링 대상은 사용자 입력이 아닌 분석 산출물이지만, 안전을 위해 autoescape를 켜
    HTML 이스케이프를 적용한다.

    Returns:
        저장된 HTML 경로.

    Raises:
        DataError: 템플릿 파일이 없을 때.
    """
    if not os.path.isfile(template_path):
        raise DataError(f"리포트 템플릿을 찾을 수 없음: {template_path}")

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(template_path)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(os.path.basename(template_path))
    html = template.render(**context)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


# ============================================================================ #
# 파이프라인 오케스트레이션
# ============================================================================ #
def main() -> None:
    """파이프라인 전체(적재·정제 → EDA → 검정 → 모델링 → 대시보드 → 리포트)를 실행한다."""
    section("[0] 데이터 적재 및 정제")
    df_raw = load_data(DATA_PATH)
    df_clean, clean_rep = clean_data(df_raw)
    print(f"* 원본 {clean_rep['n_before']:,}행 → 정제 {clean_rep['n_after']:,}행 "
          f"(결측 {clean_rep['n_missing_dropped']:,} · 중복 {clean_rep['n_dup_dropped']:,} 제거)")
    print(f"* 고소득(>50K) 비율: {clean_rep['pos_rate']:.3f}")

    # [1] 탐색적 분석 2×2 대시보드
    section("[1] EDA 시각화 4종 (2x2 서브플롯)")
    eda_plot(df_clean, EDA_PNG)
    print(f"* 저장: {EDA_PNG}")

    # [2] 가설 검정
    section("[2] 통계 검정 — t검정 · 카이제곱 · ANOVA")
    ttest = run_ttest(df_clean, TARGET, "age", 1, 0)
    print(f"* t검정  고소득(평균 {ttest['mean_a']:.1f}세) vs 저소득(평균 {ttest['mean_b']:.1f}세)"
          f"  → t={ttest['t']:.4f}")
    print(f"  해석: {ttest['msg']}")
    chi2 = run_chi2(df_clean, "gender", TARGET)
    print(f"* 카이제곱  gender × income  → chi2={chi2['chi2']:.2f}, dof={chi2['dof']}")
    print(f"  해석: {chi2['msg']}")
    anova = run_anova(df_clean, "workclass", "hours-per-week")
    print(f"* ANOVA  workclass별 근로시간  → F={anova['f']:.2f}, 그룹수={anova['k_groups']}")
    print(f"  해석: {anova['msg']}")

    # [3] 소득 예측 모델 학습·평가·직렬화·재적재 검증
    section("[3] sklearn Pipeline — 훈련·평가·저장·재로딩")
    pipe = build_pipeline(NUM_FEATURES, CAT_FEATURES)
    metrics = train_eval_save(df_clean, pipe, NUM_FEATURES, CAT_FEATURES, TARGET, MODEL_PATH)
    print(f"* 정확도={metrics['accuracy']:.4f}  고소득 정밀도={metrics['precision_pos']:.3f}"
          f"  재현율={metrics['recall_pos']:.3f}  F1={metrics['f1_pos']:.3f}"
          f"  (train {metrics['n_train']:,} / test {metrics['n_test']:,})")
    print(f"* 저장: {MODEL_PATH}  (재로딩 예측 일치 검증: {'OK' if metrics['reload_ok'] else '실패'})")

    # [4] 인터랙티브 대시보드
    section("[4] Plotly 인터랙티브 차트 저장")
    agg = build_agg(df_clean)
    plotly_chart(agg, PLOTLY_HTML)
    print(f"* 저장: {PLOTLY_HTML}")

    # [5] 자동 리포트 생성
    section("[5] Jinja2 분석 리포트 자동 생성")
    edu_top = (df_clean.groupby("education")[TARGET].mean()
               .sort_values(ascending=False).head(10))
    context = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clean": clean_rep,
        "eda_img": _img_data_uri(EDA_PNG),
        "ttest": ttest, "chi2": chi2, "anova": anova, "metrics": metrics,
        "edu_top": [{"education": k, "income_rate": float(v)} for k, v in edu_top.items()],
        "plotly_file": os.path.basename(PLOTLY_HTML),
        "features": {"numeric": NUM_FEATURES, "categorical": CAT_FEATURES, "target": TARGET},
    }
    render_report(context, TEMPLATE_PATH, REPORT_HTML)
    print(f"* 저장: {REPORT_HTML}")

    section("[완료] 산출물 4종 생성됨")
    for path in (EDA_PNG, PLOTLY_HTML, MODEL_PATH, REPORT_HTML):
        print(f"  - {path}")


if __name__ == "__main__":
    # 최상위 방어선: 종료 여부 판단은 이 진입점에서만 한다(함수는 예외를 raise할 뿐).
    #  - DataError        : 예상된 입력/정제/검증 실패 → 원인 메시지만 간결히 출력
    #  - KeyboardInterrupt: 사용자 중단
    #  - 그 외 Exception  : 예상 밖 오류를 위한 마지막 안전망
    try:
        main()
    except DataError as e:
        sys.exit(f"[치명] {e}")
    except KeyboardInterrupt:
        sys.exit("\n[중단] 사용자에 의해 중단됨")
    except Exception as e:  # noqa: BLE001 - 최상위 포괄 예외 처리
        sys.exit(f"[치명] 예기치 못한 오류: {e}")
