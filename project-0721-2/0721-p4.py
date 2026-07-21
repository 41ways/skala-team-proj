"""
════════════════════════════════════════════════════════════════════
════════════════════════════════════════════════════════════════════

머리말
[실습 4] eda 시각화 4종, 통계 검정, sklearn pipeline, plotly
- 설명: 실습3의 iqr 제거본을 이어받아 (1) 2x2 서브플롯 시각화 4종,
        (2) t-test, 카이제곱 통계 검정, (3) sklearn pipeline 구성, 저장,
        (4) plotly 인터랙티브 차트를 순서대로 진행
- 실행환경: 버전 3.11 / matplotlib 3.11.1
                    seaborn 0.13.2
                    plotly 6.9.0,
                    scikit-learn 1.9.0
                    scipy 1.17.1
                    joblib 1.5.3
                    
- 작성자: 정한결
- 변경내역:
        v1 STEP 1 최초 작성 (한글 폰트 감지, 2x2 서브플롯 4종)
        v2 STEP 2 추가 (서울, 부산 t-test / region x category 카이제곱)
        v3 카이제곱에 cramer's v 추가 (표본 95만건대라 p-value만으로는
           연관성 크기를 오해할 수 있어서 반영. 지문 범위 밖 추가 실습)
        v4 STEP 3 추가 (ridge, randomforest 2종 pipeline, 저장, 재로딩)
        v5 STEP 4 추가 (region x category 총매출 그룹막대, plotly 인터랙티브
           html 저장. region은 총매출 내림차순 정렬해서 읽기 쉽게 함)
        v6 STEP 3 데이터 누출 수정 (아래 이슈 사항 참고). 회귀 피처에서
           quantity, unit_price 제거, 누출 포함,제외 R2 비교 실험 추가
        v7 재로딩 검증을 ==에서 np.allclose로 변경. randomforest predict가
           멀티코어에서 트리 예측을 병렬 합산하다 보니 부동소수점 마지막
           자리가 실행마다 흔들려 완전 일치 assert가 걸림 (m5에서 실제 발견,
           1코어 샌드박스에서는 재현 안 됨)

이슈 사항
    - STEP 3 초기 버전은 amount 예측 피처에 quantity, unit_price를 넣었는데
      이건 데이터 누출임. 실습3에서 amount ≈ quantity × unit_price(98% 일치)를
      확인했으니, 이 둘을 넣고 amount를 예측하면 모델이 패턴을 학습한 게 아니라
      곱셈 공식을 근사한 것일 뿐임. 그래서 두 컬럼을 빼고 customer_age, region,
      category, payment_method, customer_gender만 남김. 이러면 R2가 0 근처로 떨어짐 — STEP 2 발견(t-test p=0.51, cramer's v=0.0033)과 정합적임. 누출 포함,제외 R2를 나란히 찍는 실험을 같이 둬서 "정확도가 높다고 항상 좋은 모델은 아니다"를 수치로 남김

데이터 확인 메모
    - 실습3과 동일 파일(sales_100k.csv, bom 있음, 100만행)
    - order_id는 일련번호라 상관행렬에서 제외
    - order_date는 문자열이라 datetime 변환 필요, 2023-01 ~ 2024-12(24개월)
    - 한글 폰트: macOS는 AppleGothic, 이 검증환경은 Noto Sans CJK KR로 이름이
      달라서 하드코딩하면 한쪽에서 깨짐. 설치된 폰트 중에서 자동으로 고름
    - t-test 전에 levene 등분산 검정 먼저 해봄(서울 분산 4.37e12, 부산 분산
      4.37e12, p=0.70) -> 등분산 가정이 성립해서 slice 기본 예제와 동일하게
      ttest_ind(equal_var=True)로 진행. 이분산이었으면 welch's t-test로
      바꿔야 했음(equal_var=False)
    - randomforest는 이 검증환경(1코어) 기준 전체 95만행 학습에 약 3분 걸림.
      n_jobs=-1로 코어를 최대한 씀 -> 코어 여러 개인 환경에서는 훨씬 빠름
      - 검증 환경을 중간에 클라우드 샌드박스(1코어)에서 클로드 코드(로컬 m5,
        풀코어)로 바꿈. randomforest가 onehotencoder 기본 출력(희소행렬)과
        궁합이 안 좋아 병목이 심했고(10만행 기준 희소 60초+ vs 밀집 9.6초),
        무거운 모델을 돌린 직후 다른 모델을 재면 잔여 프로세스 때문에 같은
        코드가 60초 -> 3.5초로 다르게 나오는 문제도 있었음. 풀코어로 옮기며
        반복 검증이 빨라져서 그 여유로 모델 비교 축을 4종(ridge, ridge+
        상호작용항, randomforest, histgb)으로 늘리려 했으나, quantity,
        unit_price 누출 문제를 발견하면서 방향을 바꿔 다시 축소함 (v6 참고)
════════════════════════════════════════════════════════════════════
════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

import numpy as np
import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = Path(__file__).parent / "sales_100k.csv"
GROUP_COLS = ["region", "category"]
AMOUNT_COL = "amount"
NUM_COLS = ["quantity", "unit_price", "customer_age", "amount"]  # order_id는 일련번호라 제외

# STEP 3 회귀 피처. quantity, unit_price는 amount와 곱셈 관계라 누출이므로 뺌 (이슈 사항 참고)
FEATURE_NUM = ["customer_age"]
FEATURE_CAT = ["region", "category", "payment_method", "customer_gender"]
LEAK_COLS = ["quantity", "unit_price"]  # 누출 실험에서만 잠깐 넣어봄 (넣으면 R2가 뻥튀기됨)

# macOS, windows, 리눅스별로 한글 폰트 이름이 달라서 설치된 것 중 첫 번째를 씀
_KOREAN_FONTS = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
_available = {f.name for f in fm.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in _KOREAN_FONTS if f in _available), "sans-serif")
plt.rcParams["axes.unicode_minus"] = False  # 위 폰트들은 마이너스 기호가 깨져서 별도 처리


class DataError(Exception):
    """로딩, 전처리 단계에서 복구가 안 되는 오류. 함수는 raise만 하고
    종료 여부는 진입점에서 정함 (실습3과 동일한 설계)"""


def load_clean_data(path: Path) -> pd.DataFrame:
    """csv를 로딩해 실습3과 동일한 기준(iqr, 그룹키 결측 제외)으로 정제한다.
    파일 없음, 파싱 실패를 원인별로 나눠 잡음"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        raise DataError(f"'{path}' 파일을 못 찾음") from None
    except pd.errors.ParserError as e:
        raise DataError(f"csv 파싱이 깨짐: {e}") from e

    # 실습3과 동일한 순서: iqr 경계는 그룹키 결측을 지우기 전 원본 기준으로 계산.
    # 순서를 바꾸면 사분위수 자체가 달라져서 실습3 산출물과 조금 어긋남
    q1, q3 = df[AMOUNT_COL].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df_clean = df.dropna(subset=GROUP_COLS)
    df_clean = df_clean[df_clean[AMOUNT_COL].between(lower, upper)].copy()
    df_clean["order_date"] = pd.to_datetime(df_clean["order_date"])

    print(f"정제 완료: {len(df):,}행 -> {len(df_clean):,}행 (iqr 상, 하한 [{lower:,.0f}, {upper:,.0f}])")
    return df_clean


def plot_eda_grid(df: pd.DataFrame) -> plt.Figure:
    """2x2 서브플롯으로 히스토그램+kde, 박스플롯, 월별 라인, 상관 히트맵을 그리기.
    지문이 fig, axes = plt.subplots(2,2) 구조를 명시해서 그대로 따름"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.histplot(df[AMOUNT_COL], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("amount 분포 (히스토그램 + kde)")

    sns.boxplot(data=df, x="region", y=AMOUNT_COL, ax=axes[0, 1])
    axes[0, 1].set_title("지역별 amount 박스플롯")
    axes[0, 1].tick_params(axis="x", rotation=45)

    monthly = df.groupby(df["order_date"].dt.to_period("M"))[AMOUNT_COL].sum()
    axes[1, 0].plot(monthly.index.astype(str), monthly.to_numpy(), marker="o")
    axes[1, 0].set_title("월별 총매출 추이")
    axes[1, 0].tick_params(axis="x", rotation=90)

    corr = df[NUM_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])
    axes[1, 1].set_title("수치형 컬럼 상관 히트맵")

    fig.tight_layout()
    return fig


def run_ttest(df: pd.DataFrame, region_a: str = "서울", region_b: str = "부산") -> None:
    """두 지역의 평균 amount 차이를 t-test로 검정하고 p-value 해석까지 출력.
    levene으로 등분산 확인해뒀으니(위 메모 참고) equal_var=True로 진행"""
    a = df.loc[df["region"] == region_a, AMOUNT_COL]
    b = df.loc[df["region"] == region_b, AMOUNT_COL]
    if a.empty or b.empty:
        raise DataError(f"{region_a} 또는 {region_b} 데이터가 없어 t-test 불가")

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=True)
    verdict = "통계적으로 유의미한 차이 있음" if p_value < 0.05 else "유의미한 차이 없음 (우연 범위)"
    print(f"\n[t-test] {region_a}(n={len(a):,}, 평균 {a.mean():,.0f}) vs "
          f"{region_b}(n={len(b):,}, 평균 {b.mean():,.0f})")
    print(f"t={t_stat:.3f}, p={p_value:.4f} -> {verdict} (α=0.05 기준)")


def run_chi_square(df: pd.DataFrame, col_a: str = "region", col_b: str = "category") -> None:
    """두 범주형 변수의 독립성을 카이제곱 검정으로 확인하고 분할표 출력.
    표본이 95만건대라 아주 작은 편차도 p<0.05로 잡히는 경향이 있어서, p-value
    해석만으로는 오해의 소지가 있음. cramer's v(0~1, 연관성 크기)를 같이 봐서
    통계적 유의성과 실질적 연관성 크기를 구분함"""
    table = pd.crosstab(df[col_a], df[col_b])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    verdict = "독립이 아님 (연관 있음)" if p_value < 0.05 else "독립 (연관 없음)"

    n = table.to_numpy().sum()
    min_dim = min(table.shape) - 1
    cramers_v = (chi2 / (n * min_dim)) ** 0.5
    strength = "거의 없음" if cramers_v < 0.1 else ("약함" if cramers_v < 0.3 else "뚜렷함")

    print(f"\n[카이제곱] {col_a} x {col_b} 분할표 ({table.shape[0]} x {table.shape[1]})")
    print(table)
    print(f"chi2={chi2:.3f}, dof={dof}, p={p_value:.4f} -> {verdict} (α=0.05 기준)")
    print(f"cramer's v={cramers_v:.4f} -> 연관성 크기 {strength} "
          f"(표본이 커서 작은 편차도 유의하게 잡혔을 가능성)")


def build_pipeline(model, num_cols: list) -> Pipeline:
    """수치형은 표준화, 범주형은 원핫인코딩하는 전처리기 + 모델을 pipeline으로 묶음.
    num_cols를 인자로 받는 이유: 누출 실험에서 quantity, unit_price를 넣은 버전과
    뺀 버전을 같은 코드로 돌려야 해서 수치형 목록을 밖에서 주입함. 범주형은
    FEATURE_CAT로 고정.
    sparse_output=False로 밀집 배열을 강제함: randomforest는 내부적으로 밀집
    배열이 필요한데 onehotencoder 기본값(희소)을 그대로 두면 변환 병목으로
    학습이 수십 배 느려짐(실측: 10만행 기준 희소는 60초 넘게 걸림, 밀집은 9.6초)"""
    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_CAT),
    ])
    return Pipeline([("prep", preproc), ("reg", model)])


def train_evaluate_save(df: pd.DataFrame) -> None:
    """amount를 타깃으로 ridge, randomforest 두 모델을
    학습, 평가, 저장, 재로딩 수행.
    모델 개수는 지문에 없고 강사가 자유롭게 여러 개 비교해도 된다고 했음
        -> 2종 비교로 반영.

    피처에서 quantity, unit_price를 뺐음: amount ≈ quantity × unit_price(실습3에서
    98% 일치 확인)라 이 둘을 넣으면 모델이 뭔가를 학습한 게 아니라 곱셈 공식을
    근사할 뿐이라 데이터 누출로 판단. 그래서 남는 피처는 customer_age, region, category,
    payment_method, customer_gender뿐(FEATURE_NUM + FEATURE_CAT).

    이 상태면 R2가 0 근처(또는 음수)로 나오는데 버그가 아니라 정직한 결과임.
    STEP 2와 정합적임 — t-test에서 서울·부산 매출 차이가 유의하지 않았고(p=0.51),
    카이제곱에서 region×category 연관성이 거의 없었음(cramer's v=0.0033). 즉 범주 정보만으로는 amount를 못 맞추는 게 당연한 결과라, 낮은 R2가 오히려 앞 단계
    발견과 맞아떨어짐"""
    feature_cols = FEATURE_NUM + FEATURE_CAT
    x = df[feature_cols]
    y = df[AMOUNT_COL]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
    }

    for name, model in models.items():
        pipe = build_pipeline(model, FEATURE_NUM).fit(x_train, y_train)
        r2 = pipe.score(x_test, y_test)
        print(f"\n[{name}] R2 = {r2:.4f} (0 근처 = 범주 피처만으론 amount 설명력 거의 없음)")

        model_path = Path(__file__).parent / f"model_{name}.pkl"
        try:
            joblib.dump(pipe, model_path)
        except OSError as e:
            raise DataError(f"{name} 모델 저장 실패: {e}") from e

        try:
            reloaded = joblib.load(model_path)
        except (OSError, EOFError) as e:
            raise DataError(f"{name} 모델 재로딩 실패: {e}") from e

# 재로딩한 모델이 원본과 동일한 예측을 내는지 확인.
        # ==로 완전 일치를 요구하면 randomforest에서 걸림: predict가 트리 50개
        # 예측을 병렬로 합산하는데, 스레드가 끝나는 순서가 실행마다 달라질 수
        # 있어서 부동소수점 마지막 자리가 흔들릴 수 있음(멀티코어 환경 이슈)
        # ridge는 병렬 합산이 없어 원래 문제 없었음
        pred_before = pipe.predict(x_test[:5])
        pred_after = reloaded.predict(x_test[:5])
        same = np.allclose(pred_before, pred_after, rtol=1e-9, atol=1e-6)
        assert same, (f"{name} 재로딩 후 예측값이 원본과 다름 "
                       f"(최대 차이 {np.abs(pred_before - pred_after).max():.6f})")
        print(f"저장, 재로딩 확인 완료: {model_path.name}")


def run_leakage_experiment(df: pd.DataFrame) -> None:
    """같은 ridge를 quantity, unit_price 포함 버전, 제외 버전으로 각각 돌려 R2를 직접 비교.
    포함 버전은 amount = quantity × unit_price 공식을 베끼는 셈이라 R2가 확 뛰지만,
    이건 새 데이터에 대한 예측력이 아니라 이미 결과에 들어있는 정보를 되읽는 것뿐임.
    정확도가 높다고 항상 좋은 모델은 아니라는 걸 수치로 보여주려는 짧은 실험"""
    y = df[AMOUNT_COL]
    for label, num_cols in [("누출 포함(quantity, unit_price)", FEATURE_NUM + LEAK_COLS),
                            ("누출 제외", FEATURE_NUM)]:
        x = df[num_cols + FEATURE_CAT]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        pipe = build_pipeline(Ridge(alpha=1.0), num_cols).fit(x_train, y_train)
        print(f"[누출 실험/ridge] {label}: R2 = {pipe.score(x_test, y_test):.4f}")
    print("-> 누출 포함이 R2가 훨씬 높지만 곱셈 공식을 베낀 것뿐이라 실제 예측력이 아님. "
          "정확도가 높다고 항상 좋은 모델은 아님")


def plot_sales_bar(df: pd.DataFrame) -> go.Figure:
    """region x category별 총매출을 그룹막대(region 하나 안에 category별로 나란히)로 그린다.
    plotly라 hover로 정확한 값 보기, 범례 클릭으로 category 켜고 끄기가 됨.

    region은 총매출 내림차순으로 정렬함: plotly 기본은 등장 순서라 크기 비교가
    눈에 안 들어와서, 큰 지역부터 왼쪽에 오게 category_orders로 축 순서를 고정.
    합계는 groupby 한 번이면 되니 비용도 거의 없음"""
    agg = df.groupby(GROUP_COLS, as_index=False)[AMOUNT_COL].sum()

    # region을 총매출 큰 순서로 나열 (막대 높이 비교가 바로 되게)
    region_order = (agg.groupby("region")[AMOUNT_COL].sum()
                    .sort_values(ascending=False).index.tolist())

    fig = px.bar(
        agg,
        x="region",
        y=AMOUNT_COL,
        color="category",
        barmode="group",
        category_orders={"region": region_order},
        labels={"region": "지역", "category": "카테고리", AMOUNT_COL: "총매출"},
        title="지역 x 카테고리별 총매출",
    )
    fig.update_layout(legend_title_text="카테고리")
    return fig


def main() -> None:
    df_clean = load_clean_data(CSV_PATH)

    print("\n[STEP 1] eda 시각화 4종 (2x2 서브플롯)")
    fig = plot_eda_grid(df_clean)
    out_path = Path(__file__).parent / "step1_eda_grid.png"
    fig.savefig(out_path, dpi=120)
    print(f"저장 완료: {out_path}")

    print("\n[STEP 2] 통계 검정 (t-test, 카이제곱)")
    run_ttest(df_clean)
    run_chi_square(df_clean)

    print("\n[STEP 3] sklearn pipeline 구성, 학습, 저장, 재로딩")
    train_evaluate_save(df_clean)
    print("\n[STEP 3-1] 데이터 누출 실험 (누출 피처 포함 vs 제외 R2 비교)")
    run_leakage_experiment(df_clean)

    print("\n[STEP 4] plotly 인터랙티브 차트 (region x category 총매출)")
    fig = plot_sales_bar(df_clean)
    html_path = Path(__file__).parent / "step4_sales_bar.html"
    # include_plotlyjs=True로 plotly.js를 html에 내장 -> 인터넷 없이 더블클릭만으로 열림
    fig.write_html(html_path, include_plotlyjs=True)
    print(f"저장 완료: {html_path}")


if __name__ == "__main__":
    try:
        main()
    except DataError as e:
        sys.exit(f"[오류] {e}")
