"""
════════════════════════════════════════════════════════════════════
════════════════════════════════════════════════════════════════════

머리말
[실습 3] pandas eda, polars lazy api, duckdb sql 비교
- 설명: sales_100k.csv 대상으로 (1) eda + iqr 이상치 제거, (2) pandas groupby
        named aggregation, (3) polars lazy api, (4) duckdb sql로 동일 집계를
        각각 짜서 결과가 같은지 확인하고, (5) timeit으로 세 도구 평균
        실행시간까지 비교
- 실행환경: 버전 3.11 (코드는 3.9에서도 돌아가게 작성)
- 작성자: 정한결
- 변경내역:
        v1 최초 작성 (eda,iqr 이상치 제거, 3도구 동일 집계, timeit 비교)
        v2 주석 태그 정리
        v3 예외를 원인별로 세분화, try,except,else 구조 적용
        v4 결과 일치 assert 검증 추가, 주석 정리
        v5 실제 데이터 확인 후 반영 — 그룹 컬럼 결측 행 제거를 세 도구에
           공통 적용(도구별 null 처리 차이 때문), 검증도 mean, count로 확대
        v6 엔지니어링 보강, 최종 수정

이슈 사항 (지문 범위를 넘어서 의도적으로 구현하지 않은 것)
    - csv 경로를 cli 인자로 안 받고 스크립트 옆 고정 파일로 봄. 과제가 파일
      하나만 다루는 범위라 인자 파싱까지는 범위 밖이라 판단.
    - amount 결측치는 따로 dropna 안 함. between 비교에서 nan, null이
      자연스럽게 false 처리돼서 세 도구 모두 똑같이 걸러지는 걸 그대로 씀.
    - unit_price, quantity, customer_age 쪽 이상치는 안 건드림. 지문이
      집계 대상으로 지목한 건 amount 하나라 거기까지만 처리함.
    - assert 실패는 따로 안 잡음. 입력 문제인 dataerror와 달리 assert가
      깨지는 건 세 도구가 다른 집계를 했다는 뜻이라 트레이스백을 그대로
      보는 게 원인 찾기에 낫다고 판단.

데이터 확인 메모 (실제 파일을 보고 알게 된 것)
    - 파일명은 100k인데 실제로는 100만행, 11개 컬럼임
    - region 10000행, category 8000행, amount 5000행이 결측
    - region, category 결측을 그냥 두면 세 도구 결과가 어긋남.
      pandas groupby는 dropna=True가 기본이라 결측 그룹을 알아서 빼는데
      polars group_by랑 duckdb group by는 null도 하나의 그룹으로 잡아서
      pandas만 64개 나머지 81개로 갈림. 어느 지역인지 모르는 행을 특정
      지역 매출로 칠 수는 없으니까 세 파이프라인 모두에서 명시적으로
      결측 행을 빼고 시작하도록 통일함
      
    - 파일 맨 앞에 bom이 붙어 있음. utf-8-sig로 읽어서 첫 컬럼명이
      깨지지 않게 함

════════════════════════════════════════════════════════════════════
════════════════════════════════════════════════════════════════════
"""

import sys
import timeit
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl

# 실행 위치가 아니라 스크립트 위치를 기준으로 잡아야 어디서 돌려도 파일을 찾음
CSV_PATH = Path(__file__).parent / "sales_100k.csv"
GROUP_COLS = ["region", "category"]
AMOUNT_COL = "amount"
N_REPEAT = 5  # 한 세트당 반복 횟수. 3개 도구 모두 같은 값 써야 하고 1회만은 안 됨
N_SETS = 3    # 세트 수. 첫 세트는 워밍업 탓에 느려서 세트를 나눠 재고 최솟값을 씀


class DataError(Exception):
    """로딩, 검증 단계에서 복구가 안 되는 오류.
    함수는 raise만 하고 프로세스를 죽일지는 진입점에서 정함.
    함수 안에서 sys.exit를 부르면 다른 데서 가져다 쓰거나 테스트할 수가 없어서 분리함"""


# ══════════════════════════════════
# STEP 1 · 로딩, eda
# ══════════════════════════════════

def load_and_explore(path: Path) -> pd.DataFrame:
    """csv 로딩하고 구조(df.info)랑 결측치 출력.
    파일 없는 경우랑 파싱 깨지는 경우를 원인별로 나눠서 잡음"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        raise DataError(f"'{path}' 파일을 못 찾음")
    except pd.errors.ParserError as e:
        raise DataError(f"csv 파싱이 깨짐: {e}")
    else:
        print("=" * 70, "\n[STEP 1] 기본 eda\n", "=" * 70, sep="")
        df.info()
        print("\n결측치 개수:\n", df.isnull().sum())
        return df


# ══════════════════════════════════
# STEP 2 · iqr 이상치 제거
# ══════════════════════════════════

def compute_iqr_bounds(df: pd.DataFrame, col: str = AMOUNT_COL) -> tuple:
    """iqr 공식으로 정상 범위(하한, 상한) 계산.
    이 값을 세 파이프라인에 똑같이 넘기기"""
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def remove_outliers(df: pd.DataFrame, lower: float, upper: float, col: str = AMOUNT_COL) -> pd.DataFrame:
    """정상 범위 벗어난 행 제거하고 제거 전, 후 행 수 출력"""
    before = len(df)
    df_clean = df[df[col].between(lower, upper)].copy()
    print(f"\n정상 범위: [{lower:,.0f}, {upper:,.0f}]")
    print(f"제거 전, 후 행 수: {before:,} -> {len(df_clean):,} (제거 {before - len(df_clean):,}행)")
    return df_clean
# ══════════════════════════════════
# STEP 3 · 세 도구로 동일 집계
# ══════════════════════════════════

def pandas_pipeline(path: Path, lower: float, upper: float) -> pd.DataFrame:
    """pandas groupby + named aggregation.
    agg({'amount': 'sum'})처럼 뭉뚱그리지 않고 total=('amount','sum') 형태로
    써야 컬럼명이 원하는 대로 나와서 나중에 참조하기 편함"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=GROUP_COLS)
    df = df[df[AMOUNT_COL].between(lower, upper)]
    return (
        df.groupby(GROUP_COLS)
        .agg(total=(AMOUNT_COL, "sum"), mean=(AMOUNT_COL, "mean"), count=(AMOUNT_COL, "count"))
        .reset_index()
        .sort_values("total", ascending=False)
    )


def polars_pipeline(path: Path, lower: float, upper: float) -> pl.DataFrame:
    """polars lazy api로 동일 집계.
    read_csv(eager)가 아니라 scan_csv(lazy)로 시작해야 하고, lazy 쪽은
    collect를 부르기 전까진 실행 계획만 쌓이고 실제 연산이 안 일어나는
    구조라 마지막에 꼭 collect를 불러줘야 결과가 나옴.
    drop_nulls는 pandas의 dropna(subset=...)와 같은 자리"""
    return (
        pl.scan_csv(path)
        .drop_nulls(subset=GROUP_COLS)
        .filter(pl.col(AMOUNT_COL).is_between(lower, upper))
        .group_by(GROUP_COLS)
        .agg(
            pl.col(AMOUNT_COL).sum().alias("total"),
            pl.col(AMOUNT_COL).mean().alias("mean"),
            pl.col(AMOUNT_COL).count().alias("count"),
        )
        .sort("total", descending=True)
        .collect()
    )


def duckdb_pipeline(path: Path, lower: float, upper: float) -> pd.DataFrame:
    """duckdb sql group by로 동일 집계 짜서 dataframe으로 반환.
    read_csv_auto로 파일을 바로 읽어서 별도 로딩 단계 없이 sql 한 번에 처리.
    경로랑 경계값은 f-string으로 문자열에 박지 말고 $이름으로 바인딩해서 넘김.
    파일 접근 문제(ioexception)랑 컬럼명이 안 맞는 스키마 문제(binderexception)를
    구분해서 잡아야 뭐가 잘못됐는지 바로 알 수 있어서 나눠 잡음"""
    query = f"""
        SELECT region, category,
               SUM({AMOUNT_COL}) AS total, AVG({AMOUNT_COL}) AS mean, COUNT({AMOUNT_COL}) AS count
        FROM read_csv_auto($path)
        WHERE region IS NOT NULL AND category IS NOT NULL
          AND {AMOUNT_COL} BETWEEN $lower AND $upper
        GROUP BY region, category
        ORDER BY total DESC
    """
    params = {"path": str(path), "lower": lower, "upper": upper}
    try:
        result = duckdb.sql(query, params=params).df()
    except duckdb.IOException as e:
        raise DataError(f"duckdb가 파일에 접근을 못함: {e}")
    except duckdb.BinderException as e:
        raise DataError(f"쿼리의 컬럼명이 실제 csv 구조랑 안 맞음: {e}")
    except duckdb.Error as e:
        raise DataError(f"duckdb 쿼리 실행 실패: {e}")
    else:
        return result
    # ══════════════════════════════════
# STEP 4 · 결과 검증
# ══════════════════════════════════

def validate_results(pandas_df: pd.DataFrame, polars_df: pl.DataFrame, duckdb_df: pd.DataFrame) -> None:
    
    
    """세 결과가 진짜 같은 집계인지 확인.
    지문에서 collect해서 똑같은 결과 나오는지 확인해달라고 짚었던 부분이라
    육안 확인 대신 assert로 자동 검증하게 함.
    polars 쪽은 pyarrow 없이도 돌게 to_pandas 대신 to_list로 값만 꺼냄.
    count는 정수라 완전히 같아야 하지만 total, mean은 부동소수 덧셈 순서가
    도구마다 달라서 끝자리가 미세하게 갈릴 수 있음. 그래서 절대 오차가 아니라
    값 크기 대비 상대 오차(1e-9)로 비교함. 여기 total이 수천억 단위라
    절대 오차로 잡으면 정상인데도 걸림.
    """
    
    
    
    pandas_sorted = pandas_df.sort_values(GROUP_COLS).reset_index(drop=True)
    duckdb_sorted = duckdb_df.sort_values(GROUP_COLS).reset_index(drop=True)
    polars_sorted = polars_df.sort(GROUP_COLS)

    # 8개 지역 8개 카테고리 = 64 결측 그룹을 뺐으니 세 도구 모두 정확히 64여야 함
    n = len(pandas_sorted)
    assert n == 64, f"그룹 수가 64가 아님 — pandas {n}"
    assert len(polars_sorted) == 64, f"그룹 수가 64가 아님 — polars {len(polars_sorted)}"
    assert len(duckdb_sorted) == 64, f"그룹 수가 64가 아님 — duckdb {len(duckdb_sorted)}"

    keys = list(zip(pandas_sorted["region"], pandas_sorted["category"]))
    for col in ("total", "mean", "count"):
        base = pandas_sorted[col].to_list()
        for name, other in (("polars", polars_sorted[col].to_list()), ("duckdb", duckdb_sorted[col].to_list())):
            rels = [abs(a - b) / max(abs(a), 1) for a, b in zip(base, other)]
            i = rels.index(max(rels))
            assert rels[i] < 1e-9, (
                f"{col} 불일치 — {keys[i]}: pandas {base[i]}, {name} {other[i]} (상대오차 {rels[i]})"
            )

    print(f"\n[검증] 세 도구 집계 결과 일치 확인 (그룹 {n}개, total, mean, count 상대 오차 1e-9 미만)")


# ══════════════════════════════════
# STEP 5 · 실행시간 비교
# ══════════════════════════════════

def compare_speed(path: Path, lower: float, upper: float) -> None:
    """세 파이프라인을 같은 조건으로 timeit 재서 1회 평균 실행시간 비교.
    반복 횟수를 도구들마다 똑같이 N_REPEAT 하나를 공유함.
    한 세트만 재면 첫 실행의 워밍업이나 다른 프로세스 간섭이 그대로 섞여서,
    세트를 N_SETS번 나눠 재고 그중 간섭이 가장 적었던 세트를 씀"""
    print(f"\n[STEP 5] timeit {N_REPEAT}회 x {N_SETS}세트, 세트 최솟값 기준 1회 평균")

    tools = {
        "pandas": lambda: pandas_pipeline(path, lower, upper),
        "polars": lambda: polars_pipeline(path, lower, upper),
        "duckdb": lambda: duckdb_pipeline(path, lower, upper),
    }
    times = {}
    for name, fn in tools.items():
        sets = timeit.repeat(fn, number=N_REPEAT, repeat=N_SETS)
        times[name] = min(sets) / N_REPEAT
        print(f"{name:<7} 평균 실행시간: {times[name]:.5f}초")

    fastest = min(times, key=times.get)
    slowest = max(times, key=times.get)
    print(f"\n가장 빠른 도구: {fastest} ({times[fastest]:.5f}초), "
          f"{slowest} 대비 약 {times[slowest] / times[fastest]:.1f}배")


def main() -> None:
    df = load_and_explore(CSV_PATH)
    lower, upper = compute_iqr_bounds(df)
    remove_outliers(df, lower, upper)

    pandas_result = pandas_pipeline(CSV_PATH, lower, upper)
    polars_result = polars_pipeline(CSV_PATH, lower, upper)
    duckdb_result = duckdb_pipeline(CSV_PATH, lower, upper)

    print("\n[STEP 3-1] pandas\n", pandas_result, sep="")
    print("\n[STEP 3-2] polars lazy\n", polars_result, sep="")
    print("\n[STEP 3-3] duckdb sql\n", duckdb_result, sep="")

    validate_results(pandas_result, polars_result, duckdb_result)
    compare_speed(CSV_PATH, lower, upper)


if __name__ == "__main__":
    # 종료 판단은 여기서만 함. 함수들은 dataerror를 올리기만 하고 직접 죽지 않음
    try:
        main()
    except DataError as e:
        sys.exit(f"[오류] {e}")
        
# ══════════════════════════════════
# 실행 결과
# ══════════════════════════════════
# 정상 범위: [-3,999,504, 8,685,056]
# 제거 전, 후 행 수: 1,000,000 -> 973,806 (제거 26,194행)
# 그룹 64개 (8지역 8카테고리), 세 도구 결과 일치
#
# timeit 5회 x 3세트, 세트 최솟값 기준 1회 평균
#   pandas  0.54628초
#   polars  0.03677초
#   duckdb  0.07795초
#   -> polars가 pandas보다 약 14.9배 빠름

# 해석: pandas는 100만행을 전부 메모리에 올린 뒤 필터링해서 가장 느림.
#       polars는 lazy라 실행 계획을 먼저 짜고 필요한 컬럼만 읽어서 제일 빠름.
#       duckdb는 sql 엔진이 최적화하지만 결과를 dataframe으로 변환하느라 중간.