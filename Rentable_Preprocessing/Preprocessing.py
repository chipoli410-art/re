import pandas as pd
import glob
import os

# origin_data 폴더
folder_path = "origin_data"

# 모든 csv 파일 찾기
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

print(f"총 {len(csv_files)}개의 파일 발견")

df_list = []

# 요일 변환용
weekday_map = {
    0: '월요일',
    1: '화요일',
    2: '수요일',
    3: '목요일',
    4: '금요일',
    5: '토요일',
    6: '일요일'
}

for file in csv_files:
    print(f"읽는 중: {file}")

    try:
        df = pd.read_csv(file, encoding='cp949')
    except:
        df = pd.read_csv(file, encoding='utf-8')

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 빈 행 제거
    df = df.dropna(how='all')

    # 필요한 컬럼 선택
    df = df[['일시', '대여소번호', '대여소명', '시간대', '거치대수량']]

    # 날짜 변환
    df['일시'] = pd.to_datetime(df['일시'])

    # 숫자형 변환
    df['대여소번호'] = pd.to_numeric(df['대여소번호'], errors='coerce')
    df['시간대'] = pd.to_numeric(df['시간대'], errors='coerce')
    df['거치대수량'] = pd.to_numeric(df['거치대수량'], errors='coerce')

    # 결측치 제거
    df = df.dropna()

    # -----------------------------
    # 전처리
    # -----------------------------

    # 월 생성
    df['월'] = df['일시'].dt.month.astype(str) + '월'

    # 주차 생성
    df['주차'] = df['일시'].dt.isocalendar().week.astype(str) + '주차'

    # 요일 생성
    df['요일'] = df['일시'].dt.weekday.map(weekday_map)

    # 대여소명에서 번호 제거
    df['대여소명'] = df['대여소명'].str.replace(
        r'^\d+\.\s*',
        '',
        regex=True
    )

    # 컬럼 순서 정리
    df = df[
        [
            '주차',
            '월',
            '요일',
            '대여소번호',
            '대여소명',
            '시간대',
            '거치대수량'
        ]
    ]

    # 중복 제거
    df = df.drop_duplicates()

    df_list.append(df)

# 전체 병합
combined_df = pd.concat(df_list, ignore_index=True)

print("전체 데이터 크기:", combined_df.shape)

# 저장
output_path = "preprocessed_data.csv"
combined_df.to_csv(
    output_path,
    index=False,
    encoding='utf-8-sig'
)

print(f"저장 완료: {output_path}")
