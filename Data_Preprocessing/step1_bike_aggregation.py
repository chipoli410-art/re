# step1_bike_aggregation.py
import pandas as pd
import itertools
import glob
import os
from tqdm import tqdm
import warnings

# 불필요한 경고 메시지 숨김
warnings.filterwarnings('ignore')

print("1. [STEP 1] 'origin_data' 폴더 내의 1년 치 원본 파일 탐색 중...")
file_pattern = os.path.join("origin_data", "*.csv")  
file_list = glob.glob(file_pattern)

if not file_list:
    print("❌ 에러: 'origin_data' 폴더를 찾을 수 없거나 파일이 없습니다!")
    exit()

print(f"총 {len(file_list)}개의 파일을 발견했습니다. 전처리를 시작합니다...\n")

all_processed_data = []

# ---------------------------------------------------------
# 2. 모든 파일을 순회하며 개별 전처리 및 메모리 최적화
# ---------------------------------------------------------
# tqdm으로 365개 파일의 진행 상태바 생성
for file_path in tqdm(file_list, desc="개별 파일 집계 중"):
    # 가. 다중 인코딩 시도 로직 (원본 코드의 훌륭한 방어 로직 적용)
    encodings_to_try = ['cp949', 'utf-8', 'utf-8-sig', 'euc-kr']
    df = None
    
    for enc in encodings_to_try:
        try:
            # low_memory=False 옵션 추가로 대용량 파일 경고 방지
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
            
    if df is None:
        print(f"\n[경고] 파일을 열 수 없어 건너뜁니다: {file_path}")
        continue

    # 나. [핵심] '출발시간' 기준 데이터만 필터링 (반납 중복 집계 방지)
    df_depart = df[df['집계_기준'] == '출발시간'].copy()
    
    # 다. 시간대 정제 및 날짜 정수 변환
    df_depart['기준_시간대'] = df_depart['기준_시간대'].astype(str).str.zfill(4).str[:2].astype(int)
    # 안전한 날짜(int) 변환을 위해 기호 제거
    df_depart['기준_날짜'] = df_depart['기준_날짜'].astype(str).str.replace(r'\D', '', regex=True).astype(int)
    # 천 단위 콤마 등 찌꺼기 제거 후 정수 변환
    df_depart['전체_건수'] = df_depart['전체_건수'].astype(str).str.replace(',', '')
    df_depart['전체_건수'] = pd.to_numeric(df_depart['전체_건수'], errors='coerce').fillna(0).astype(int)
    
    # 라. 파일 단위 1차 그룹화 (메모리를 획기적으로 줄이는 핵심 비법)
    df_grouped = df_depart.groupby(['기준_날짜', '기준_시간대', '시작_대여소_ID', '시작_대여소명'])['전체_건수'].sum().reset_index()
    
    all_processed_data.append(df_grouped)

# ---------------------------------------------------------
# 3. 1년 치 데이터 결합 및 컬럼명 변경
# ---------------------------------------------------------
print("\n3. 집계된 1년 치 데이터를 하나의 테이블로 병합 중...")
combined_df = pd.concat(all_processed_data, ignore_index=True)

# 혹시 파일 간 겹치는 날짜/시간이 있을 수 있으므로 최종 그룹화 1회 추가 수행
combined_df = combined_df.groupby(['기준_날짜', '기준_시간대', '시작_대여소_ID', '시작_대여소명'])['전체_건수'].sum().reset_index()

combined_df.rename(columns={
    '기준_날짜': '대여일자',
    '기준_시간대': '대여시간(시)',
    '시작_대여소_ID': '대여소_ID',
    '시작_대여소명': '대여소명',
    '전체_건수': '총_대여건수(Y)'
}, inplace=True)

# ---------------------------------------------------------
# 4. 빈 시간대(수요 0) 1년 치 풀(Full) 패딩 채우기
# ---------------------------------------------------------
print("4. 365일 24시간 전체 그리드 생성 및 제로 패딩 적용 중... (수 분 소요)")

unique_dates = combined_df['대여일자'].unique()
all_hours = list(range(24))
unique_stations = combined_df[['대여소_ID', '대여소명']].drop_duplicates(subset=['대여소_ID'], keep='last')

# 1년 치 날짜 x 24시간 x 전체 대여소의 모든 조합 생성
combinations = list(itertools.product(unique_dates, all_hours, unique_stations['대여소_ID']))
base_df = pd.DataFrame(combinations, columns=['대여일자', '대여시간(시)', '대여소_ID'])

# 기본 틀에 대여소명 매핑
base_df = pd.merge(base_df, unique_stations, on='대여소_ID', how='left')

# 생성된 1년 치 기본 틀에 실제 집계된 데이터 병합
df_final = pd.merge(base_df, combined_df, on=['대여일자', '대여시간(시)', '대여소_ID', '대여소명'], how='left')

# 빈칸 0으로 채우기
df_final['총_대여건수(Y)'] = df_final['총_대여건수(Y)'].fillna(0).astype(int)

# ---------------------------------------------------------
# 5. 정렬 및 최종 파일 저장
# ---------------------------------------------------------
df_final = df_final.sort_values(by=['대여일자', '대여소_ID', '대여시간(시)']).reset_index(drop=True)

output_filename = "step1_aggregated_bike.csv"
print(f"\n5. 최종 데이터 저장 중... (데이터 형태: {df_final.shape})")
df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n🎉 완료! '{output_filename}' 파일이 성공적으로 생성되었습니다.")
print(df_final.head(10))