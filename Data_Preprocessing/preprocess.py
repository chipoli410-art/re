import pandas as pd
import itertools
import glob
import os

# 1. 1년치 원본 파일들이 들어있는 폴더 경로 지정
file_pattern = "raw_data/*.csv"  
file_list = glob.glob(file_pattern)

print(f"총 {len(file_list)}개의 파일을 발견했습니다. 전처리를 시작합니다...\n")

all_processed_data = [] # 각 일별로 전처리가 끝난 데이터를 모아둘 리스트

# 2. 모든 파일을 순회하며 개별 전처리
for i, file_path in enumerate(file_list):
    try:
        # [✨ 핵심 개선 사항] 다양한 인코딩 방식을 순서대로 시도
        encodings_to_try = ['cp949', 'utf-8', 'utf-8-sig', 'euc-kr']
        df = None
        
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                break  # 에러 없이 성공하면 for문을 빠져나감
            except UnicodeDecodeError:
                continue  # 에러가 나면 다음 인코딩으로 다시 시도
                
        # 모든 인코딩으로도 열리지 않는 심각한 오류 파일인 경우 스킵
        if df is None:
            print(f"[경고] 파일을 열 수 없어 건너뜁니다 (인코딩 문제): {file_path}")
            continue

        # 가. '출발시간' 기준 데이터만 필터링
        df_depart = df[df['집계_기준'] == '출발시간'].copy()
        
        # 나. 시간대 정제 (분 단위 -> 시간 단위)
        df_depart['기준_시간대'] = df_depart['기준_시간대'].astype(str).str.zfill(4).str[:2].astype(int)
        
        # 다. 1차 그룹화 (일별/시간대별 합산)
        df_grouped = df_depart.groupby(['기준_날짜', '기준_시간대', '시작_대여소_ID', '시작_대여소명'])['전체_건수'].sum().reset_index()
        
        # 집계된 가벼운 데이터를 리스트에 추가
        all_processed_data.append(df_grouped)
        
        # 진행 상황 출력 (50개 파일마다)
        if (i + 1) % 50 == 0:
            print(f">>> {i + 1}개 파일 처리 완료...")
            
    except Exception as e:
        print(f"오류 발생 파일 ({file_path}): {e}")

# 3. 리스트에 모인 1년치 데이터를 하나의 거대한 데이터프레임으로 병합
print("\n모든 파일 병합 중...")
combined_df = pd.concat(all_processed_data, ignore_index=True)

# 4. 컬럼명 직관적으로 변경
combined_df.rename(columns={
    '기준_날짜': '대여일자',
    '기준_시간대': '대여시간(시)',
    '시작_대여소_ID': '대여소_ID',
    '시작_대여소명': '대여소명',
    '전체_건수': '총_대여건수(Y)'
}, inplace=True)

# ---------------------------------------------------------
# 5. 빈 시간대 0 채우기 (1년 전체 기간 대상)
# ---------------------------------------------------------
print("빈 시간대(수요 0) 채우기 작업 중... (시간이 조금 걸릴 수 있습니다)")

# 가. 1년 동안 등장한 모든 날짜, 0~23시, 고유 대여소 목록 추출
unique_dates = combined_df['대여일자'].unique()
all_hours = list(range(24))

# 대여소명 변경 등의 이슈를 방지하기 위해 ID를 기준으로 고유값 추출
unique_stations = combined_df[['대여소_ID', '대여소명']].drop_duplicates(subset=['대여소_ID'], keep='last')

# 나. 1년치 날짜 x 24시간 x 전체 대여소의 모든 조합 생성
combinations = list(itertools.product(unique_dates, all_hours, unique_stations['대여소_ID']))
base_df = pd.DataFrame(combinations, columns=['대여일자', '대여시간(시)', '대여소_ID'])

# 다. 기본 틀에 대여소명 매핑
base_df = pd.merge(base_df, unique_stations, on='대여소_ID', how='left')

# 라. 생성된 1년치 기본 틀에 실제 집계된 데이터(combined_df) 병합
df_final = pd.merge(base_df, combined_df, on=['대여일자', '대여시간(시)', '대여소_ID', '대여소명'], how='left')

# 마. 빈칸 0으로 채우기
df_final['총_대여건수(Y)'] = df_final['총_대여건수(Y)'].fillna(0).astype(int)

# 바. 날짜, 대여소, 시간 순으로 예쁘게 정렬
df_final = df_final.sort_values(by=['대여일자', '대여소_ID', '대여시간(시)']).reset_index(drop=True)

# ---------------------------------------------------------
# 6. 최종 파일 저장
# ---------------------------------------------------------
output_filename = "preprocessed_1year_bike_demand.csv"
print(f"\n최종 데이터 저장 중... (데이터 형태: {df_final.shape})")
df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"완료! '{output_filename}' 파일이 성공적으로 생성되었습니다. 🎉")