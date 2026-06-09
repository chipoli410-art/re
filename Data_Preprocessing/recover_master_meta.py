# recover_master_meta.py
import pandas as pd
import os

print("⚙️ [Phase 1] 실제 위치 좌표(위도/경도)가 포함된 매핑 테이블 생성을 시작합니다...")

file_25 = 'step3_bike_weather_poi_25.csv'
file_26 = 'step3_bike_weather_poi_26.csv'
file_stations = 'bike_stations.csv' # 좌표가 들어있는 파일

if not (os.path.exists(file_25) and os.path.exists(file_26) and os.path.exists(file_stations)):
    print(f"❌ 에러: 필수 파일들이 누락되었습니다. 폴더를 확인해 주세요.")
    exit()

# 1. Step 4와 동일하게 병합 및 인코딩 기준 생성
df_25 = pd.read_csv(file_25, usecols=['대여소_ID', '대여소명'])
df_26 = pd.read_csv(file_26, usecols=['대여소_ID', '대여소명'])
df_combined = pd.concat([df_25, df_26], ignore_index=True)
df_combined['대여소_ID_num'] = df_combined['대여소_ID'].astype('category').cat.codes

mapping_base = df_combined[['대여소_ID_num', '대여소_ID', '대여소명']].drop_duplicates(subset=['대여소_ID_num'])

# 2. 실제 좌표 데이터 결합 (encoding은 step3와 동일하게 cp949 적용)
print(" 🗺️ bike_stations.csv에서 실제 위도/경도 좌표를 추출하여 결합합니다...")
stations_coords = pd.read_csv(file_stations, encoding='cp949')

# 대여소_ID 기둥을 기준으로 실제 위도, 경도 병합 (Left Join)
mapping_df = pd.merge(mapping_base, stations_coords[['대여소_ID', '위도', '경도']], on='대여소_ID', how='left')

# 혹시 좌표가 누락된 신규 대여소가 있다면 서울시청 좌표로 안전하게 방어
mapping_df['위도'] = mapping_df['위도'].fillna(37.5665)
mapping_df['경도'] = mapping_df['경도'].fillna(126.9780)

# 필요한 기둥만 남기기
mapping_df = mapping_df[['대여소_ID_num', '대여소명', '위도', '경도']].sort_values(by='대여소_ID_num').reset_index(drop=True)

# 3. 초경량 로제타석 저장
mapping_df.to_csv('station_id_mapping.csv', index=False)

print("✨ [성공] 위치 좌표가 포함된 초경량 매핑 테이블 'station_id_mapping.csv' 생성 완료!\n")