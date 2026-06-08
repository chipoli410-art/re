# step3_poi_merge.py
import pandas as pd
import requests
import time
from tqdm import tqdm

print("1. [STEP 3] Step 2 날씨 병합 데이터 로드 중...")
bike_weather_df = pd.read_csv('step2_bike_weather_test.csv')

print("2. 대여소 좌표 데이터(bike_stations.csv) 로드 중...")
station_df = pd.read_csv('bike_stations.csv', encoding='cp949')

print("3. 카카오 API로 1km 이내 시설물 개수 수집 시작...")
KAKAO_API_KEY = '09611d17ff9500ed2d94a6d607cf3609' # 실제 API 키로 변경
url = "https://dapi.kakao.com/v2/local/search/category.json"
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

poi_data = []
for idx, row in tqdm(station_df.iterrows(), total=len(station_df)):
    station_id, lng, lat = row['대여소_ID'], str(row['경도']), str(row['위도'])
    
    # 지하철(SW8), 학교(SC4) 카운트
    for code, col_name in [('SW8', '지하철역_수_1km'), ('SC4', '학교_수_1km')]:
        res = requests.get(url, headers=headers, params={'category_group_code': code, 'x': lng, 'y': lat, 'radius': 1000})
        count = res.json()['meta']['total_count'] if res.status_code == 200 else 0
        
        # 딕셔너리 생성 및 업데이트 방식을 사용하여 한 줄로 처리
        if code == 'SW8': current_poi = {'대여소_ID': station_id, col_name: count}
        else: current_poi[col_name] = count
    
    poi_data.append(current_poi)
    time.sleep(0.05)

poi_df = pd.DataFrame(poi_data)

print("4. 기존 데이터에 공간 정보 병합 중...")
merged_df = pd.merge(bike_weather_df, poi_df, on='대여소_ID', how='left')

output_file = 'step3_bike_weather_poi_test.csv'
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f" ✔️ [STEP 3 완료] 공간 정보 병합 데이터 '{output_file}' 생성됨!")