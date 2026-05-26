import requests
import pandas as pd
import time
from tqdm import tqdm # 진행률을 보기 위한 라이브러리 (pip install tqdm)

# 1. 발급받은 카카오 REST API 키 입력
KAKAO_API_KEY = '09611d17ff9500ed2d94a6d607cf3609'
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
url = "https://dapi.kakao.com/v2/local/search/category.json"

# 2. 따릉이 대여소 기본 정보 불러오기 (위도, 경도 포함)
print("대여소 위도/경도 데이터를 불러옵니다...")
# 예시 파일. 실제 파일명과 컬럼명('대여소_ID', '위도', '경도')에 맞게 수정하세요.
station_df = pd.read_csv('bike_stations.csv', encoding='cp949')

# 결과를 담을 리스트
poi_data = []

print("카카오 API로 주변 시설물(지하철, 학교) 정보를 수집합니다...")
# tqdm을 씌워서 몇 개나 진행되었는지 진행바(Progress bar) 표시
for idx, row in tqdm(station_df.iterrows(), total=len(station_df)):
    station_id = row['대여소_ID']
    # 주의: 카카오 API는 x에 경도(Longitude), y에 위도(Latitude)를 넣습니다!
    lng = str(row['경도']) 
    lat = str(row['위도'])
    
    # ----------------------------------------------------
    # A. 반경 500m 이내 지하철역(SW8) 개수 조회
    # ----------------------------------------------------
    params_subway = {
        'category_group_code': 'SW8',
        'x': lng, 'y': lat,
        'radius': 500, # 500 미터
    }
    res_subway = requests.get(url, headers=headers, params=params_subway)
    subway_count = 0
    if res_subway.status_code == 200:
        # 카카오 API는 친절하게도 meta.total_count 안에 총 개수를 줍니다.
        subway_count = res_subway.json()['meta']['total_count']
        
    # ----------------------------------------------------
    # B. 반경 500m 이내 학교(SC4) 개수 조회
    # ----------------------------------------------------
    params_school = {
        'category_group_code': 'SC4',
        'x': lng, 'y': lat,
        'radius': 500, 
    }
    res_school = requests.get(url, headers=headers, params=params_school)
    school_count = 0
    if res_school.status_code == 200:
        school_count = res_school.json()['meta']['total_count']
        
    # 데이터 저장
    poi_data.append({
        '대여소_ID': station_id,
        '지하철역_수_500m': subway_count,
        '학교_수_500m': school_count
    })
    
    # 카카오 서버 과부하(차단)를 막기 위해 아주 짧게 휴식
    time.sleep(0.05) 

# 3. 수집된 POI 데이터를 데이터프레임으로 변환
poi_df = pd.DataFrame(poi_data)
print("\nAPI 수집 완료! 데이터 샘플:")
print(poi_df.head())

# ---------------------------------------------------------
# 4. 기존 머신러닝 데이터와 최종 병합
# ---------------------------------------------------------
print("\n기존 머신러닝 데이터와 병합합니다...")
# 이전에 만든 ML 데이터 불러오기
ml_df = pd.read_csv('ml_ready_bike_data_basic.csv') 

# 대여소_ID 기준으로 결합
final_ml_df = pd.merge(ml_df, poi_df, on='대여소_ID', how='left')

# 저장
final_ml_df.to_csv('ml_ready_bike_data_with_api_poi.csv', index=False, encoding='utf-8-sig')
print("🎉 카카오 API 기반 공간 변수 추가 및 최종 저장 완료!")