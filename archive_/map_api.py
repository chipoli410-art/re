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
    # A. 반경 1km 이내 지하철역(SW8) 개수 조회
    # ----------------------------------------------------
    params_subway = {
        'category_group_code': 'SW8',
        'x': lng, 'y': lat,
        'radius': 1000, # 1000 미터
    }
    res_subway = requests.get(url, headers=headers, params=params_subway)
    subway_count = 0
    if res_subway.status_code == 200:
        # 카카오 API는 친절하게도 meta.total_count 안에 총 개수를 줍니다.
        subway_count = res_subway.json()['meta']['total_count']
        
    # ----------------------------------------------------
    # B. 반경 1km 이내 학교(SC4) 개수 조회
    # ----------------------------------------------------
    params_school = {
        'category_group_code': 'SC4',
        'x': lng, 'y': lat,
        'radius': 1000, 
    }
    res_school = requests.get(url, headers=headers, params=params_school)
    school_count = 0
    if res_school.status_code == 200:
        school_count = res_school.json()['meta']['total_count']
        
    # 데이터 저장
    poi_data.append({
        '대여소_ID': station_id,
        '지하철역_수_1km': subway_count,
        '학교_수_1km': school_count
    })
    
    # 카카오 서버 과부하(차단)를 막기 위해 아주 짧게 휴식
    time.sleep(0.05) 

# 3. 수집된 POI 데이터를 데이터프레임으로 변환
poi_df = pd.DataFrame(poi_data)
print("\nAPI 수집 완료! 데이터 샘플:")
print(poi_df.head())

# ---------------------------------------------------------
# 4. 기존 머신러닝 데이터와 최종 병합 (수정된 부분)
# ---------------------------------------------------------
print("\n기존 머신러닝 데이터와 병합합니다...")

# 1) 이전에 만든 ML 데이터(숫자만 있는 파일) 불러오기
ml_df = pd.read_csv('ml_ready_bike_data_basic.csv') 

# 2) 문자열 ID와 숫자 ID가 모두 들어있는 원본(병합 전) 데이터 불러오기
# (파일명은 예전에 만드셨던 원본 파일명으로 맞춰주세요)
original_df = pd.read_csv('preprocessed_1year_merged_final.csv')

# 3) 원본 데이터에서 [문자열 ID, 숫자 ID] 두 개만 추출해서 짝꿍 사전(매핑 테이블) 만들기
# 원본 데이터에 카테고리화 코드를 다시 한 번 적용하여 번호를 알아냅니다.
original_df['대여소_ID_num'] = original_df['대여소_ID'].astype('category').cat.codes
mapping_table = original_df[['대여소_ID', '대여소_ID_num']].drop_duplicates()

# 4) 카카오 API로 수집한 데이터(poi_df)에 '대여소_ID_num' 컬럼 붙여주기
poi_df_mapped = pd.merge(poi_df, mapping_table, on='대여소_ID', how='left')

# 5) 드디어! 머신러닝 데이터(ml_df)와 POI 데이터(poi_df_mapped)를 숫자 ID 기준으로 결합!
final_ml_df = pd.merge(ml_df, poi_df_mapped, on='대여소_ID_num', how='left')

# 불필요해진 문자열 '대여소_ID' 컬럼은 모델에 안 쓸 거니까 삭제
final_ml_df.drop('대여소_ID', axis=1, inplace=True)

# 저장
final_ml_df.to_csv('ml_ready_bike_data_with_api_poi.csv', index=False, encoding='utf-8-sig')
print("🎉 카카오 API 기반 공간 변수 추가 및 최종 저장 완료!")