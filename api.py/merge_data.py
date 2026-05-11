import requests
import pandas as pd
import json
#feat(data): 시간별 수요 데이터와 기상청 날씨 데이터 병합 로직 작성
# 1. API 키 설정 
API_KEY = '6cdfb8721d44bfefa545d6a41a7e6f56c02b78423aa8dcf8fd54c80d072fe552'

# 2. 기상청 날씨 데이터 수집
print("기상청 API에서 날씨 데이터를 수집합니다...")
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
params = {
    'serviceKey': API_KEY,
    'pageNo': '1',
    'numOfRows': '24',
    'dataType': 'JSON',
    'dataCd': 'ASOS',
    'dateCd': 'HR',
    'stnIds': '108', # 108은 서울을 의미함
    'endDt': '20260430',
    'endHh': '23',
    'startHh': '00',
    'startDt': '20260430'
}

response = requests.get(url, params=params)

try:
    # JSON 데이터 파싱
    data = json.loads(response.text)
    items = data['response']['body']['items']['item']
    
    weather_list = []
    for item in items:
        weather_list.append({
            '시간_시': int(item['tm'][11:13]), # 시간만 추출 (예: '2026-04-30 01:00' -> 1)
            '기온': float(item['ta']) if item['ta'] else 0.0,
            '강수량': float(item['rn']) if item['rn'] else 0.0,
            '풍속': float(item['ws']) if item['ws'] else 0.0,
            '습도': float(item['hm']) if item['hm'] else 0.0
        })
    
    weather_df = pd.DataFrame(weather_list)
    print("날씨 데이터 수집 완료!\n", weather_df.head())
    
    # 3. 따릉이 집계 데이터와 날씨 데이터 병합
    print("\n따릉이 데이터와 날씨 데이터를 병합합니다...")
    bike_df = pd.read_csv('tpss_hourly_demand.csv')
    
    # 시간_시를 기준으로 병합 (Left Join)
    merged_df = pd.merge(bike_df, weather_df, on='시간_시', how='left')
    
    # 혹시 모를 결측치를 0으로 채우기
    merged_df.fillna(0, inplace=True)
    
    # 최종 파일 저장
    merged_df.to_csv('bike_weather_merged.csv', index=False, encoding='utf-8-sig')
    print("\n최종 데이터 병합 완료! 'bike_weather_merged.csv' 파일이 생성되었습니다.")
    
except Exception as e:
    print("API 호출 중 오류가 발생했습니다. API 키가 등록되는 중이거나 잘못 입력되었을 수 있습니다.")
    print("자세한 에러 메시지:", response.text)