import requests
import pandas as pd
import json
import time

# 1. 전처리된 따릉이 데이터 로드
print("1. 전처리된 따릉이 데이터를 불러옵니다...")
bike_df = pd.read_csv('preprocessed_1year_bike_demand.csv')

# 데이터에서 시작일과 종료일 자동 추출 (API 호출용)
start_date = str(bike_df['대여일자'].min())
end_date = str(bike_df['대여일자'].max())

print(f"   - 데이터 기간: {start_date} ~ {end_date}")

# 2. 기상청 날씨 데이터 수집 (페이지 반복 호출)
API_KEY = '6cdfb8721d44bfefa545d6a41a7e6f56c02b78423aa8dcf8fd54c80d072fe552'
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'

print("\n2. 기상청 API에서 전체 기간의 날씨 데이터를 수집합니다...")
weather_list = []
page_no = 1
total_pages = 1 # 초기값

while True:
    params = {
        'serviceKey': API_KEY,
        'pageNo': str(page_no),
        'numOfRows': '999', # API 1회 최대 호출 한도
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'HR',
        'stnIds': '108',    # 서울
        'startDt': start_date,
        'startHh': '00',
        'endDt': end_date,
        'endHh': '23'
    }

    try:
        response = requests.get(url, params=params)
        data = json.loads(response.text)
        
        # 정상 응답 확인
        if data['response']['header']['resultCode'] != '00':
            print(f"API 에러 발생: {data['response']['header']['resultMsg']}")
            break

        items = data['response']['body']['items']['item']
        total_count = int(data['response']['body']['totalCount'])
        
        # 날씨 데이터 파싱하여 리스트에 추가
        for item in items:
            # 기상청 날짜 포맷(2026-05-14 14:00)을 따릉이 데이터와 동일하게 변환
            # 대여일자: 20260514 (정수형) / 대여시간: 14 (정수형)
            weather_list.append({
                '대여일자': int(item['tm'][:10].replace('-', '')),
                '대여시간(시)': int(item['tm'][11:13]), 
                '기온': float(item['ta']) if item['ta'] else 0.0,
                '강수량': float(item['rn']) if item['rn'] else 0.0,
                '풍속': float(item['ws']) if item['ws'] else 0.0,
                '습도': float(item['hm']) if item['hm'] else 0.0
            })
        
        print(f"   - {page_no}페이지 수집 완료... (현재까지 {len(weather_list)}시간 데이터 확보)")
        
        # 종료 조건 파악 (지금까지 모은 데이터가 전체 데이터 개수와 같거나 크면 종료)
        if len(weather_list) >= total_count:
            break
            
        page_no += 1
        time.sleep(0.5) # API 서버 과부하 방지를 위해 0.5초 대기

    except Exception as e:
        print(f"호출 중 오류 발생 (Page {page_no}): {e}")
        break

# 수집된 날씨 리스트를 데이터프레임으로 변환
weather_df = pd.DataFrame(weather_list)
print("\n날씨 데이터 수집 완료! 총 시간:", len(weather_df))

# 3. 데이터 병합 (Merge)
print("\n3. 따릉이 데이터와 날씨 데이터를 병합합니다...")

# [핵심] 일자와 시간을 동시에 기준으로 삼아 병합 (Left Join)
merged_df = pd.merge(bike_df, weather_df, on=['대여일자', '대여시간(시)'], how='left')

# 강수량 등 날씨 정보가 없는(기상청 누락) 결측치를 0으로 채우기
merged_df.fillna(0, inplace=True)

# 4. 최종 파일 저장
output_filename = 'preprocessed_1year_merged_final.csv'
merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n최종 병합 완료! '{output_filename}' 파일이 생성되었습니다. 🎉")