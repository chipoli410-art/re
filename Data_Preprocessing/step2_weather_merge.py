# step2_weather_merge.py
import pandas as pd
import requests
import json
import time

print("1. [STEP 2] Step 1 집계 데이터 로드 중...")
bike_df = pd.read_csv('step1_aggregated_bike_smart_test.csv')

# Step 1 데이터에서 자동으로 시작일과 종료일 추출
start_date, end_date = str(bike_df['대여일자'].min()), str(bike_df['대여일자'].max())

print(f"2. 기상청 API 날씨 데이터 수집 시작 ({start_date} ~ {end_date})...")
# 🚨 [주의] 본인의 실제 기상청 인코딩(Encoding) API 키로 반드시 변경하세요!
API_KEY = '6cdfb8721d44bfefa545d6a41a7e6f56c02b78423aa8dcf8fd54c80d072fe552' 
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'

weather_list = []
page_no = 1
while True:
    params = {
        'serviceKey': API_KEY, 'pageNo': str(page_no), 'numOfRows': '999',
        'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'HR', 'stnIds': '108', # 108: 서울 지점코드
        'startDt': start_date, 'startHh': '00', 'endDt': end_date, 'endHh': '23'
    }
    
    try:
        response = requests.get(url, params=params)
        data = json.loads(response.text)
        
        # 정상 응답(00)이 아니면(예: 데이터 끝 도달) 루프 종료
        if data['response']['header']['resultCode'] != '00': 
            break
            
        items = data['response']['body']['items']['item']
        
        for item in items:
            weather_list.append({
                '대여일자': int(item['tm'][:10].replace('-', '')),
                '대여시간(시)': int(item['tm'][11:13]), # 🌟 옛날 '시간_시'에서 '대여시간(시)'로 변경!
                '기온': float(item['ta']) if item['ta'] else 0.0,
                '강수량': float(item['rn']) if item['rn'] else 0.0,
                '풍속': float(item['ws']) if item['ws'] else 0.0,
                '습도': float(item['hm']) if item['hm'] else 0.0
            })
            
        # 총 데이터 개수만큼 다 가져왔으면 루프 종료
        if len(weather_list) >= int(data['response']['body']['totalCount']): 
            break
            
        page_no += 1
        time.sleep(0.5) # 기상청 서버 과부하 방지
        print(f" - {page_no}페이지 데이터 수집 중...")
        
    except Exception as e:
        print(f"API 호출 중 에러 발생: {e}")
        break

weather_df = pd.DataFrame(weather_list)
print(f" -> 날씨 데이터 총 {len(weather_df)}건 수집 완료!")

print("\n3. 따릉이 데이터와 날씨 데이터 병합 중...")
# 🌟 [핵심 수정] 병합 기준을 '대여일자'와 '대여시간(시)'로 정확히 맞춤
merged_df = pd.merge(bike_df, weather_df, on=['대여일자', '대여시간(시)'], how='left')

# 혹시 날씨 API에 누락된 시간이 있다면 0으로 안전하게 채움
merged_df.fillna(0, inplace=True) 

output_file = 'step2_bike_weather_test.csv'
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n ✔️ [STEP 2 완료] 날씨 병합 데이터 '{output_file}' 생성 완료!")
print("🌟 병합된 데이터 샘플 확인:")
print(merged_df[['대여일자', '대여시간(시)', '대여소명', '기온', '강수량', '총_대여건수(Y)']].head(5))