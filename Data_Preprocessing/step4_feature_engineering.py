import pandas as pd

print("1. [STEP 4] Step 3 종합 데이터를 불러옵니다...")
df = pd.read_csv('step3_bike_weather_poi.csv')

print("2. 달력 및 기상 파생 변수를 생성합니다...")
# 날짜 변수 생성
df['날짜_dt'] = pd.to_datetime(df['대여일자'].astype(str), format='%Y%m%d')

df['월'] = df['날짜_dt'].dt.month
df['주차'] = df['날짜_dt'].dt.isocalendar().week.astype(int)

# 🌟 [문제 해결] 1~5주차 패턴 추출 변수 추가!
df['월별_주차'] = ((df['날짜_dt'].dt.day - 1) // 7) + 1

df['요일'] = df['날짜_dt'].dt.dayofweek
df['주말_여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)
df['비옴_여부'] = df['강수량'].apply(lambda x: 1 if x > 0 else 0)

print("3. 대여소 문자열 ID를 숫자로 인코딩 중...")
df['대여소_ID_num'] = df['대여소_ID'].astype('category').cat.codes

df = df.dropna().reset_index(drop=True)

print("4. 머신러닝 투입용 핵심 변수만 추출하여 저장합니다...")
features_to_keep = [
    '대여일자', 
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', 
    '지하철역_수_1km', '학교_수_1km', 
    '총_대여건수(Y)' 
]
final_ml_df = df[features_to_keep]

output_file = 'step4_final_ml_ready.csv'
final_ml_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✔️ [STEP 4 완료] '{output_file}' 생성 완료! 🚀")
print("-" * 50)
print("👀 [검증] 최종 데이터 컬럼 목록 확인:")
print(final_ml_df.columns.tolist())
print("-" * 50)