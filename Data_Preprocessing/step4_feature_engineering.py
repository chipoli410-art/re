import pandas as pd

print("1. 2024년과 2025년의 Step 3(날씨/공간 병합 완료) 데이터를 불러옵니다...")
# 🚨 파일명은 질문자님이 24년, 25년 데이터를 가공하며 저장하셨던 Step3 파일명으로 변경해 주세요!
df_25 = pd.read_csv('step3_bike_weather_poi_25.csv')
df_26 = pd.read_csv('step3_bike_weather_poi_26.csv')

print("2. 인코딩 기준을 통일하기 위해 두 데이터를 임시로 병합합니다...")
df_25['데이터_연도'] = 2025
df_26['데이터_연도'] = 2026
df_combined = pd.concat([df_25, df_26], ignore_index=True)

print("3. 파생 변수를 생성합니다...")
df_combined['날짜_dt'] = pd.to_datetime(df_combined['대여일자'].astype(str), format='%Y%m%d')
df_combined['월'] = df_combined['날짜_dt'].dt.month
df_combined['주차'] = df_combined['날짜_dt'].dt.isocalendar().week.astype(int)
df_combined['월별_주차'] = ((df_combined['날짜_dt'].dt.day - 1) // 7) + 1
df_combined['요일'] = df_combined['날짜_dt'].dt.dayofweek
df_combined['주말_여부'] = df_combined['요일'].apply(lambda x: 1 if x >= 5 else 0)
df_combined['비옴_여부'] = df_combined['강수량'].apply(lambda x: 1 if x > 0 else 0)

print("4. 🌟[핵심 수정] 24-25년 통합 기준으로 대여소 ID를 고유 번호로 인코딩합니다!")
# 이렇게 해야 24년의 강남역(10번)이 25년에도 똑같이 10번을 유지합니다.
df_combined['대여소_ID_num'] = df_combined['대여소_ID'].astype('category').cat.codes

# 결측치 제거
df_combined = df_combined.dropna(subset=['기온', '지하철역_수_1km', '학교_수_1km']).reset_index(drop=True)

print("5. 다시 2024년과 2025년 데이터로 완벽하게 분리합니다...")
features_to_keep = [
    '대여일자', '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '지하철역_수_1km', '학교_수_1km', '총_대여건수(Y)'
]

final_25 = df_combined[df_combined['데이터_연도'] == 2025][features_to_keep]
final_26 = df_combined[df_combined['데이터_연도'] == 2026][features_to_keep]

print("6. 최종 머신러닝 준비 파일을 저장합니다...")
final_25.to_csv('step4_final_ml_ready_25.csv', index=False, encoding='utf-8-sig')
final_26.to_csv('step4_final_ml_ready_26.csv', index=False, encoding='utf-8-sig')

print("\n✅ 완료! 기준이 완벽하게 통일된 24년, 25년 ML 데이터가 생성되었습니다.")