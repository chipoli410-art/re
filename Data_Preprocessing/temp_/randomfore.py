import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib # 모델과 사전을 저장하기 위한 라이브러리

# 1. 데이터 로드 (Lag 변수가 없는 기본 데이터 사용)
print("데이터 로딩 중...")
df = pd.read_csv('ml_ready_bike_data_basic.csv')

# 2. [핵심 1] 시계열 분할 (랜덤 섞기 금지!)
# 앞의 80% 기간(예: 1~10월)은 학습용(Train), 뒤의 20%(예: 11~12월)는 평가용(Test)
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(f"학습 데이터: {len(train_df)}건 / 평가 데이터: {len(test_df)}건")

# ---------------------------------------------------------
# 3. [핵심 2] 2025년 데이터 기반 패턴 프로파일링 (사전 만들기)
# ---------------------------------------------------------
print("\n과거 평균 패턴(프로파일링)을 생성하고 매핑합니다...")

# 학습 데이터(Train)에서만 평균을 구해야 미래(Test) 정보를 훔쳐보지 않음
# 대여소별 + 요일별 + 시간별 평균 대여 건수 계산
profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

# ---------------------------------------------------------
# 4. Train과 Test 데이터에 프로파일링 변수 추가 (Merge)
# ---------------------------------------------------------
# VLOOKUP 처럼 조건에 맞는 평균값을 쏙쏙 집어넣음
train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

# (안전장치) 만약 Test 기간에 처음 등장하는 조건이 있다면? -> 전체 평균값으로 채움
fallback_mean = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'].fillna(fallback_mean, inplace=True)
test_df['과거_평균_대여량'].fillna(fallback_mean, inplace=True)

# 5. 독립변수(X)와 정답(y) 분리
features = [
    '대여소_ID_num', '월', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', 
    '과거_평균_대여량' # <- 방금 만든 강력한 치트키 변수!
]

X_train = train_df[features]
y_train = train_df['총_대여건수(Y)']
X_test = test_df[features]
y_test = test_df['총_대여건수(Y)']

# 6. 모델 학습 (LinearRegression 대신 더 강력한 RandomForest 사용)
print("\n랜덤 포레스트(RandomForest) 모델 학습을 시작합니다... (시간이 다소 소요될 수 있습니다)")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 7. 예측 및 성능 평가
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n--- 프로파일링 기반 모델 최종 성능 ---")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f} (대여 건수 오차)")
print(f"R²   : {r2:.4f}")

# 8. (실제 서비스 배포용) 만든 '사전'과 '학습된 모델'을 파일로 저장
profile_df.to_csv('station_profile_dict.csv', index=False, encoding='utf-8-sig')
joblib.dump(model, 'bike_demand_model.pkl')
print("\n🎉 'station_profile_dict.csv' (패턴 사전)과 'bike_demand_model.pkl' (모델) 저장 완료!")