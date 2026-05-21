import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로드
print("데이터 로딩 중...")
df = pd.read_csv('ml_ready_bike_data_advanced.csv')

# 2. 독립변수(X)와 종속변수(y) 분리
features = [
    '대여소_ID_num', '월', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', 
    'Lag_1h', 'Lag_24h', 'Lag_168h', 'Rolling_Mean_3h'
]

X = df[features]
y = df['총_대여건수(Y)']

# 3. [핵심] 시계열 데이터 분할 (랜덤 셔플 절대 금지!)
# 데이터를 섞지 않고, 앞의 80% 기간을 Train, 뒤의 20% 기간을 Test로 나눕니다.
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"학습 데이터(Train) 크기: {len(X_train)}건")
print(f"테스트 데이터(Test) 크기: {len(X_test)}건")

# 4. 선형 회귀 모델 학습
print("\n모델 학습을 시작합니다...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 테스트 데이터로 예측
predictions = model.predict(X_test)

# 6. 성능 평가
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n--- 선형 회귀 베이스라인 모델 성능 ---")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f} (예측값이 실제와 평균적으로 {rmse:.2f}대 정도 차이남)")
print(f"R²   : {r2:.4f} (1에 가까울수록 성능이 좋음)")