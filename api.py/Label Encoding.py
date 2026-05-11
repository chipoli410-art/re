import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('bike_weather_merged.csv')

# 1. 텍스트로 된 대여소 ID를 기계가 이해할 수 있는 숫자로 변환 (Label Encoding)
print("대여소 위치 데이터를 변환 중입니다...")
le = LabelEncoder()
df['대여소_코드'] = le.fit_transform(df['시작_대여소_ID'])

# 2. X값에 대여소_코드 추가
X = df[['대여소_코드', '시간_시', '기온', '강수량', '풍속', '습도']]
y = df['전체_건수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 랜덤 포레스트 모델 재학습
print("랜덤 포레스트 모델을 다시 학습합니다 (시간이 조금 걸릴 수 있습니다)...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 4. 예측 및 평가
rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

print("\n--- 위치 데이터가 추가된 랜덤 포레스트 성능 ---")
print(f"MSE  : {rf_mse:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.4f}")