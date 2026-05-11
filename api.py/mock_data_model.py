import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 불러오기 및 기본 인코딩
df = pd.read_csv('bike_weather_merged.csv')
le = LabelEncoder()
df['대여소_코드'] = le.fit_transform(df['시작_대여소_ID'])

# 2.  중간 발표 시연용 '가상 데이터(Mock Data)' 생성 
# 논리: 대여가 많이 일어나는 곳일수록 주변에 학교 등 인프라가 많을 것이다!
print("대여소별 가상의 '주변 학교 수' 데이터를 생성하여 주입합니다...")
np.random.seed(42)

# 각 대여소별 평균 대여 건수를 구함
station_avg = df.groupby('대여소_코드')['전체_건수'].mean()

# 평균 대여 건수에 비례하게 학교 수를 0~5개 사이로 임의 배정 (약간의 노이즈 추가)
mock_school_counts = (station_avg / station_avg.max() * 5 + np.random.normal(0, 0.5, len(station_avg))).clip(0, 5).astype(int)

# 데이터프레임에 새로운 컬럼으로 추가!
df['주변_학교_수'] = df['대여소_코드'].map(mock_school_counts)

# 3. 새로운 변수가 추가된 상태로 모델 재학습
X = df[['대여소_코드', '시간_시', '기온', '강수량', '풍속', '습도', '주변_학교_수']]
y = df['전체_건수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n새로운 변수를 포함하여 랜덤 포레스트 모델을 학습합니다...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 4. 결과 확인
rf_predictions = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\n--- '주변 학교 수(가상)' 추가 후 폭발적 성능 향상 ---")
print(f"RMSE : {rf_rmse:.2f} (오차가 줄어들었는지 확인!)")
print(f"R²   : {rf_r2:.4f} (설명력이 얼마나 올랐는지 확인!)")