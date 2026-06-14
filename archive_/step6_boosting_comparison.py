# step6_boosting_comparison.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings

# 불필요한 경고 메시지 숨김
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 데이터 로드 및 분할
# ---------------------------------------------------------
print("1. 데이터를 불러오고 분할합니다...")
df = pd.read_csv('step4_final_ml_ready_25.csv') 

# 시간순 분할 (미래 데이터 누수 방지)
df = df.sort_values(by=['대여일자', '대여시간(시)']).reset_index(drop=True)
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# ---------------------------------------------------------
# 2. 과거 평균 패턴(프로파일링) 결합
# ---------------------------------------------------------
print("2. Train 데이터 기반 과거 대여 패턴을 학습하고 결합합니다...")
profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

fallback_mean = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback_mean)
test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback_mean)

# 독립/종속 변수 분리
features = [
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', 
    '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

X_train, y_train = train_df[features], train_df['총_대여건수(Y)']
X_test, y_test = test_df[features], test_df['총_대여건수(Y)']

# ---------------------------------------------------------
# 3. [모델 1] LightGBM (속도의 제왕)
# ---------------------------------------------------------
print("\n🚀 [모델 1] LightGBM 학습 시작...")
lgb_start_time = time.time()

# 트리를 300개나 만들지만 속도는 랜덤포레스트(50개)보다 빠릅니다.
lgb_model = lgb.LGBMRegressor(
    n_estimators=300, 
    learning_rate=0.1, 
    random_state=42, 
    n_jobs=-1
)
lgb_model.fit(X_train, y_train)
lgb_predictions = lgb_model.predict(X_test)

lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_predictions))
lgb_r2 = r2_score(y_test, lgb_predictions)
lgb_time = time.time() - lgb_start_time
print(f" ✔️ LightGBM 완료 (소요시간: {lgb_time:.1f}초)")

# ---------------------------------------------------------
# 4. [모델 2] XGBoost (정교함의 제왕)
# ---------------------------------------------------------
print("\n🤖 [모델 2] XGBoost 학습 시작... ")
xgb_start_time = time.time()

# XGBoost는 트리를 깊게 팔수록 메모리를 많이 쓰므로 max_depth=7 정도로 제한
xgb_model = xgb.XGBRegressor(
    n_estimators=300, 
    learning_rate=0.1, 
    max_depth=7,
    random_state=42, 
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)
xgb_time = time.time() - xgb_start_time
print(f" ✔️ XGBoost 완료 (소요시간: {xgb_time:.1f}초)")

# ---------------------------------------------------------
# 5. 최종 결과 대결판 출력
# ---------------------------------------------------------
print("\n" + "="*50)
print(" 🏆 따릉이 수요 예측 부스팅(Boosting) 모델 비교")
print("="*50)
print(f"[기존 기록] 랜덤 포레스트 : R² 0.6304 / 447.4초 소요")
print("-" * 50)
print(f"[선수 1] LightGBM")
print(f"  - RMSE: {lgb_rmse:.2f} 대")
print(f"  - R²  : {lgb_r2:.4f}")
print(f"  - 속도: {lgb_time:.1f}초 ⚡ (랜덤 포레스트 대비 약 {447.4 / lgb_time:.1f}배 빠름!)")
print("-" * 50)
print(f"[선수 2] XGBoost")
print(f"  - RMSE: {xgb_rmse:.2f} 대")
print(f"  - R²  : {xgb_r2:.4f}")
print(f"  - 속도: {xgb_time:.1f}초 🛡️")
print("="*50)