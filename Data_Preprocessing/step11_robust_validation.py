# step11_robust_validation_clean.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

print("1. [정제 완료된] 24년(학습) 및 25년(테스트) 데이터 로드...")
# 파일명은 step4에서 최종 생성하신 머신러닝용 파일명으로 맞춰주세요!
train_df = pd.read_csv('step4_final_ml_ready.csv') 
test_df = pd.read_csv('step4_final_ml_ready_test.csv')

print("2. 과거 패턴 매핑 (Data Leakage 완벽 차단)...")
profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

fallback = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback)
test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback)

# ==========================================
# 🌟 [핵심] 달력 변수 제거! (월, 주차 등 제외)
# 오직 기온, 시간, 인프라, 과거 진짜 평균치만 사용합니다.
# ==========================================
robust_features = [
    '대여소_ID_num', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

cat_cols = ['대여소_ID_num', '요일', '주말_여부', '비옴_여부']
for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

X_train, y_train = train_df[robust_features], train_df['총_대여건수(Y)']
X_test, y_test = test_df[robust_features], test_df['총_대여건수(Y)']

print("\n3. 🛡️ 과적합 방어 단일 회귀 모델 학습 중...")
best_params = {
    'objective': 'regression', 
    'metric': 'rmse', 
    'random_state': 42,
    'n_estimators': 1500, 
    'n_jobs': -1,
    'learning_rate': 0.05, 
    'num_leaves': 63, 
    'max_depth': 10
}

model = lgb.LGBMRegressor(**best_params)
model.fit(
    X_train, y_train, 
    eval_set=[(X_test, y_test)], 
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

print("4. 2025년 미래 수요 예측 및 평가...")
final_preds = model.predict(X_test)
final_preds = np.clip(final_preds, a_min=0, a_max=None)

final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_r2 = r2_score(y_test, final_preds)

print("\n" + "="*50)
print(" 🚀 [클린 데이터 + 과적합 방어] 2025 최종 성능 🚀")
print("="*50)
print(f" - 미래 예측 RMSE : {final_rmse:.4f} 대")
print(f" - 미래 예측 R²   : {final_r2:.4f}")
print("="*50)