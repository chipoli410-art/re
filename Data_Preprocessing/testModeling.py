import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

print("1. 24년, 25년 클린 데이터를 로드하여 시계열을 병합합니다...")
train_full = pd.read_csv('step4_final_ml_ready.csv') 
test_full = pd.read_csv('step4_final_ml_ready_test.csv')

df_all = pd.concat([train_full, test_full], ignore_index=True)

# ==========================================
# 💡 [핵심] 계절성(작년 여름) + 최신성(직전 9개월) 동시 반영!
# ==========================================
print("\n2. 학습(최근 1년)과 평가(25년 여름) 기간을 분할합니다...")

# Train: 2024년 6월 1일 ~ 2025년 5월 31일 (정확히 과거 1년 치)
train_df = df_all[(df_all['대여일자'] >= 20240601) & (df_all['대여일자'] <= 20250531)].copy()

# Test: 2025년 6월 1일 ~ 2025년 8월 31일 (타겟 여름 3개월)
test_df = df_all[(df_all['대여일자'] >= 20250601) & (df_all['대여일자'] <= 20250831)].copy()

print(f" - Train 기간: {train_df['대여일자'].min()} ~ {train_df['대여일자'].max()}")
print(f" - Test 기간: {test_df['대여일자'].min()} ~ {test_df['대여일자'].max()}")

print("\n3. 과거 패턴 매핑 (Data Leakage 완벽 차단)...")
profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

fallback = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback)
test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback)

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

print("\n4. 단일 회귀 모델(LightGBM) 학습 중...")
best_params = {
    'objective': 'regression', 'metric': 'rmse', 'random_state': 42,
    'n_estimators': 1500, 'n_jobs': -1, 'learning_rate': 0.05, 
    'num_leaves': 63, 'max_depth': 10
}

model = lgb.LGBMRegressor(**best_params)
model.fit(
    X_train, y_train, 
    eval_set=[(X_test, y_test)], 
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

print("\n5. 2025년 여름 타겟 예측 및 평가...")
final_preds = model.predict(X_test)
final_preds = np.clip(final_preds, a_min=0, a_max=None)

final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_r2 = r2_score(y_test, final_preds)

print("\n" + "="*50)
print(" 👑 [궁극의 가설 검증] 1Year Rolling (24.06~25.05 ➔ 25년 여름) 👑")
print("="*50)
print(f" - 예측 RMSE : {final_rmse:.4f} 대")
print(f" - 예측 R²   : {final_r2:.4f}")
print("="*50)