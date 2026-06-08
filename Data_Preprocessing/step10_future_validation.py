import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 시계열 순서에 맞는 데이터 로드
# ==========================================
print("1. 과거(2024년) 학습 데이터와 미래(2025년) 테스트 데이터를 불러옵니다...")
# 파일명은 실제 저장하신 이름으로 변경해 주세요!
train_df = pd.read_csv('step4_final_ml_ready.csv') 
test_df = pd.read_csv('step4_final_ml_ready_test.csv')

# ==========================================
# 2. 미래 데이터 누수(Data Leakage) 완벽 차단 전처리
# ==========================================
print("2. [핵심] 오직 2024년 데이터만 사용하여 '과거 평균 대여량' 사전을 구축합니다...")
profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

# 24년(학습)과 25년(평가) 데이터에 각각 매핑
train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

# 25년에 새로 생겼거나 패턴이 없는 대여소는 24년 전체 평균으로 안전하게 대체
fallback = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback)
test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback)

features = [
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

# LightGBM용 카테고리 변환
for col in train_df[features].select_dtypes(include=['object']).columns:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

X_train, y_train = train_df[features], train_df['총_대여건수(Y)']
X_test, y_test = test_df[features], test_df['총_대여건수(Y)']

# ==========================================
# 3. Two-Stage 모델 학습 및 예측 (2024 ➔ 2025)
# ==========================================
print("\n3. [Stage 1] 수요 발생 여부(0 or 1) 분류 모델 학습 중...")
y_train_bin = (y_train > 0).astype(int) 

clf_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05,
    'num_leaves': 63, 'random_state': 42, 'n_estimators': 500, 'n_jobs': -1, 'is_unbalance': True 
}
classifier = lgb.LGBMClassifier(**clf_params)
classifier.fit(X_train, y_train_bin)

prob_preds = classifier.predict_proba(X_test)[:, 1] # 2025년 대여 발생 확률

print("4. [Stage 2] 실제 대여량 회귀 모델 학습 중...")
mask_train = y_train > 0
X_train_pos, y_train_pos = X_train[mask_train], y_train[mask_train]
y_train_pos_log = np.log1p(y_train_pos)

reg_params = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
    'num_leaves': 63, 'random_state': 42, 'n_estimators': 800, 'n_jobs': -1
}
regressor = lgb.LGBMRegressor(**reg_params)
regressor.fit(X_train_pos, y_train_pos_log)

reg_preds_log = regressor.predict(X_test)
reg_preds = np.expm1(reg_preds_log) # 2025년 실제 대여량 예측

# ==========================================
# 4. 최종 결과 출력
# ==========================================
print("\n5. 2025년 미래 데이터를 예측하고 최종 성능을 계산합니다...")
final_preds = prob_preds * reg_preds
final_preds = np.clip(final_preds, a_min=0, a_max=None)

final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_r2 = r2_score(y_test, final_preds)

print("\n" + "="*50)
print(" 🚀 [시계열 일반화 검증(미래 예측) 최종 성능] 🚀")
print("="*50)
print(f" - Train: 2024년 데이터 ➔ Test: 2025년 데이터")
print(f" - 미래 예측 RMSE : {final_rmse:.4f} 대")
print(f" - 미래 예측 R²   : {final_r2:.4f}")
print("="*50)