# step8_ultimate_stacking.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings

warnings.filterwarnings('ignore')

print("1. 데이터를 준비하고 과거 패턴을 결합합니다...")
df = pd.read_csv('step4_final_ml_ready.csv') 
df = df.sort_values(by=['대여일자', '대여시간(시)']).reset_index(drop=True)
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

fallback = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback)
test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback)

features = [
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', 
    '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

X_train, y_train = train_df[features], train_df['총_대여건수(Y)']
X_test, y_test = test_df[features], test_df['총_대여건수(Y)']

# ---------------------------------------------------------
# 🌟 [비기 1] 타겟 변수 로그 변환 (Log Transformation)
# 대여량에 log(x+1)을 씌워 비대칭성을 해결합니다.
# ---------------------------------------------------------
print("2. [마법] 타겟 변수(대여량)에 로그(Log) 변환을 적용합니다...")
y_train_log = np.log1p(y_train)

# ---------------------------------------------------------
# 🌟 [비기 2] 스태킹 앙상블 모델 구축 (Stacking Ensemble)
# ---------------------------------------------------------
print("3. [결성] 3대 부스팅 모델 어벤져스를 결성합니다... (시간이 꽤 소요됩니다!)")
stacking_start_time = time.time()

# 3-1. 개별 모델(Base Models) 정의
# (스태킹은 내부적으로 5번씩 반복 학습하므로, 무거워지지 않게 튜닝값을 살짝 다이어트합니다)
estimators = [
    ('lgb', lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1)),
    ('xgb', xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
]

# 3-2. 메타 모델(Meta Model) 정의
# 3개의 천재들이 낸 답안지를 취합해서 최종 결론을 내릴 리더(선형 회귀의 일종인 Ridge)입니다.
meta_model = Ridge()

# 3-3. 스태킹 앙상블 조립
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,       # 5겹 교차검증으로 과적합 완벽 방지
    n_jobs=1    # 메모리 폭발 방지를 위해 순차적 진행
)

print("\n🚀 궁극의 스태킹 모델 학습 시작... (컴퓨터 사양에 따라 5분 ~ 15분 소요)")
stacking_model.fit(X_train, y_train_log)

# ---------------------------------------------------------
# 4. 예측 및 역 로그 변환
# ---------------------------------------------------------
print("\n4. 테스트 데이터를 예측하고 원래 단위(자전거 대수)로 복원합니다...")
# 로그가 씌워진 상태로 예측값이 나옵니다.
preds_log = stacking_model.predict(X_test)
# 역 로그(exp(x)-1)를 씌워 실제 대여 대수로 되돌립니다!
final_preds = np.expm1(preds_log)

# 혹시 모를 음수 예측 방지 (대여량은 0보다 작을 수 없음)
final_preds = np.clip(final_preds, a_min=0, a_max=None)

# ---------------------------------------------------------
# 5. 최종 결과 확인
# ---------------------------------------------------------
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_r2 = r2_score(y_test, final_preds)
stacking_time = time.time() - stacking_start_time

print("\n" + "="*50)
print(" 👑 궁극의 스태킹 앙상블 & 로그 변환 최종 결과 👑")
print("="*50)
print(f"[이전 최고 기록] 단일 LightGBM 극한 튜닝: R² 0.6352")
print("-" * 50)
print(f"[최종 합체 모델] Stacking Ensemble (LGBM + XGB + RF)")
print(f"  - RMSE: {final_rmse:.2f} 대")
print(f"  - R²  : {final_r2:.4f}")
print(f"  - 소요 시간: {stacking_time/60:.1f} 분 🐢 (정확도를 위해 시간을 바쳤습니다)")
print("="*50)