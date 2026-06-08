# step12_baseline_generalization_compare.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings

warnings.filterwarnings('ignore')

print("1. 24년(학습) 및 25년(테스트) 데이터를 로드합니다...")
train_df = pd.read_csv('step4_final_ml_ready_test.csv') 
test_df = pd.read_csv('step4_final_ml_ready.csv')

print("2. 과거 패턴(과거 평균 대여량) 매핑 중 (오직 24년 기준)...")
profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

fallback = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback)
test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback)

# ==========================================
# 🌟 두 가지 피처(변수) 세트 준비
# ==========================================
# [A] 전체 변수 포함 (달력 변수 O - 과적합 위험군)
all_features = [
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

# [B] 달력 변수 제거 방어형 (달력 변수 X - 기상/시간/장소의 본질만)
robust_features = [
    '대여소_ID_num', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

y_train = train_df['총_대여건수(Y)']
y_test = test_df['총_대여건수(Y)']

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, a_min=0, a_max=None) # 음수 방지
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    elapsed = time.time() - start_time
    
    return rmse, r2, elapsed

# ---------------------------------------------------------
# 3. 모델 학습 및 평가 진행
# ---------------------------------------------------------
print("\n3. 모델 학습 및 평가를 시작합니다...")

# (1) 선형 회귀 - 전체 변수
lr_all_rmse, lr_all_r2, t1 = train_and_evaluate(
    LinearRegression(), train_df[all_features], y_train, test_df[all_features], y_test, "LR (All)"
)
print(" ✔️ 선형 회귀 (전체 변수) 완료")

# (2) 선형 회귀 - 방어형 변수
lr_rob_rmse, lr_rob_r2, t2 = train_and_evaluate(
    LinearRegression(), train_df[robust_features], y_train, test_df[robust_features], y_test, "LR (Robust)"
)
print(" ✔️ 선형 회귀 (방어형 변수) 완료")

# (3) 랜덤 포레스트 - 전체 변수
# 파라미터는 메모리 방어를 위해 깊이 15, 트리 50개 고정
rf_all = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf_all_rmse, rf_all_r2, t3 = train_and_evaluate(
    rf_all, train_df[all_features], y_train, test_df[all_features], y_test, "RF (All)"
)
print(" ✔️ 랜덤 포레스트 (전체 변수) 완료")

# (4) 랜덤 포레스트 - 방어형 변수
rf_rob = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf_rob_rmse, rf_rob_r2, t4 = train_and_evaluate(
    rf_rob, train_df[robust_features], y_train, test_df[robust_features], y_test, "RF (Robust)"
)
print(" ✔️ 랜덤 포레스트 (방어형 변수) 완료")

# ---------------------------------------------------------
# 4. 최종 결과 출력
# ---------------------------------------------------------
print("\n" + "="*60)
print(" 🏆 [시계열 미래 예측] 모든 변수 vs 달력 변수 제거 🏆")
print("="*60)

print("📌 [1] 선형 회귀 (Linear Regression)")
print(f"  ▶ 전체 변수 사용   | RMSE: {lr_all_rmse:.4f} 대 | R²: {lr_all_r2:.4f} | 소요시간: {t1:.1f}초")
print(f"  ▶ 달력 변수 제거   | RMSE: {lr_rob_rmse:.4f} 대 | R²: {lr_rob_r2:.4f} | 소요시간: {t2:.1f}초")

print("\n📌 [2] 랜덤 포레스트 (Random Forest)")
print(f"  ▶ 전체 변수 사용   | RMSE: {rf_all_rmse:.4f} 대 | R²: {rf_all_r2:.4f} | 소요시간: {t3:.1f}초")
print(f"  ▶ 달력 변수 제거   | RMSE: {rf_rob_rmse:.4f} 대 | R²: {rf_rob_r2:.4f} | 소요시간: {t4:.1f}초")
print("="*60)