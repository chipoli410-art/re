# step5_modeling_comparison.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings

# 불필요한 경고 메시지 숨김
warnings.filterwarnings('ignore')

# 1. Step 4 전처리 완료된 마스터 데이터 로드
print("1. Step 4 최종 마스터 데이터를 불러오는 중...")
# 전처리 파일명이 다를 경우 실제 파일명으로 수정해 주세요.
df = pd.read_csv('step4_final_ml_ready_25.csv') 

# 2. 데이터 분할 (시간순 80:20 분할)
print("2. 데이터를 학습용(Train)과 평가용(Test)으로 분할합니다...")
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# ---------------------------------------------------------
# 3. 과거 평균 대여량(프로파일링) 구축 (⭐데이터 누수 방지 필수)
# ---------------------------------------------------------
print("3. Train 데이터를 기반으로 과거 대여 패턴(프로파일링) 사전을 구축합니다...")
# 미래 데이터가 섞이지 않도록 오직 Train 데이터로만 평균을 계산합니다.
profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

# 학습 데이터와 테스트 데이터에 각각 과거 패턴 매핑
train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

# 처음 등장하는 새로운 패턴(결측치)은 전체 평균으로 안전하게 채우기
fallback_mean = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback_mean)
test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback_mean)

# ---------------------------------------------------------
# 4. 독립변수(X)와 종속변수(Y) 설정
# ---------------------------------------------------------
# Step 4에서 추가된 '주차'와 '월별_주차' 변수를 모두 포함합니다.
features = [
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', 
    '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

X_train, y_train = train_df[features], train_df['총_대여건수(Y)']
X_test, y_test = test_df[features], test_df['총_대여건수(Y)']

print(f" - 모델 예측에 사용되는 변수 개수: {len(features)}개")
print(f" - 학습 데이터 수: {len(X_train):,}건 / 테스트 데이터 수: {len(X_test):,}건\n")

# ---------------------------------------------------------
# 5. [모델 1] 선형 회귀 (Linear Regression)
# ---------------------------------------------------------
print("🤖 [모델 1] 선형 회귀 모델 학습 및 평가 시작...")
lr_start_time = time.time()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)
print(f" ✔️ 선형 회귀 완료 (소요시간: {time.time() - lr_start_time:.1f}초)")

# ---------------------------------------------------------
# 6. [모델 2] 랜덤 포레스트 (Random Forest)
# ---------------------------------------------------------
print("\n🤖 [모델 2] 랜덤 포레스트 모델 학습 시작... (가용 메모리 최적화 설정)")
rf_start_time = time.time()

# 제로 패딩으로 행이 늘어났으므로 메모리 방어를 위해 depth=15, n_estimators=50으로 안정화
rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)
print(f" ✔️ 랜덤 포레스트 완료 (소요시간: {time.time() - rf_start_time:.1f}초)")

# ---------------------------------------------------------
# 7. 최종 성능 비교 및 인사이트 출력
# ---------------------------------------------------------
print("\n" + "="*50)
print(" 🏆 따릉이 수요 예측 모델 최종 성능 비교 결과 🏆")
print("="*50)
print(f"[1] 선형 회귀 (Baseline)")
print(f"  - RMSE (평균 오차): {lr_rmse:.2f} 대")
print(f"  - R²   (설명력)  : {lr_r2:.4f}")
print("-" * 50)
print(f"[2] 랜덤 포레스트 (Advanced)")
print(f"  - RMSE (평균 오차): {rf_rmse:.2f} 대")
print(f"  - R²   (설명력)  : {rf_r2:.4f}")
print("="*50)

# 두 주차 변수의 중요도 분석 결과 출력
print("\n💡 [인사이트 분석] 어떤 주차 변수가 더 효과적이었을까?")
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).set_index('Feature')
print(f" - 1~52주차 (거시적 계절성 기후 흐름) 중요도 : {feat_imp.loc['주차', 'Importance']:.4f}")
print(f" - 월별 주차 (미시적 인간 생활 패턴) 중요도 : {feat_imp.loc['월별_주차', 'Importance']:.4f}")