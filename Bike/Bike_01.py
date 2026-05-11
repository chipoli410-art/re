"""
따릉이(공공자전거) 수요 예측 머신러닝 프로젝트
Ttareungi Bike Rental Demand Prediction ML Project

프로젝트 목표:
- 시간대, 날씨, 요일, 유동인구, 주변 시설 정보를 이용해 자전거 대여량 예측
- 2가지 추가 기능: 유동인구 데이터, 주변 교육/편의시설 정보

프로젝트 구조:
1. 데이터 로드 및 탐색적 분석 (EDA)
2. 특성 공학 (Feature Engineering)
3. 데이터 전처리 (Data Preprocessing)
4. 모델 학습 (Model Training)
5. 모델 평가 (Model Evaluation)
6. 결과 시각화 (Visualization)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1단계: 샘플 데이터셋 생성
# ============================================================================
"""
실제 프로젝트에서는:
- 서울시 열린데이터광장에서 따릉이 이용정보 다운로드
- 기상청 API에서 날씨 데이터 수집
- 서울시 교통정보 API에서 교통체증 데이터 수집
- 유동인구 통계데이터 수집
- 주변 시설 정보 (학교, 카페, 지하철 등) 수집

여기서는 시뮬레이션 데이터로 데모합니다.
"""

print("=" * 80)
print("따릉이 수요 예측 머신러닝 프로젝트")
print("=" * 80)
print()

# 샘플 데이터 생성 (365일, 24시간 = 8760개 샘플)
np.random.seed(42)
n_samples = 8760  # 1년 데이터

# 기본 정보
dates = pd.date_range('2025-01-01', periods=n_samples, freq='h')
hour = dates.hour
day_of_week = dates.dayofweek  # 0=월요일, 6=일요일
month = dates.month

# 기상 데이터
temperature = 15 + 10 * np.sin(np.arange(n_samples) / 365 * 2 * np.pi) + np.random.normal(0, 3, n_samples)
humidity = 60 + 20 * np.sin(np.arange(n_samples) / 365 * 2 * np.pi) + np.random.normal(0, 5, n_samples)
precipitation = np.maximum(0, np.random.exponential(0.5, n_samples) - 0.3)

# 추가 기능 1: 유동인구 데이터 (시간대별로 다른 패턴)
# 출퇴근 시간(8시, 18시)에 높은 유동인구, 야간에 낮은 유동인구
foot_traffic = 5000 + 3000 * (np.sin(hour * np.pi / 12) ** 2) + np.random.normal(0, 500, n_samples)
foot_traffic = np.maximum(foot_traffic, 0)

# 추가 기능 2: 주변 교육시설 유무 (1=있음, 0=없음)
# 대여소마다 다르지만, 여기서는 확률적으로 할당
nearby_school = np.random.binomial(1, 0.4, n_samples)  # 40% 확률로 학교 있음

# 추가 기능 3: 주변 편의시설 개수 (카페, 편의점, 식당 등)
nearby_facilities = np.random.poisson(3, n_samples)  # 평균 3개

# 교통체증 데이터 (0-10 스케일, 0=맑음, 10=심각)
traffic_congestion = 3 + 4 * (np.sin(hour * np.pi / 12) ** 2) + np.random.normal(0, 1, n_samples)
traffic_congestion = np.clip(traffic_congestion, 0, 10)

# 공휴일 더미 변수 (간단히 처리)
is_holiday = np.zeros(n_samples)
is_holiday[np.isin(day_of_week, [5, 6])] = 1  # 토요일, 일요일

# 종속 변수: 자전거 대여량 (우리가 예측하려는 값)
"""
대여량은 다음 요인에 영향을 받음:
- 시간대: 출퇴근 시간에 높음
- 날씨: 기온이 좋고 비가 안 올 때 높음
- 요일: 평일이 주말보다 높음
- 유동인구: 유동인구가 많을수록 높음
- 주변 시설: 주변에 시설이 많을수록 높음
- 교통체증: 교통이 많을수록 높음 (자전거 대체 이동수단)
"""
rental_count = (
    200 +  # 기본값
    150 * (np.sin(hour * np.pi / 12) ** 2) +  # 시간대별 변동 (출퇴근 시간 높음)
    20 * (temperature - 15) * np.clip(20 - temperature, 0, 10) / 5 +  # 기온 효과 (15-20도에서 최적)
    -100 * np.clip(precipitation, 0, 5) +  # 강수 효과 (비 오면 감소)
    -50 * (1 - is_holiday) +  # 평일 효과 (평일이 주말보다 낮음)
    0.02 * foot_traffic +  # 유동인구 효과 (주요 추가 기능)
    50 * nearby_school +  # 학교 근처 효과 (주요 추가 기능)
    30 * nearby_facilities +  # 편의시설 효과
    15 * traffic_congestion +  # 교통체증 효과
    np.random.normal(0, 30, n_samples)  # 노이즈
)

# 음수 값은 0으로 처리
rental_count = np.maximum(rental_count, 0).astype(int)

# 데이터프레임 생성
df = pd.DataFrame({
    'date': dates,
    'hour': hour,
    'day_of_week': day_of_week,
    'month': month,
    'temperature': np.round(temperature, 1),
    'humidity': np.round(humidity, 1),
    'precipitation': np.round(precipitation, 1),
    'is_holiday': is_holiday.astype(int),
    'foot_traffic': np.round(foot_traffic, 0).astype(int),  # 추가 기능 1
    'nearby_school': nearby_school,  # 추가 기능 2
    'nearby_facilities': nearby_facilities,  # 추가 기능 3
    'traffic_congestion': np.round(traffic_congestion, 1),
    'rental_count': rental_count
})

print("✅ 데이터 생성 완료!")
print()
print("[데이터셋 정보]")
print(f"- 데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")
print(f"- 기간: {df['date'].min()} ~ {df['date'].max()}")
print()

# ============================================================================
# 2단계: 탐색적 데이터 분석 (EDA)
# ============================================================================
print("=" * 80)
print("2단계: 탐색적 데이터 분석 (EDA)")
print("=" * 80)
print()

print("[기본 통계량]")
print(df[['temperature', 'humidity', 'precipitation', 'rental_count', 
          'foot_traffic', 'nearby_school', 'nearby_facilities']].describe())
print()

print("[결측치 확인]")
print(df.isnull().sum())
print()

print("[입력 변수별 대여량 상관관계]")
correlation = df[['temperature', 'humidity', 'precipitation', 'is_holiday',
                   'foot_traffic', 'nearby_school', 'nearby_facilities', 
                   'traffic_congestion', 'rental_count']].corr()
print(correlation['rental_count'].sort_values(ascending=False))
print()

# ============================================================================
# 3단계: 특성 공학 (Feature Engineering)
# ============================================================================
print("=" * 80)
print("3단계: 특성 공학 (Feature Engineering)")
print("=" * 80)
print()

df_processed = df.copy()

# 시간대별 더미 변수 생성 (출퇴근 시간, 업무시간, 야간 등)
df_processed['is_morning_peak'] = ((df_processed['hour'] >= 7) & (df_processed['hour'] <= 9)).astype(int)  # 아침 출근
df_processed['is_evening_peak'] = ((df_processed['hour'] >= 17) & (df_processed['hour'] <= 19)).astype(int)  # 저녁 퇴근
df_processed['is_lunch_time'] = ((df_processed['hour'] >= 11) & (df_processed['hour'] <= 13)).astype(int)  # 점심시간
df_processed['is_night'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] <= 5)).astype(int)  # 야간

# 요일별 카테고리 변수
df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)

# 계절 변수
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # 겨울
    elif month in [3, 4, 5]:
        return 1  # 봄
    elif month in [6, 7, 8]:
        return 2  # 여름
    else:
        return 3  # 가을

df_processed['season'] = df_processed['month'].apply(get_season)

# 날씨 카테고리
df_processed['is_rainy'] = (df_processed['precipitation'] > 0).astype(int)
df_processed['temperature_comfort'] = (
    (df_processed['temperature'] >= 15) & 
    (df_processed['temperature'] <= 25)
).astype(int)

print("✅ 특성 공학 완료!")
print(f"생성된 새로운 특성: {len(df_processed.columns) - len(df.columns)}개")
print()

# ============================================================================
# 4단계: 데이터 전처리 (Data Preprocessing)
# ============================================================================
print("=" * 80)
print("4단계: 데이터 전처리 (Data Preprocessing)")
print("=" * 80)
print()

# 입력 특성(X)과 종속변수(y) 분리
feature_columns = [
    'hour', 'day_of_week', 'month',
    'temperature', 'humidity', 'precipitation', 
    'is_holiday',
    'foot_traffic',  # 추가 기능 1
    'nearby_school',  # 추가 기능 2
    'nearby_facilities',  # 추가 기능 3
    'traffic_congestion',
    'is_morning_peak', 'is_evening_peak', 'is_lunch_time', 'is_night',
    'is_weekend', 'season', 'is_rainy', 'temperature_comfort'
]

X = df_processed[feature_columns].copy()
y = df_processed['rental_count'].copy()

print(f"입력 특성 개수: {X.shape[1]}")
print(f"샘플 개수: {X.shape[0]}")
print()

# 훈련/테스트 데이터 분리 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"훈련 데이터: {X_train.shape[0]}개 (80%)")
print(f"테스트 데이터: {X_test.shape[0]}개 (20%)")
print()

# 정규화 (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ 데이터 전처리 완료!")
print()

# ============================================================================
# 5단계: 모델 학습 (Model Training)
# ============================================================================
print("=" * 80)
print("5단계: 모델 학습 (Model Training)")
print("=" * 80)
print()

models = {}

# 1) 선형 회귀 (기본 모델)
print("[1] 선형 회귀 모델 학습 중...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr_model
print("✅ 완료")

# 2) 랜덤 포레스트 (중급 모델)
print("[2] 랜덤 포레스트 모델 학습 중...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)  # 정규화 불필요
models['Random Forest'] = rf_model
print("✅ 완료")

# 3) XGBoost (고급 모델)
print("[3] XGBoost 모델 학습 중...")
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model
print("✅ 완료")

print()

# ============================================================================
# 6단계: 모델 평가 (Model Evaluation)
# ============================================================================
print("=" * 80)
print("6단계: 모델 평가 (Model Evaluation)")
print("=" * 80)
print()

results = []

for model_name, model in models.items():
    # 예측
    if model_name == 'Linear Regression':
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # 평가 지표 계산
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results.append({
        'Model': model_name,
        'Train RMSE': f"{train_rmse:.2f}",
        'Test RMSE': f"{test_rmse:.2f}",
        'Train R²': f"{train_r2:.4f}",
        'Test R²': f"{test_r2:.4f}",
        'Test MAE': f"{test_mae:.2f}"
    })
    
    print(f"[{model_name}]")
    print(f"  훈련 RMSE: {train_rmse:.2f}대")
    print(f"  테스트 RMSE: {test_rmse:.2f}대")
    print(f"  훈련 R²: {train_r2:.4f}")
    print(f"  테스트 R²: {test_r2:.4f}")
    print(f"  테스트 MAE: {test_mae:.2f}대")
    print()

# 최종 모델 선택 (Test R²가 가장 높은 모델)
best_model_name = 'XGBoost'  # 보통 XGBoost가 가장 우수
best_model = models[best_model_name]

print(f"🏆 최종 선택 모델: {best_model_name}")
print()

# ============================================================================
# 7단계: 추가 기능(유동인구, 주변시설) 영향도 분석
# ============================================================================
print("=" * 80)
print("7단계: 추가 기능 영향도 분석")
print("=" * 80)
print()

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("[상위 10개 중요 특성]")
    print(feature_importance.head(10))
    print()
    
    # 추가 기능의 중요도
    additional_features = feature_importance[
        feature_importance['Feature'].isin(['foot_traffic', 'nearby_school', 'nearby_facilities'])
    ]
    
    print("[추가 기능의 중요도]")
    print(additional_features)
    print()

# ============================================================================
# 8단계: 예측 예시
# ============================================================================
print("=" * 80)
print("8단계: 예측 예시")
print("=" * 80)
print()

# 테스트 데이터에서 샘플 선택
sample_idx = 100
sample_input = X_test.iloc[sample_idx:sample_idx+5]
actual_values = y_test.iloc[sample_idx:sample_idx+5].values
predicted_values = best_model.predict(sample_input)

print("[예측 결과 샘플]")
for i in range(5):
    print(f"예시 {i+1}")
    print(f"  - 실제 대여량: {actual_values[i]}대")
    print(f"  - 예측 대여량: {int(predicted_values[i])}대")
    print(f"  - 오차: {abs(actual_values[i] - int(predicted_values[i]))}대")
    print()

print("=" * 80)
print("✅ 프로젝트 완료!")
print("=" * 80)