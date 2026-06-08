# step13_seasonal_evaluation.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

print("1. [정제 완료] 24년(학습) 및 25년(테스트) 데이터 로드 중...")
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

# 과적합 방어 변수 세팅 (달력 제외)
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

print("\n3. 단일 회귀 모델(LightGBM) 학습 중...")
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

print("\n4. 2025년 전체 예측 및 사계절 분리 평가 진행...")
# 예측 수행 및 클리핑 (0 이하의 수요는 없으므로)
test_df['예측_대여건수'] = np.clip(model.predict(X_test), a_min=0, a_max=None)

# 날짜(YYYYMMDD)에서 '월' 추출
test_df['월'] = (test_df['대여일자'] // 100) % 100

# 한국의 일반적인 4계절 기준 매핑
def get_season(month):
    if month in [3, 4, 5]: return '봄'
    elif month in [6, 7, 8]: return '여름'
    elif month in [9, 10, 11]: return '가을'
    else: return '겨울' # 12, 1, 2월

test_df['계절'] = test_df['월'].apply(get_season)

print("\n" + "="*50)
print(" 🌸☀️🍂❄️ [사계절 예측 난이도 최종 성적표] 🌸☀️🍂❄️")
print("="*50)

seasons = ['봄', '여름', '가을', '겨울']
results = []

for season in seasons:
    mask = test_df['계절'] == season
    
    # 해당 계절의 데이터가 존재하는지 확인 (데이터 기간 필터링에 따라 없을 수도 있음)
    if mask.sum() == 0:
        continue
        
    y_true_season = test_df.loc[mask, '총_대여건수(Y)']
    y_pred_season = test_df.loc[mask, '예측_대여건수']
    
    rmse = np.sqrt(mean_squared_error(y_true_season, y_pred_season))
    r2 = r2_score(y_true_season, y_pred_season)
    variance = np.var(y_true_season) # 실제 수요의 분산(널뛰는 정도)
    
    results.append({
        '계절': season,
        'RMSE': rmse,
        'R²': r2,
        '분산(Variance)': variance
    })

# 결과를 DataFrame으로 변환하여 보기 좋게 출력
res_df = pd.DataFrame(results).set_index('계절')
print(res_df.to_string(float_format=lambda x: f"{x:,.4f}"))
print("="*50)