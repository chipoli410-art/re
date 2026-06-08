# step14_walk_forward_validation.py
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

# =========================================================
# 💡 [핵심] 사계절 타겟별 직전 9개월(Train) & 타겟 3개월(Test) 세팅
# =========================================================
# 데이터 범위를 고려하여 겨울은 24/25년도 겨울(12~2월)을 타겟으로 잡았습니다.
seasons_config = [
    {'name': '겨울 (24.12~25.02)', 'train_start': 20240301, 'train_end': 20241130, 'test_start': 20241201, 'test_end': 20250228},
    {'name': '봄 (25.03~25.05)',   'train_start': 20240601, 'train_end': 20250228, 'test_start': 20250301, 'test_end': 20250531},
    {'name': '여름 (25.06~25.08)', 'train_start': 20240901, 'train_end': 20250531, 'test_start': 20250601, 'test_end': 20250831},
    {'name': '가을 (25.09~25.11)', 'train_start': 20241201, 'train_end': 20250831, 'test_start': 20250901, 'test_end': 20251130}
]

robust_features = [
    '대여소_ID_num', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

cat_cols = ['대여소_ID_num', '요일', '주말_여부', '비옴_여부']

best_params = {
    'objective': 'regression', 'metric': 'rmse', 'random_state': 42,
    'n_estimators': 1500, 'n_jobs': -1, 'learning_rate': 0.05, 
    'num_leaves': 63, 'max_depth': 10
}

results = []

print("\n2. 계절별 롤링 윈도우(Rolling Window) 학습 및 평가 시작!\n" + "-"*50)

for config in seasons_config:
    season_name = config['name']
    print(f"🔄 [{season_name}] 모델 준비 중...")
    
    # 1. Train / Test 기간 엄격 분할 (미래 참조 완벽 차단)
    train_df = df_all[(df_all['대여일자'] >= config['train_start']) & (df_all['대여일자'] <= config['train_end'])].copy()
    test_df = df_all[(df_all['대여일자'] >= config['test_start']) & (df_all['대여일자'] <= config['test_end'])].copy()
    
    # 2. 🌟 직전 9개월(Train)의 데이터만으로 '과거 평균 대여량(타겟 인코딩)' 계산 🌟
    profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
    profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

    train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
    test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

    fallback = train_df['총_대여건수(Y)'].mean()
    train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback)
    test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback)
    
    # 3. 데이터 타입 세팅
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    X_train, y_train = train_df[robust_features], train_df['총_대여건수(Y)']
    X_test, y_test = test_df[robust_features], test_df['총_대여건수(Y)']
    
    # 4. 모델 학습 (해당 계절 전용 모델 탄생)
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # 5. 예측 및 평가
    preds = np.clip(model.predict(X_test), a_min=0, a_max=None)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    variance = np.var(y_test)
    
    results.append({
        '계절 (타겟)': season_name,
        'RMSE': rmse,
        'R²': r2,
        '분산(Variance)': variance
    })
    print(f" ✔️ 완료! (RMSE: {rmse:.4f} / R²: {r2:.4f})")

print("\n" + "="*60)
print(" 🚀 [Walk-Forward Validation] 사계절 직전 9개월 롤링 평가 🚀")
print("="*60)
res_df = pd.DataFrame(results).set_index('계절 (타겟)')
print(res_df.to_string(float_format=lambda x: f"{x:,.4f}"))
print("="*60)