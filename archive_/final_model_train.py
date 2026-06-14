import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 깨짐 방지 설정 (윈도우 기준 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 전체 데이터 로드 및 전처리
# ==========================================
print("전체 마스터 데이터셋(100%)을 불러오는 중...")
df = pd.read_csv('step4_final_ml_ready.csv') # 본인의 파일명으로 변경!

print("학습용 데이터 준비 중...")
X = df.drop(['총_대여건수(Y)', '대여일자'], axis=1) 
y = df['총_대여건수(Y)']

# 범주형(문자열) 변수 category 타입 변환
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

# 전체 데이터 Train / Test 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. 튜닝된 최적의 파라미터로 최종 모델 학습
# ==========================================
print("최종 모델 학습을 시작합니다... (데이터가 커서 수 분 정도 소요될 수 있습니다)")

# Optuna가 찾아준 최고의 파라미터 적용
best_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': 42,
    'learning_rate': 0.08481707241420981,
    'num_leaves': 115,
    'max_depth': 14,
    'min_child_samples': 29,
    'subsample': 0.6952482730656839,
    'colsample_bytree': 0.8954572829873737,
    'n_estimators': 1000 # Early stopping을 위해 넉넉히 설정
}

final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
)

# ==========================================
# 3. 최종 성능 평가 (RMSE, R2)
# ==========================================
preds = final_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, preds))
final_r2 = r2_score(y_test, preds)

print("\n" + "="*40)
print("🚀 [최종 모델 성능 평가 결과] 🚀")
print(f"최종 RMSE : {final_rmse:.4f}")
print(f"최종 R²   : {final_r2:.4f}")
print("="*40 + "\n")

# ==========================================
# 4. 변수 중요도(Feature Importance) 시각화 저장
# ==========================================
print("변수 중요도 그래프를 저장합니다...")
plt.figure(figsize=(10, 8))
lgb.plot_importance(final_model, max_num_features=15, height=0.6, 
                    title='LightGBM 변수 중요도 (Feature Importance)', 
                    xlabel='중요도 (기여도)', ylabel='변수명',
                    importance_type='gain') # 'gain': 오차를 줄이는데 기여한 정도
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

print("모든 과정이 완료되었습니다! 'feature_importance.png' 파일을 확인해보세요.")