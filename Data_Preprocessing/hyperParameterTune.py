import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# ==========================================
# 0. 데이터 불러오기
# ==========================================
print("마스터 데이터셋을 불러오는 중입니다... (시간이 조금 걸릴 수 있습니다)")
# 본인의 실제 마스터 파일명으로 변경해주세요 (예: '마스터데이터.csv')
df = pd.read_csv('step4_final_ml_ready.csv') 

# ==========================================
# 1. 튜닝용 데이터 샘플링 및 분리
# ==========================================
print("데이터 로드 완료! 튜닝을 위해 20% 샘플링을 진행합니다...")
tune_df = df.sample(frac=0.2, random_state=42) 

# [수정됨] 실제 데이터에 맞게 제거할 컬럼과 정답 컬럼 지정
# 대여일자는 고유한 날짜값이므로 학습에서 제외하고, 총_대여건수(Y)는 정답이므로 X에서 뺍니다.
X_tune = tune_df.drop(['총_대여건수(Y)', '대여일자'], axis=1) 
y_tune = tune_df['총_대여건수(Y)']

# [안전장치 추가] LightGBM을 위해 문자열 데이터('요일' 등)를 category 타입으로 변환
for col in X_tune.select_dtypes(include=['object']).columns:
    X_tune[col] = X_tune[col].astype('category')

# Train / Valid 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_tune, y_tune, test_size=0.2, random_state=42)


# ==========================================
# 2. Optuna 목적 함수 정의
# ==========================================
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 7, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'n_estimators': 1000 # 넉넉히 주고 Early stopping으로 멈춤
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # 모델 학습 (Early Stopping 적용)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # 검증 데이터로 예측 및 RMSE 계산
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse


# ==========================================
# 3. 최적화 실행
# ==========================================
print("Optuna 하이퍼파라미터 튜닝을 시작합니다...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30) # 30번 탐색

print("\n==================================")
print("🏆 튜닝 완료! 최적의 파라미터 🏆")
print(study.best_params)
print(f"Best RMSE: {study.best_value:.4f}")
print("==================================")