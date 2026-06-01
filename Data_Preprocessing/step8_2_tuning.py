import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

print("1. 전처리된 학습 데이터(Train)를 불러옵니다...")
train_df = pd.read_csv('train_preprocessed.csv')

features = [
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

# 카테고리 데이터 타입 변환
for col in train_df[features].select_dtypes(include=['object']).columns:
    train_df[col] = train_df[col].astype('category')

# 튜닝 전용 Train / Valid 시계열 분리 (8:2)
tune_split_idx = int(len(train_df) * 0.8)
X_tr, y_tr = train_df.iloc[:tune_split_idx][features], train_df.iloc[:tune_split_idx]['총_대여건수(Y)']
X_val, y_val = train_df.iloc[tune_split_idx:][features], train_df.iloc[tune_split_idx:]['총_대여건수(Y)']

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 150),
        'max_depth': trial.suggest_int('max_depth', 7, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': 500
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

print("2. Optuna 튜닝 시작 (총 20회 탐색)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# 💡 [결과 확인] 최적의 파라미터 딕셔너리 출력
print("\n" + "="*40)
print("🏆 [튜닝 완료] 찾아낸 최적의 파라미터 (이 값을 3단계에 복사하세요!)")
print(study.best_params)
print(f"Best RMSE: {study.best_value:.4f}")
print("="*40)