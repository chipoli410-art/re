import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings('ignore')

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
X_tr = train_df.iloc[:tune_split_idx][features]
y_tr = train_df.iloc[:tune_split_idx]['총_대여건수(Y)']

X_val = train_df.iloc[tune_split_idx:][features]
y_val = train_df.iloc[tune_split_idx:]['총_대여건수(Y)']

# 🌟 [핵심] 타겟 변수 로그(Log) 변환 적용
# 0이 많은 비대칭 데이터를 정규분포 형태로 부드럽게 펴줍니다.
y_tr_log = np.log1p(y_tr)
y_val_log = np.log1p(y_val)

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 200), # 잎사귀 제한 상향 (더 깊은 패턴 학습)
        'max_depth': trial.suggest_int('max_depth', 7, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': 800
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # 모델은 '로그 변환된 값'을 보고 학습합니다.
    model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], callbacks=[lgb.early_stopping(30, verbose=False)])
    
    # 예측 후 지수 변환(expm1)하여 원래 단위(자전거 대수)로 복원
    preds_log = model.predict(X_val)
    preds_original = np.expm1(preds_log)
    
    # 원래 정답(y_val)과 복원된 예측값 간의 진짜 RMSE 반환
    return np.sqrt(mean_squared_error(y_val, preds_original))

print("2. [로그 변환 적용] Optuna 튜닝 시작 (총 50회 탐색)...")
# 50번을 탐색하므로 이전보다 시간이 조금 더 걸립니다.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("\n" + "="*40)
print("🏆 [튜닝 완료] 찾아낸 최적의 파라미터 (이 값을 3단계에 복사하세요!)")
print(study.best_params)
print(f"Best RMSE (원래 스케일 기준): {study.best_value:.4f}")
print("="*40)