# step15_rolling_tuning.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import optuna  # pip install optuna (설치 안 되어있다면 터미널에서 설치)

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ [제어 스위치] 튜닝 모드 설정
# ==========================================
# True로 변경 시, 약 수 시간~수십 시간이 소요되는 극한의 자동 튜닝 진행
# False로 설정 시, 실무 검증된 '방어형 황금 파라미터'로 즉시 예측 진행
RUN_OPTUNA = True  
OPTUNA_TRIALS = 20 # 튜닝 시도 횟수 (늘릴수록 오래 걸리지만 더 정교해짐)

print("1. 24년, 25년 데이터를 병합하여 전체 시계열을 준비합니다...")
train_full = pd.read_csv('step4_final_ml_ready.csv') 
test_full = pd.read_csv('step4_final_ml_ready_test.csv')
df_all = pd.concat([train_full, test_full], ignore_index=True)

# =========================================================
# 📅 [타겟 구간 정의] 24년 10월 ~ 25년 12월 계절별 롤링 세팅
# =========================================================
# 각 타겟 계절마다 정확히 '직전 9개월'을 학습하도록 설정합니다.
seasons_config = [
    {'name': '24년 가을 (24.10~11)', 'train_start': 20240101, 'train_end': 20240930, 'test_start': 20241001, 'test_end': 20241130},
    {'name': '24/25 겨울 (24.12~25.02)', 'train_start': 20240301, 'train_end': 20241130, 'test_start': 20241201, 'test_end': 20250228},
    {'name': '25년 봄 (25.03~25.05)',   'train_start': 20240601, 'train_end': 20250228, 'test_start': 20250301, 'test_end': 20250531},
    {'name': '25년 여름 (25.06~25.08)', 'train_start': 20240901, 'train_end': 20250531, 'test_start': 20250601, 'test_end': 20250831},
    {'name': '25년 가을 (25.09~25.11)', 'train_start': 20241201, 'train_end': 20250831, 'test_start': 20250901, 'test_end': 20251130},
    {'name': '25년 겨울 (25.12)',       'train_start': 20250301, 'train_end': 20251130, 'test_start': 20251201, 'test_end': 20251231}
]

robust_features = [
    '대여소_ID_num', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]
cat_cols = ['대여소_ID_num', '요일', '주말_여부', '비옴_여부']

# 💡 실무 검증 완료된 강력한 방어형 파라미터 (Optuna 안 켤 때 기본 사용)
# L1/L2 규제와 데이터 Subsample(랜덤 솎아내기)을 추가하여 방어력을 극한으로 올림
golden_params = {
    'objective': 'regression', 'metric': 'rmse', 'random_state': 42,
    'n_estimators': 2000,       # 트리 개수를 넉넉히 주어 충분히 학습
    'learning_rate': 0.03,      # 학습 속도를 낮춰서 더 촘촘하고 세밀하게 학습
    'num_leaves': 63,           
    'max_depth': 10,
    'min_child_samples': 50,    # 노드당 최소 데이터 (아웃라이어 방어)
    'subsample': 0.8,           # 데이터의 80%만 무작위 사용 (과적합 방지)
    'subsample_freq': 1,
    'colsample_bytree': 0.8,    # 변수의 80%만 무작위 사용 (과적합 방지)
    'reg_alpha': 0.5,           # L1 규제 (잔가지치기)
    'reg_lambda': 1.0,          # L2 규제 (가중치 폭발 방지)
    'n_jobs': -1
}

results = []
print(f"\n2. 🚀 총 6개 구간 롤링 윈도우 예측 시작 (Optuna 튜닝 모드: {RUN_OPTUNA})\n" + "-"*60)

for config in seasons_config:
    season_name = config['name']
    print(f"🔄 [{season_name}] 파이프라인 가동 중...")
    
    # [1] 데이터 분할
    train_df = df_all[(df_all['대여일자'] >= config['train_start']) & (df_all['대여일자'] <= config['train_end'])].copy()
    test_df = df_all[(df_all['대여일자'] >= config['test_start']) & (df_all['대여일자'] <= config['test_end'])].copy()
    
    # [2] 타겟 인코딩 (Data Leakage 차단)
    profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
    profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)
    train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
    test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

    fallback = train_df['총_대여건수(Y)'].mean()
    train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback)
    test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback)
    
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    X_train, y_train = train_df[robust_features], train_df['총_대여건수(Y)']
    X_test, y_test = test_df[robust_features], test_df['총_대여건수(Y)']
    
    # [3] 모델 학습 및 튜닝
    current_params = golden_params.copy()

    if RUN_OPTUNA:
        print("   🔍 Optuna 자동 튜닝 진행 중...")
        def objective(trial):
            param = {
                'objective': 'regression', 'metric': 'rmse', 'random_state': 42,
                'n_estimators': 1000, 'n_jobs': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 127),
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'subsample_freq': 1,
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            opt_model = lgb.LGBMRegressor(**param)
            opt_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], categorical_feature=cat_cols, callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = np.clip(opt_model.predict(X_test), 0, None)
            return np.sqrt(mean_squared_error(y_test, preds))

        study = optuna.create_study(direction='minimize')
        # 진행률을 보고 싶다면 아래 주석을 해제하세요
        # optuna.logging.set_verbosity(optuna.logging.INFO)
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        
        print(f"   ✨ 튜닝 완료! (Best RMSE: {study.best_value:.4f})")
        current_params.update(study.best_params)

    # 최종 파라미터로 모델 재학습 (Early Stopping 적용)
    model = lgb.LGBMRegressor(**current_params)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # [4] 예측 및 평가
    preds = np.clip(model.predict(X_test), a_min=0, a_max=None)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results.append({
        '계절 (타겟)': season_name,
        'RMSE': rmse,
        'R²': r2
    })
    print(f" ✔️ 평가 완료 (RMSE: {rmse:.4f} / R²: {r2:.4f})")

print("\n" + "="*60)
print(" 🚀 [최종 MLOps 완성] 24.10 ~ 25.12 롤링 예측 성적표 🚀")
print("="*60)
res_df = pd.DataFrame(results).set_index('계절 (타겟)')
print(res_df.to_string(float_format=lambda x: f"{x:,.4f}"))
print("="*60)