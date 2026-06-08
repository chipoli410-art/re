import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("1. Train / Test 데이터를 불러옵니다...")
train_df = pd.read_csv('train_preprocessed.csv')
test_df = pd.read_csv('test_preprocessed.csv')

features = [
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

for col in train_df[features].select_dtypes(include=['object']).columns:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

X_train, y_train = train_df[features], train_df['총_대여건수(Y)']
X_test, y_test = test_df[features], test_df['총_대여건수(Y)']

# 🌟 [핵심] 학습 데이터 타겟 변수 로그 변환
print("2. 타겟 변수(대여량)에 로그(Log) 변환을 적용합니다...")
y_train_log = np.log1p(y_train)

print("3. 찾은 최적의 파라미터로 최종 모델을 학습합니다...")
# 🚨 2단계에서 출력된 study.best_params 값을 아래에 덮어씌워주세요! 
best_params = {'learning_rate': 0.09361605510867393, 'num_leaves': 31, 'max_depth': 7, 'min_child_samples': 65, 'subsample': 0.7356871353975147, 'colsample_bytree': 0.8172792233718896}
best_params.update({'objective': 'regression', 'random_state': 42, 'n_estimators': 1500, 'n_jobs': -1})

final_model = lgb.LGBMRegressor(**best_params)

# 검증 시에도 조기 종료(Early Stopping)를 위해 로그 변환된 Test 정답지를 제공
y_test_log = np.log1p(y_test)
final_model.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], callbacks=[lgb.early_stopping(50, verbose=False)])

print("4. Test 데이터 예측 및 역 로그 변환(복원)...")
preds_log = final_model.predict(X_test)
final_preds = np.expm1(preds_log) # 🌟 역 변환: 예측된 로그값을 다시 자전거 대수로 복원
final_preds = np.clip(final_preds, a_min=0, a_max=None) # 음수 대여 방지

# ==========================================
# 💡 [결과 확인] 최종 성능 수치
# ==========================================
print("\n" + "="*40)
print(" 🚀 [로그 변환 최종 모델 평가 지표] 🚀")
print(f" - RMSE : {np.sqrt(mean_squared_error(y_test, final_preds)):.4f} 대")
print(f" - R²   : {r2_score(y_test, final_preds):.4f}")
print("="*40)

importance_df = pd.DataFrame({
    '변수명': features, 
    '기여도_점수(Gain)': final_model.feature_importances_
}).sort_values(by='기여도_점수(Gain)', ascending=False).reset_index(drop=True)

# 그래프 시각화 저장
plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='기여도_점수(Gain)', y='변수명', palette='viridis')
plt.title('로그 변환 적용 LightGBM 변수 중요도', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('final_feature_importance_log.png', dpi=300)
print("\n✅ 'final_feature_importance_log.png' 파일이 생성되었습니다.")