import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("1. 1단계에서 저장한 Train / Test 데이터를 불러옵니다...")
train_df = pd.read_csv('train_preprocessed.csv')
test_df = pd.read_csv('test_preprocessed.csv')

features = [
    '대여소_ID_num', '월', '주차', '월별_주차', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
    '지하철역_수_1km', '학교_수_1km'
]

# 카테고리 변환
for col in train_df[features].select_dtypes(include=['object']).columns:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

X_train, y_train = train_df[features], train_df['총_대여건수(Y)']
X_test, y_test = test_df[features], test_df['총_대여건수(Y)']

print("2. 최종 모델 학습 중...")
# 🚨 2단계에서 출력된 study.best_params 값을 아래에 덮어씌워주세요! 
best_params = {
    'learning_rate': 0.0848, # (예시값) 2단계 결과를 넣어주세요
    'num_leaves': 115,       # (예시값)
    'max_depth': 14,         # (예시값)
    'min_child_samples': 29, # (예시값)
    'subsample': 0.695,      # (예시값)
    'colsample_bytree': 0.895 # (예시값)
}
best_params.update({'objective': 'regression', 'random_state': 42, 'n_estimators': 1000, 'n_jobs': -1})

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(30, verbose=False)])

final_preds = final_model.predict(X_test)

# 💡 [결과 확인 1] 최종 성능 수치
print("\n" + "="*40)
print(" 🚀 [최종 결과] 평가 지표 🚀")
print(f" - RMSE : {np.sqrt(mean_squared_error(y_test, final_preds)):.4f} 대")
print(f" - R²   : {r2_score(y_test, final_preds):.4f}")
print("="*40)

# 💡 [결과 확인 2] 변수 중요도 (Feature Importance) 실제 점수 출력
importance_df = pd.DataFrame({
    '변수명': features, 
    '기여도_점수(Gain)': final_model.feature_importances_
}).sort_values(by='기여도_점수(Gain)', ascending=False).reset_index(drop=True)

print("\n💡 [인사이트] 최종 모델의 변수별 기여도 실제 수치 랭킹:")
print(importance_df)

# 그래프 시각화 저장
plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='기여도_점수(Gain)', y='변수명', palette='viridis')
plt.title('최종 모델 변수 중요도 (Feature Importance)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('final_feature_importance.png', dpi=300)
print("\n✅ 'final_feature_importance.png' 파일이 생성되었습니다.")