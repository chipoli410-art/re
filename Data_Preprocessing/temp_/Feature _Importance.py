import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import platform

# 1. 한글 폰트 깨짐 방지 설정 (운영체제별 세팅)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic' # 윈도우 맑은고딕
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'   # 맥 애플고딕
plt.rcParams['axes.unicode_minus'] = False        # 마이너스 기호 깨짐 방지

# 2. 학습된 모델 불러오기
print("저장된 모델을 불러옵니다...")
model = joblib.load('bike_demand_model.pkl')

# 3. 모델에 사용했던 변수(Feature) 이름들
features = [
    '대여소_ID_num', '월', '요일', '대여시간(시)', '주말_여부', 
    '기온', '강수량', '풍속', '습도', '비옴_여부', 
    '과거_평균_대여량' 
]

# 4. 모델에서 변수 중요도 추출
importances = model.feature_importances_

# 5. 보기 좋게 데이터프레임으로 만들고 점수(중요도) 순으로 정렬
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})
# 중요도가 높은 순으로 내림차순 정렬
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 6. 시각화 (막대 그래프)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')

plt.title('따릉이 대여 수요 예측에 가장 큰 영향을 미친 요인 (Feature Importance)', fontsize=15)
plt.xlabel('중요도 점수 (0~1 사이, 클수록 중요)', fontsize=12)
plt.ylabel('입력 변수', fontsize=12)
plt.tight_layout()

# 그래프를 이미지 파일로 저장 (PPT에 바로 쓰기 좋음!)
plt.savefig('feature_importance_plot.png', dpi=300)
plt.show()

print("\n상위 5개 중요 변수:")
print(importance_df.head(5))