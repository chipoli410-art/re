import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. 한글 폰트 및 그래프 스타일 설정
# ---------------------------------------------------------
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa", "figure.facecolor": "#ffffff"})
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic') 
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')

print("1. 데이터를 로드하는 중입니다... (데이터 크기에 따라 시간이 소요될 수 있습니다)")
df = pd.read_csv('step4_final_ml_ready.csv') 

print("\n" + "="*50)
print(" 🚀 따릉이 데이터 EDA 핵심 인사이트 리포트 🚀")
print("="*50)

# ---------------------------------------------------------
# 🌟 EDA를 위한 '과거_평균_대여량' 임시 계산 (Data Leakage 방지 반영)
# ---------------------------------------------------------
print("2. 파생 변수(과거_평균_대여량)를 동적으로 계산 중입니다...")
if '대여소_ID_num' in df.columns:
    station_col = '대여소_ID_num'
elif '대여소_ID' in df.columns:
    station_col = '대여소_ID'

profile = df.groupby([station_col, '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)
df = pd.merge(df, profile, on=[station_col, '요일', '대여시간(시)'], how='left')

# ---------------------------------------------------------
# 1. 시간대별 패턴 (평일 vs 주말 분리) ✨ [업데이트 완]
# ---------------------------------------------------------
print("3. 시간대별 패턴 분석 중 (평일/주말 분리)...")
plt.figure(figsize=(12, 6))

weekend_col = '주말_여부' if '주말_여부' in df.columns else '주말여부'
df['주말_라벨'] = df[weekend_col].astype(str).apply(
    lambda x: '주말 (Weekend)' if '1' in x or '주말' in x else '평일 (Weekday)'
)

hourly_demand = df.groupby(['대여시간(시)', '주말_라벨'])['총_대여건수(Y)'].mean().reset_index()

sns.lineplot(data=hourly_demand, x='대여시간(시)', y='총_대여건수(Y)', hue='주말_라벨', 
             marker='o', linewidth=2.5, markersize=8, palette=['#3b82f6', '#ef4444'])

plt.title('시간대별 평균 자전거 대여량 (평일 vs 주말 비교)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('대여 시간 (Hour)', fontsize=12)
plt.ylabel('평균 대여 건수', fontsize=12)
plt.xticks(range(0, 24))
plt.legend(title='구분', fontsize=11)
plt.tight_layout()
plt.savefig('eda_1_hourly_pattern_weekend.png', dpi=300)
plt.close()

# 텍스트 출력
weekday_data = hourly_demand[hourly_demand['주말_라벨'] == '평일 (Weekday)'].sort_values(by='총_대여건수(Y)', ascending=False)
weekend_data = hourly_demand[hourly_demand['주말_라벨'] == '주말 (Weekend)'].sort_values(by='총_대여건수(Y)', ascending=False)

print(f"\n[1. 시간대별 패턴 (평일 vs 주말)]")
print(f" - [평일] 출퇴근 피크 타임: {int(weekday_data.iloc[0]['대여시간(시)'])}시, {int(weekday_data.iloc[1]['대여시간(시)'])}시")
print(f" - [주말] 레저 피크 타임  : {int(weekend_data.iloc[0]['대여시간(시)'])}시, {int(weekend_data.iloc[1]['대여시간(시)'])}시")

# ---------------------------------------------------------
# 2. 요일별 패턴 (Day of week) - X축 한글화 및 정렬 적용
# ---------------------------------------------------------
print("4. 요일별 패턴 분석 중...")
plt.figure(figsize=(10, 6))

daily_demand = df.groupby('요일')['총_대여건수(Y)'].mean().reset_index()

# 요일이 숫자(0~6)인 경우를 대비한 매핑 딕셔너리
day_mapping = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}

# 요일 컬럼이 숫자형인지 확인하고, 맞다면 한글로 변환
if pd.api.types.is_numeric_dtype(daily_demand['요일']):
    daily_demand['요일명'] = daily_demand['요일'].map(day_mapping)
else:
    # 이미 문자열이라면 (예: '화(1)') 첫 글자('화')만 추출
    daily_demand['요일명'] = daily_demand['요일'].astype(str).str[0]
    
# 월요일부터 일요일까지 순서대로 정렬하기 위한 리스트
order = ['월', '화', '수', '목', '금', '토', '일']

# order 파라미터를 통해 월~일 순서 강제 고정
sns.barplot(data=daily_demand, x='요일명', y='총_대여건수(Y)', order=order, palette='viridis')
plt.title('요일별 평균 자전거 대여량', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('요일', fontsize=12)
plt.ylabel('평균 대여 건수', fontsize=12)
plt.tight_layout()
plt.savefig('eda_2_daily_pattern.png', dpi=300)
plt.close()

# 텍스트 출력도 한글 요일명으로 깔끔하게 나오도록 수정
daily_demand = daily_demand.set_index('요일명')
top_day = daily_demand['총_대여건수(Y)'].idxmax()
top_day_val = daily_demand['총_대여건수(Y)'].max()
bottom_day = daily_demand['총_대여건수(Y)'].idxmin()
bottom_day_val = daily_demand['총_대여건수(Y)'].min()

print(f"\n[2. 요일별 패턴]")
print(f" - 대여량이 가장 높은 요일: {top_day}요일 (평균 {top_day_val:.2f}건)")
print(f" - 대여량이 가장 낮은 요일: {bottom_day}요일 (평균 {bottom_day_val:.2f}건)")

# ---------------------------------------------------------
# 3. 기온과 대여량의 관계 (평균의 함정 파헤치기) ✨ [업데이트 완]
# ---------------------------------------------------------
print("5. 기온별 패턴 분석 중 (평균, 중앙값, 데이터 개수 동시 확인)...")

df['기온_구간'] = pd.cut(df['기온'], bins=range(-15, 45, 5), right=False)
temp_stats = df.groupby('기온_구간')['총_대여건수(Y)'].agg(['mean', 'median', 'count']).reset_index()
temp_stats['기온_구간'] = temp_stats['기온_구간'].astype(str)

print("\n[🚨 팩트 체크: 기온 구간별 상세 통계 🚨]")
print(temp_stats)

# 3-1. 기온별 데이터 개수 (Sample Size)
plt.figure(figsize=(10, 6))
sns.barplot(data=temp_stats, x='기온_구간', y='count', color='#94a3b8')
plt.title('기온 구간별 데이터 개수 (Sample Size)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('기온 구간 (℃)', fontsize=12)
plt.ylabel('데이터 개수', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_3_1_temperature_count.png', dpi=300)
plt.close()

# 3-2. 이상치에 흔들리지 않는 '중앙값(Median)' 그래프
plt.figure(figsize=(10, 6))
sns.barplot(data=temp_stats, x='기온_구간', y='median', color='#3b82f6')
plt.title('기온 구간별 자전거 대여량 [중앙값]', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('기온 구간 (℃)', fontsize=12)
plt.ylabel('대여 건수 중앙값', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_3_2_temperature_median.png', dpi=300)
plt.close()

# 텍스트 출력 (표본수가 너무 적은 구간은 제외하고 진짜 최적 온도 찾기)
valid_temp = temp_stats[temp_stats['count'] > 1000] # 최소 표본수 확보
if not valid_temp.empty:
    optimal_temp = valid_temp.loc[valid_temp['median'].idxmax()]
    print(f"\n[3. 기온의 영향 (중앙값 기준)]")
    print(f" - 자전거를 가장 많이 타는 최적 기온: {optimal_temp['기온_구간']}도 사이 (중앙값 {optimal_temp['median']:.1f}건)")

# ---------------------------------------------------------
# 4. 강수 여부 (Rain)
# ---------------------------------------------------------
print("6. 강수 여부 영향 분석 중...")

# 1. 맑은 날/비 오는 날 구분 컬럼 생성
df['날씨상태'] = df['비옴_여부'].apply(lambda x: '비 옴 (Rain)' if x == 1 else '맑음 (Clear)')

# 2. 박스플롯 대신 평균을 시각화하는 바플롯으로 변경
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=df, x='날씨상태', y='총_대여건수(Y)', palette=['#fbbf24', '#3b82f6'], errorbar=None)

# 3. 그래프 위에 평균값 텍스트 추가 (심사위원들이 직관적으로 이해하도록)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=14, fontweight='bold', padding=5)

plt.title('강수 여부에 따른 평균 대여량 비교', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('날씨 상태', fontsize=12)
plt.ylabel('평균 대여 건수', fontsize=12)
plt.tight_layout()
plt.savefig('eda_4_rain_impact.png', dpi=300)
plt.close()

# 텍스트 출력 (기존 로직 그대로 유지)
rain_demand = df.groupby('비옴_여부')['총_대여건수(Y)'].mean()
clear_mean = rain_demand[0] if 0 in rain_demand else 0
rain_mean = rain_demand[1] if 1 in rain_demand else 0
drop_rate = ((clear_mean - rain_mean) / clear_mean) * 100 if clear_mean > 0 else 0
print(f"\n[4. 비의 영향력]")
print(f" - 맑은 날 평균 대여량: {clear_mean:.2f}건")
print(f" - 비 오는 날 평균 대여량: {rain_mean:.2f}건")
print(f"   👉 비가 오면 수요가 평균적으로 약 {drop_rate:.1f}% 감소함!")

# ---------------------------------------------------------
# 5. 주요 변수 간의 상관관계 (Correlation)
# ---------------------------------------------------------
print("7. 상관관계 히트맵 분석 중...")
plt.figure(figsize=(10, 8))

candidate_cols = [
    '총_대여건수(Y)', '과거_평균_대여량', '대여시간(시)', '기온', '습도', '풍속', '지하철역_수_1km'
]
corr_cols = [col for col in candidate_cols if col in df.columns]

corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('주요 변수 간 상관관계', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_5_correlation_heatmap.png', dpi=300)
plt.close()

# 텍스트 출력
target_col = corr_cols[0] 
target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
print(f"\n[5. 정답({target_col})과의 상관관계 랭킹]")
print(f" 1위. {target_corr.index[0]} (상관계수: {target_corr.iloc[0]:.3f})")
print(f" 2위. {target_corr.index[1]} (상관계수: {target_corr.iloc[1]:.3f})")
print(f" 꼴찌. {target_corr.index[-1]} (상관계수: {target_corr.iloc[-1]:.3f}) - 방해 요소(음의 상관)")

print("\n" + "="*50)
print("✅ 분석 및 시각화 완벽하게 완료! 터미널의 결과를 복사해 주세요.")