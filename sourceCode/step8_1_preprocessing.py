import pandas as pd

print("1. 마스터 데이터를 불러옵니다... (step4_final_ml_ready.csv)")
df = pd.read_csv('step4_final_ml_ready.csv') 

print("2. 시간순 정렬 및 Train / Test (8:2) 분리 중...")
df = df.sort_values(by=['대여일자', '대여시간(시)']).reset_index(drop=True)
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("3. '과거 평균 대여량' 파생 변수를 생성하고 결합합니다...")
profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)

train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
test_df = pd.merge(test_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')

fallback = train_df['총_대여건수(Y)'].mean()
train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(fallback)
test_df['과거_평균_대여량'] = test_df['과거_평균_대여량'].fillna(fallback)

# 💡 [결과 확인] 최종 전처리된 데이터가 어떤 수치들로 이루어져 있는지 출력
print("\n👀 [확인] 최종 전처리된 학습 데이터(Train) 상위 5줄 미리보기:")
features_to_show = ['대여일자', '대여시간(시)', '기온', '과거_평균_대여량', '총_대여건수(Y)']
print(train_df[features_to_show].head(5))

print("\n4. 변환 완료된 데이터를 CSV로 저장합니다...")
train_df.to_csv('train_preprocessed.csv', index=False)
test_df.to_csv('test_preprocessed.csv', index=False)
print("✅ 완료! [train_preprocessed.csv], [test_preprocessed.csv] 파일이 저장되었습니다.")