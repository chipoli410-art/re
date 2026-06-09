import pandas as pd
import os

# ==========================================
# ⚙️ 1. 설정: 읽어올 파일 목록과 타겟 날짜 지정
# ==========================================
# 합치고자 하는 년도별 파일 이름들을 리스트에 넣어주세요.
file_list = [
    'step4_final_ml_ready_25.csv',  # 25년도 파일 (예시)
    'step4_final_ml_ready_26.csv'   # 26년도 파일 (예시)
]

# 추출할 9개월 타겟 날짜 (YYYYMMDD 형식)
start_date = 20250901
end_date = 20260531

print(f"🎯 데이터 추출 구간: {start_date} ~ {end_date}")
print("-" * 50)

merged_chunks = []

# ==========================================
# 🔄 2. 파일 순회 및 조건부 필터링 (ETL 과정)
# ==========================================
for file_name in file_list:
    if os.path.exists(file_name):
        print(f"📂 '{file_name}' 파일을 읽는 중...")
        
        # 파일이 클 경우를 대비해 메모리를 절약하며 읽어옵니다.
        df = pd.read_csv(file_name)
        
        # '대여일자'가 타겟 기간 안에 있는 행(Row)만 싹둑 잘라냅니다.
        filtered_df = df[(df['대여일자'] >= start_date) & (df['대여일자'] <= end_date)]
        
        if not filtered_df.empty:
            merged_chunks.append(filtered_df)
            print(f"   ✔️ 해당 기간 데이터 {len(filtered_df):,}건 추출 완료!")
        else:
            print("   ⚠️ 이 파일에는 타겟 기간의 데이터가 존재하지 않습니다.")
    else:
        print(f"❌ 에러: '{file_name}' 파일을 찾을 수 없습니다. 이름이 맞는지 확인해 주세요.")

print("-" * 50)

# ==========================================
# 💾 3. 최종 병합 및 파일로 굽기
# ==========================================
if merged_chunks:
    print("🔄 추출된 데이터 블록들을 하나로 병합하는 중...")
    
    # 리스트에 모인 데이터프레임들을 위아래로 이어 붙입니다.
    final_df = pd.concat(merged_chunks, ignore_index=True)
    
    # 시계열 데이터의 안정성을 위해 날짜와 시간순으로 정렬해 줍니다.
    final_df = final_df.sort_values(by=['대여일자', '대여시간(시)']).reset_index(drop=True)
    
    # MLOps 재학습 스크립트가 요구하는 이름으로 저장합니다.
    output_filename = 'total_clean_data.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"✨ 성공! 총 {len(final_df):,}건의 정제된 9개월 치 데이터가 '{output_filename}'로 저장되었습니다.")
else:
    print("❌ 병합할 데이터가 없습니다. 날짜 조건이나 파일 데이터를 다시 확인해 주세요.")