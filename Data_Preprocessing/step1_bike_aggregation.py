import pandas as pd
import itertools
import glob
import os
from tqdm import tqdm
import warnings

# 불필요한 경고 메시지 숨김
warnings.filterwarnings('ignore')

print("1. [STEP 1] 'origin_data' 폴더 내의 파일 탐색 중...")
file_pattern = os.path.join("origin_data_26", "*.csv")  
file_list = glob.glob(file_pattern)

if not file_list:
    print("❌ 에러: 'origin_data' 폴더를 찾을 수 없거나 파일이 없습니다!")
    exit()

print(f"총 {len(file_list)}개의 파일을 발견했습니다. 전처리를 시작합니다...\n")

all_processed_data = []

def find_col(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def parse_hour(x):
    s = str(x).strip()
    s = ''.join(filter(str.isdigit, s))
    if not s: 
        return 0
    if len(s) >= 3: 
        return int(s[:-2])
    else:
        return int(s)

# ---------------------------------------------------------
# 2. 모든 파일을 순회하며 개별 전처리
# ---------------------------------------------------------
for file_path in tqdm(file_list, desc="개별 파일 집계 중"):
    df = None
    for enc in ['utf-8', 'cp949', 'utf-8-sig', 'euc-kr']:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
            
    if df is None:
        continue

    # 💡 [핵심 방어 1] 7월 이후 띄어쓰기가 들어간 컬럼명 완벽 대응 (모든 공백 제거)
    df.columns = df.columns.str.replace(r'\s+', '', regex=True)

    col_agg = find_col(df, ['집계_기준', '집계기준'])
    if col_agg is not None:
        df = df[df[col_agg].astype(str).str.strip() == '출발시간'].copy()
    
    # ==========================================================
    # 💡 [업데이트] 7월 이후 몰래 바뀐 공공데이터 컬럼명 완벽 추가
    # ==========================================================
    col_time = find_col(df, ['기준_시간대', '시간대', '대여시간', '대여시간(시)', '기준_시간'])
    col_date = find_col(df, ['기준_날짜', '대여일자', '일자'])
    col_st_id = find_col(df, ['시작_대여소_ID', '시작대여소ID', '시작대여소번호', '대여소_ID', '대여소번호', '대여소ID', '시작_대여소'])
    col_st_nm = find_col(df, ['시작_대여소명', '대여소명', '대여소이름'])
    col_count = find_col(df, ['전체_건수', '이용건수', '대여건수', '건수', '전체건수'])

    if not all([col_time, col_date, col_st_id, col_st_nm, col_count]):
        # 실패 시 도대체 어떤 컬럼명으로 바뀌었는지 확인하기 위해 출력
        print(f"\n[스킵] {os.path.basename(file_path)} | 현재 컬럼: {list(df.columns)}")
        continue
    
    try:
        df[col_time] = df[col_time].apply(parse_hour)
        df[col_date] = df[col_date].astype(str).str.replace(r'\D', '', regex=True).astype(int)
        
        # 💡 [핵심 방어 2] 침묵의 에러(KeyError) 방지 및 대여소 포맷 강제 통일
        df[col_st_id] = df[col_st_id].astype(str).str.extract(r'(\d+)', expand=False)
        df[col_st_id] = 'ST-' + df[col_st_id].fillna('0') 
        
        df[col_count] = df[col_count].astype(str).str.replace(',', '')
        df[col_count] = pd.to_numeric(df[col_count], errors='coerce').fillna(0).astype(int)
        
        df_grouped = df.groupby([col_date, col_time, col_st_id, col_st_nm])[col_count].sum().reset_index()
        df_grouped.columns = ['대여일자', '대여시간(시)', '대여소_ID', '대여소명', '총_대여건수(Y)']
        
        all_processed_data.append(df_grouped)
    except Exception as e:
        print(f"\n[에러 스킵] {os.path.basename(file_path)} 처리 중 문제 발생: {e}")
        continue

# ---------------------------------------------------------
# 3. 전체 데이터 결합
# ---------------------------------------------------------
print("\n3. 집계된 전체 데이터를 하나의 테이블로 병합 중...")
if not all_processed_data:
    print("❌ 에러: 병합할 데이터가 없습니다. 모든 파일이 스킵되었습니다.")
    exit()
    
combined_df = pd.concat(all_processed_data, ignore_index=True)
combined_df = combined_df.groupby(['대여일자', '대여시간(시)', '대여소_ID', '대여소명'])['총_대여건수(Y)'].sum().reset_index()

# ---------------------------------------------------------
# 4. [수정됨] 대여소별 생애주기 기반 스마트 제로 패딩
# ---------------------------------------------------------
print("4. 대여소 생애주기 기반 스마트 제로 패딩 진행 중... (시간이 조금 소요됩니다)")

unique_dates = combined_df['대여일자'].unique()
unique_dates.sort()
all_hours = list(range(24))

unique_stations = combined_df[['대여소_ID', '대여소명']].drop_duplicates(subset=['대여소_ID'], keep='last')

# 💡 대여소별 최초 운영일과 최종 운영일 파악
station_lifespan = combined_df.groupby('대여소_ID')['대여일자'].agg(최초일='min', 최종일='max').reset_index()

valid_grids = []

for _, row in station_lifespan.iterrows():
    st_id = row['대여소_ID']
    start_date = row['최초일']
    end_date = row['최종일']
    
    # 해당 대여소가 실제로 운영되었던 날짜 구간만 추출
    active_dates = unique_dates[(unique_dates >= start_date) & (unique_dates <= end_date)]
    
    grid = pd.DataFrame(list(itertools.product(active_dates, all_hours, [st_id])), columns=['대여일자', '대여시간(시)', '대여소_ID'])
    valid_grids.append(grid)

base_df = pd.concat(valid_grids, ignore_index=True)
base_df = pd.merge(base_df, unique_stations, on='대여소_ID', how='left')

df_final = pd.merge(base_df, combined_df, on=['대여일자', '대여시간(시)', '대여소_ID', '대여소명'], how='left')
df_final['총_대여건수(Y)'] = df_final['총_대여건수(Y)'].fillna(0).astype(int)

# ---------------------------------------------------------
# 5. 정렬 및 최종 파일 저장
# ---------------------------------------------------------
df_final = df_final.sort_values(by=['대여일자', '대여소_ID', '대여시간(시)']).reset_index(drop=True)

output_filename = "step1_aggregated_bike_smart_26.csv"
print(f"\n5. 최종 데이터 저장 중... (총 {len(df_final):,} 행)")
df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n🎉 완료! 유령 데이터가 제거된 아주 깔끔한 '{output_filename}' 파일이 생성되었습니다.")