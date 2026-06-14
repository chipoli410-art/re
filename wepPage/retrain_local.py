# retrain_local.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
import os

warnings.filterwarnings('ignore')

def run_monthly_local_retraining(execution_year, execution_month):
    print(f"🔄 [로컬 MLOps] {execution_year}년 {execution_month}월 기준 재학습 파이프라인 가동")
    
    # 1. 📅 롤링 윈도우 날짜 계산 (직전 9개월)
    current_target_dt = datetime(execution_year, execution_month, 1)
    end_dt = current_target_dt - relativedelta(days=1)
    start_dt = current_target_dt - relativedelta(months=9)
    
    int_start = int(start_dt.strftime('%Y%m%d'))
    int_end = int(end_dt.strftime('%Y%m%d'))
    
    print(f" 📂 로컬 파일 분석 중... 학습 데이터 구간: {int_start} ~ {int_end}")
    
    # 2. 로컬 저장소의 '학습용' CSV 파일 로드
    try:
        df_all = pd.read_csv('total_clean_data.csv')
    except FileNotFoundError:
        print("❌ 에러: 'total_clean_data.csv' 파일이 존재하지 않습니다.")
        return
        
    train_df = df_all[(df_all['대여일자'] >= int_start) & (df_all['대여일자'] <= int_end)].copy()
    
    if train_df.empty:
        print("❌ 경고: 해당 기간에 알맞은 데이터가 로컬 파일에 없습니다.")
        return
        
    # 3. 최신 9개월 기준 '과거 평균 대여량' 프로필 생성
    profile_df = train_df.groupby(['대여소_ID_num', '요일', '대여시간(시)'])['총_대여건수(Y)'].mean().reset_index()
    profile_df.rename(columns={'총_대여건수(Y)': '과거_평균_대여량'}, inplace=True)
    
    train_df = pd.merge(train_df, profile_df, on=['대여소_ID_num', '요일', '대여시간(시)'], how='left')
    train_df['과거_평균_대여량'] = train_df['과거_평균_대여량'].fillna(train_df['총_대여건수(Y)'].mean())
    
    # 4. 범주형 변수 및 피처 세팅
    cat_cols = ['대여소_ID_num', '요일', '주말_여부', '비옴_여부']
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        
    robust_features = [
        '대여소_ID_num', '요일', '대여시간(시)', '주말_여부', 
        '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', 
        '지하철역_수_1km', '학교_수_1km'
    ]
    X_train, y_train = train_df[robust_features], train_df['총_대여건수(Y)']
    
    # 5. 👑 검증된 하이퍼파라미터 적용 및 모델 학습
    golden_params = {
        'objective': 'regression', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1,
        'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 10, 'num_leaves': 63,
        'min_child_samples': 50, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
        'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8,
        'verbose': -1
    }
    
    print(" 🏋️‍♂️ 로컬 LightGBM 모델 학습 중 (잠시만 기다려주세요)...")
    model = lgb.LGBMRegressor(**golden_params)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    
    # 6. 💾 모델 및 과거 평균 대여량 파일 저장
    joblib.dump(model, 'ttareungi_model_v1.pkl')
    profile_df.to_csv('rolling_profile_9m.csv', index=False)
    
  # ====================================================================
    # 🌟 7. [좌표 반영 버전] 초경량 마스터 메타 정보 구축
    # ====================================================================
    print(" 🗺️ 매핑 테이블을 이용해 실제 위치 좌표가 포함된 마스터 메타를 구축합니다...")
    
    infra_df = train_df[['대여소_ID_num', '지하철역_수_1km', '학교_수_1km']].drop_duplicates()
    mapping_file = 'station_id_mapping.csv'
    
    if os.path.exists(mapping_file):
        mapping_df = pd.read_csv(mapping_file)
        
        # 인프라 데이터에 한글 이름과 [위도, 경도]를 한 번에 결합
        final_meta = pd.merge(infra_df, mapping_df, on='대여소_ID_num', how='left')
        print("   ✔️ 실제 위치 좌표 및 대여소명 단어장 연동 완료!")
    else:
        print(f"   ⚠️ '{mapping_file}' 파일이 없습니다. 임시 값을 부여합니다.")
        final_meta = infra_df.copy()
        final_meta['대여소명'] = final_meta['대여소_ID_num'].apply(lambda x: f"알수없음(ID:{int(x)})")
        final_meta['위도'] = 37.5665
        final_meta['경도'] = 126.9780
        
    final_meta = final_meta[['대여소명', '대여소_ID_num', '지하철역_수_1km', '학교_수_1km', '위도', '경도']]
    final_meta['대여소명'] = final_meta['대여소명'].fillna(final_meta['대여소_ID_num'].apply(lambda x: f"알수없음(ID:{int(x)})"))
    final_meta['위도'] = final_meta['위도'].fillna(37.5665)
    final_meta['경度'] = final_meta['경도'].fillna(126.9780)
    
    final_meta.to_csv('station_meta.csv', index=False)
    print(" ✔️ [완료] 실제 위치 좌표가 반영된 'station_meta.csv' 최신화 완료!\n")
    
if __name__ == "__main__":
    run_monthly_local_retraining(execution_year=2026, execution_month=6)