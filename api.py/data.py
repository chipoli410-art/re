import pandas as pd
#feat(data): 원본 따릉이 데이터 시간대별/대여소별 수요 집계 로직 구현

file_path = r'C:\vscord_11\re\api.py\tpss_bcycl_od_statnhm_20260430.csv'
df = pd.read_csv(file_path, encoding='cp949')

df['시간_시'] = df['기준_시간대'] // 100
df_aggregated = df.groupby(['시작_대여소_ID', '시작_대여소명', '시간_시'])['전체_건수'].sum().reset_index()
df_aggregated.to_csv('tpss_hourly_demand.csv', index=False, encoding='utf-8-sig')