import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('tpss_hourly_demand.csv')

hourly_demand = df.groupby('시간_시')['전체_건수'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=hourly_demand, x='시간_시', y='전체_건수', palette='viridis')

plt.title('시간대별 따릉이 총 대여 건수', fontsize=16)
plt.xlabel('시간 (시)', fontsize=12)
plt.ylabel('총 대여 건수', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()