import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="따릉이 수요 예측 대시보드", layout="wide")

st.title("🚲 따릉이 수요 예측 및 혼잡도 대시보드")

st.sidebar.header("환경 변수 설정")
current_hour = st.sidebar.slider("시간", 0, 23, 12)
temperature = st.sidebar.slider("기온 (℃)", -10, 35, 20)
school_count = st.sidebar.slider("주변 학교 수", 0, 5, 2)

base_demand = 10 + (school_count * 5)

if current_hour in [8, 9, 18, 19]:
    base_demand *= 2.5
elif current_hour < 6:
    base_demand *= 0.2

if temperature < 0 or temperature > 30:
    base_demand *= 0.6
elif 15 <= temperature <= 25:
    base_demand *= 1.2

predicted_demand = int(base_demand)

st.subheader(f"예상 대여 수요량: {predicted_demand}대")

if predicted_demand > 50:
    st.error("🚨 혼잡 (자전거 부족 주의)")
elif predicted_demand > 20:
    st.warning("🟡 보통")
else:
    st.success("🟢 널널 (여유 있음)")

st.markdown("---")
st.subheader("🚴 라이딩 날씨 지수")

if 15 <= temperature <= 25:
    st.info("쾌적함: 자전거 타기 완벽한 날씨!")
elif 5 <= temperature <= 14 or 26 <= temperature <= 29:
    st.warning("보통: 가벼운 겉옷이나 물을 챙기세요.")
else:
    st.error("주의: 야외 활동에 주의가 필요한 날씨입니다.")

st.markdown("---")
st.subheader("📅 내일 혼잡 시간대 예측")
st.write(f"내일은 주변 학교 수({school_count}개)의 영향을 받아 **오전 8~9시**, **오후 6~7시**에 대여소가 매우 혼잡할 것으로 예상됩니다.")

hours = np.arange(24)
demands = np.random.randint(5, 20, size=24) + (school_count * 3)
demands[8:10] += 30
demands[18:20] += 40

chart_data = pd.DataFrame({'시간': hours, '수요량': demands})
st.bar_chart(chart_data.set_index('시간'))