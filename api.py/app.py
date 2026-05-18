import streamlit as st
import pandas as pd
import numpy as np
import requests

def get_nearby_school_count(lat, lng, api_key):
    if not api_key:
        return 0
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"category_group_code": "SC4", "y": lat, "x": lng, "radius": 1000}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()['meta']['total_count']
        else:
            return 0
    except:
        return 0

st.set_page_config(page_title="따릉이 수요 예측 대시보드", layout="wide")
st.title("🚲 따릉이 수요 예측 (실시간 카카오 API 연동)")

st.sidebar.header("API 설정")
kakao_api_key = st.sidebar.text_input("9f7b0fa620b4a25d3ce0eb7dc1291afb", type="password")

st.sidebar.header("환경 변수 설정")

location_coords = {
    "강남역 (강남구)": (37.4979, 127.0276),
    "여의도역 (영등포구)": (37.5215, 126.9246),
    "홍대입구역 (마포구)": (37.5568, 126.9245),
    "서울숲 (성동구)": (37.5443, 127.0440),
    "노원역 (노원구)": (37.6542, 127.0568)
}

location = st.sidebar.selectbox("현재 위치", list(location_coords.keys()))
current_hour = st.sidebar.slider("시간", 0, 23, 12)
temperature = st.sidebar.slider("기온 (℃)", -10, 35, 20)

lat, lng = location_coords[location]

if kakao_api_key:
    school_count = get_nearby_school_count(lat, lng, kakao_api_key)
else:
    school_count = 0

base_demand = 30 + (school_count * 5)

if current_hour in [8, 9, 18, 19]:
    base_demand *= 2.5
elif current_hour < 6:
    base_demand *= 0.2

if temperature < 0 or temperature > 30:
    base_demand *= 0.6
elif 15 <= temperature <= 25:
    base_demand *= 1.2

predicted_demand = int(base_demand)

st.subheader(f"📍 {location} (API 실시간 수집 - 반경 1km 학교 수: {school_count}개)")
st.subheader(f"예상 대여 수요량: {predicted_demand}대")

if predicted_demand > 80:
    st.error("🚨 혼잡 (자전거 부족 주의)")
elif predicted_demand > 40:
    st.warning("🟡 보통")
else:
    st.success("🟢 널널 (여유 있음)")

st.markdown("---")
st.subheader("📅 내일 혼잡 시간대 예측")
st.write(f"내일은 {location}의 특성을 반영하여 오전 8\~9시, 오후 6\~7시에 대여소가 매우 혼잡할 것으로 예상됩니다.")

hours = np.arange(24)
demands = np.random.randint(10, 30, size=24) + (base_demand * 0.2)
demands[8:10] += 40
demands[18:20] += 50

chart_data = pd.DataFrame({'시간': hours, '수요량': demands})
st.bar_chart(chart_data.set_index('시간'))