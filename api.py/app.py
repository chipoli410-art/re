import streamlit as st
import pandas as pd
import numpy as np
import requests

# 카카오 API로 주변 인프라 개수 세어오는 함수 (카테고리 코드 추가)
def get_nearby_poi_count(lat, lng, api_key, category_code):
    if not api_key:
        return 0
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"category_group_code": category_code, "y": lat, "x": lng, "radius": 1000}
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()['meta']['total_count']
    except:
        pass
    return 0

st.set_page_config(page_title="따릉이 수요 예측 대시보드", layout="wide")
st.title("🚲 따릉이 다차원 수요 예측 대시보드")

st.sidebar.header("🔑 API 설정")
kakao_api_key = st.sidebar.text_input("9f7b0fa620b4a25d3ce0eb7dc1291afb", type="password")

st.sidebar.header("🌍 시뮬레이션 환경 설정")

location_coords = {
    "강남역 (오피스/환승)": (37.4979, 127.0276),
    "여의도역 (오피스/공원)": (37.5215, 126.9246),
    "홍대입구역 (대학가/유흥)": (37.5568, 126.9245),
    "서울숲 (여가/공원)": (37.5443, 127.0440),
    "노원역 (주거/학원가)": (37.6542, 127.0568)
}

location = st.sidebar.selectbox("📍 지역 선택", list(location_coords.keys()))
day_type = st.sidebar.radio("📅 요일 유형", ["평일 (출퇴근 위주)", "주말/공휴일 (여가 위주)"])
weather_condition = st.sidebar.selectbox("☔ 기상 상태", ["맑음", "비/눈", "미세먼지 나쁨"])
current_hour = st.sidebar.slider("⏰ 시간대", 0, 23, 18)

# API를 통해 학교(SC4)와 지하철역(SW8) 개수 실시간 수집
lat, lng = location_coords[location]
school_count = get_nearby_poi_count(lat, lng, kakao_api_key, "SC4") if kakao_api_key else 0
subway_count = get_nearby_poi_count(lat, lng, kakao_api_key, "SW8") if kakao_api_key else 0

# 🧠 수요 예측 로직 시작
base_demand = 20 + (school_count * 3) + (subway_count * 8) # 지하철역이 학교보다 가중치가 큼

# 1. 요일 및 시간대별 패턴 가중치 적용
if "평일" in day_type:
    if current_hour in [8, 9, 18, 19]:
        base_demand *= 2.8 # 평일 출퇴근 폭발
    elif current_hour < 6:
        base_demand *= 0.1
else:
    if 14 <= current_hour <= 18:
        base_demand *= 2.0 # 주말 오후 나들이 폭발
    elif current_hour < 8:
        base_demand *= 0.2

# 2. 날씨 가중치 적용 (가장 치명적)
if weather_condition == "비/눈":
    base_demand *= 0.1  # 비 오면 90% 감소
elif weather_condition == "미세먼지 나쁨":
    base_demand *= 0.6  # 미세먼지 심하면 40% 감소

predicted_demand = int(base_demand)

# 🖥️ 화면 출력부
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"📍 {location.split()[0]} 인프라 데이터")
    st.write(f"- 반경 1km 내 학교 수: **{school_count}개**")
    st.write(f"- 반경 1km 내 지하철역: **{subway_count}개**")

with col2:
    st.subheader("📊 실시간 예측 결과")
    st.markdown(f"### 예상 수요량: **{predicted_demand}대**")
    if predicted_demand > 80:
        st.error("🚨 매우 혼잡 (자전거 재배치 트럭 출동 요망)")
    elif predicted_demand > 40:
        st.warning("🟡 보통 (주의 관찰)")
    else:
        st.success("🟢 널널 (대여소 여유)")

st.markdown("---")
st.subheader(f"📈 {day_type.split()[0]} 24시간 예상 수요 패턴")

# 평일/주말에 따른 그래프 모양 변경
hours = np.arange(24)
demands = np.random.randint(5, 15, size=24) + int(base_demand * 0.1)

if "평일" in day_type:
    demands[8:10] += 50 + (subway_count * 5)
    demands[18:20] += 60 + (subway_count * 5)
else:
    demands[14:19] += 40 + (school_count * 3)

chart_data = pd.DataFrame({'시간': hours, '수요량': demands})
st.bar_chart(chart_data.set_index('시간'))