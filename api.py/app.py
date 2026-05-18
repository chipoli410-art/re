import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import random

def get_nearby_poi_data(lat, lng, api_key, category_code):
    if not api_key:
        return [], "키 미입력"
    
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"category_group_code": category_code, "y": lat, "x": lng, "radius": 1000, "size": 15}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()['documents'], "성공"
        else:
            return [], f"에러코드 {response.status_code}"
    except Exception as e:
        return [], f"요청 실패: {str(e)}"

st.set_page_config(page_title="따릉이 수요 예측 대시보드", layout="wide")
st.title("🚲 따릉이 실시간 인프라 분석 & 수요 예측")

kakao_api_key = "09611d17ff9500ed2d94a6d607cf3609"

st.sidebar.header("🌍 시뮬레이션 환경 설정")
location_coords = {
    "강남역 (오피스/환승)": {"coords": (37.4979, 127.0276), "base": 60},
    "여의도역 (오피스/공원)": {"coords": (37.5215, 126.9246), "base": 55},
    "홍대입구역 (대학가/유흥)": {"coords": (37.5568, 126.9245), "base": 65},
    "서울숲역 (여가/공원)": {"coords": (37.5443, 127.0440), "base": 45},
    "노원역 (주거/학원가)": {"coords": (37.6542, 127.0568), "base": 50},
    "잠실역 (쇼핑/테마파크)": {"coords": (37.5133, 127.1001), "base": 65},
    "신도림역 (대형 환승거점)": {"coords": (37.5088, 126.8912), "base": 55},
    "건대입구역 (대학가/유흥)": {"coords": (37.5404, 127.0692), "base": 60}
}

location = st.sidebar.selectbox("📍 지역 선택", list(location_coords.keys()))
day_type = st.sidebar.radio("📅 요일 유형", ["평일 (출퇴근 위주)", "주말/공휴일 (여가 위주)"])
weather_condition = st.sidebar.selectbox("☔ 기상 상태", ["맑음", "비/눈", "미세먼지 나쁨"])
current_hour = st.sidebar.slider("⏰ 시간대", 0, 23, 18)

# 🚦 새로운 교통 상황 변수 추가!
traffic_condition = st.sidebar.selectbox("🚗 주변 교통 상황 (시뮬레이션)", ["원활 (초록)", "서행 (노랑)", "정체 (빨강)"])

lat, lng = location_coords[location]["coords"]
loc_base_demand = location_coords[location]["base"]

schools, s_status = get_nearby_poi_data(lat, lng, kakao_api_key, "SC4")
subways, sw_status = get_nearby_poi_data(lat, lng, kakao_api_key, "SW8")

school_count = len(schools)
subway_count = len(subways)

# --- 상단: 실시간 교통 알림 배너 (정체 시에만 등장) ---
if traffic_condition == "정체 (빨강)":
    st.error("🚨 **[교통 혼잡 알림]** 현재 대여소 주변 주요 도로가 매우 혼잡합니다. 자전거 이용 시 안전에 유의하시고, 재배치 트럭은 우회 도로를 이용해주세요!")
elif traffic_condition == "서행 (노랑)":
    st.warning("⚠️ **[교통 서행]** 주변 도로에 차량이 많습니다. 자전거 라이딩 시 주의가 필요합니다.")

st.subheader(f"🗺️ {location.split()[0]} 주변 인프라 및 교통 상황")

# Folium 지도 생성
m = folium.Map(location=[lat, lng], zoom_start=15)
folium.Marker([lat, lng], popup="선택한 대여소", icon=folium.Icon(color='black', icon='info-sign')).add_to(m)
folium.Circle([lat, lng], radius=1000, color='blue', fill=True, fill_opacity=0.1).add_to(m)

for s in schools:
    folium.Marker(location=[float(s['y']), float(s['x'])], popup=s['place_name'], icon=folium.Icon(color='orange', icon='graduation-cap', prefix='fa')).add_to(m)
for sw in subways:
    folium.Marker(location=[float(sw['y']), float(sw['x'])], popup=sw['place_name'], icon=folium.Icon(color='blue', icon='subway', prefix='fa')).add_to(m)

# 🗺️ 지도 위에 '교통 상황 도로선' 그리기 (시뮬레이션)
traffic_colors = {"원활 (초록)": "green", "서행 (노랑)": "orange", "정체 (빨강)": "red"}
t_color = traffic_colors[traffic_condition]

# 대여소 주변을 지나는 2개의 가상 주요 도로망 그리기
road1_start = [lat - 0.005, lng - 0.005]
road1_end = [lat + 0.005, lng + 0.005]
road2_start = [lat + 0.005, lng - 0.005]
road2_end = [lat - 0.005, lng + 0.005]

folium.PolyLine([road1_start, road1_end], color=t_color, weight=8, opacity=0.7, tooltip=f"현재 도로 상황: {traffic_condition}").add_to(m)
folium.PolyLine([road2_start, road2_end], color=t_color, weight=8, opacity=0.7, tooltip=f"현재 도로 상황: {traffic_condition}").add_to(m)

# 지도 출력
st_folium(m, width=1100, height=400)

st.markdown("---")

# --- 하단: 예측 결과 및 차트 ---
col1, col2 = st.columns(2)

base_demand = loc_base_demand + (school_count * 2) + (subway_count * 4) 
if "평일" in day_type:
    if current_hour in [8, 9, 18, 19]: base_demand *= 2.2 
    elif current_hour < 6: base_demand *= 0.2
else:
    if 14 <= current_hour <= 18: base_demand *= 1.8 
    elif current_hour < 8: base_demand *= 0.3

if weather_condition == "비/눈": base_demand *= 0.15 
elif weather_condition == "미세먼지 나쁨": base_demand *= 0.7 

# 차가 막히면 오히려 단거리 자전거 수요가 약간 증가하는 로직 추가!
if traffic_condition == "정체 (빨강)":
    base_demand *= 1.15 

predicted_demand = int(base_demand)

with col1:
    st.subheader("📊 실시간 예측 결과")
    st.markdown(f"### 예상 수요량: **{predicted_demand}대**")
    
    if traffic_condition == "정체 (빨강)":
        st.info("💡 (차량 정체로 인해 단거리 대체 이동 수단인 자전거 수요가 15% 상승했습니다)")

    if predicted_demand > 100:
        st.error("🚨 매우 혼잡 (자전거 재배치 트럭 출동 요망)")
    elif predicted_demand > 50:
        st.warning("🟡 보통 (주의 관찰)")
    else:
        st.success("🟢 널널 (대여소 여유)")

with col2:
    st.subheader("🔍 수집된 데이터 요약")
    st.write(f"- 인근 학교: **{school_count}개** (주황색 마커)")
    st.write(f"- 인근 지하철역: **{subway_count}개** (파란색 마커)")
    if sw_status != "성공": st.warning(f"API 상태: {sw_status}")

st.markdown("---")
st.subheader(f"📈 {day_type.split()[0]} 24시간 예상 수요 패턴")

hours = np.arange(24)
demands = np.random.randint(10, 30, size=24) + int(loc_base_demand * 0.3)
if "평일" in day_type:
    demands[8:10] += int(loc_base_demand * 1.5) + (subway_count * 3)
    demands[18:20] += int(loc_base_demand * 1.8) + (subway_count * 3)
else:
    demands[14:19] += int(loc_base_demand * 1.2) + (school_count * 2)

chart_data = pd.DataFrame({'시간': hours, '수요량': demands})
st.bar_chart(chart_data.set_index('시간'))