import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium

# API에서 장소 리스트(이름, 좌표)를 가져오는 함수
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

# API 설정 (사용자 키 강제 고정)
kakao_api_key = "09611d17ff9500ed2d94a6d607cf3609"

# 사이드바 설정
st.sidebar.header("🌍 시뮬레이션 환경 설정")
location_coords = {
    "강남역 (오피스/환승)": {"coords": (37.4979, 127.0276), "base": 60},
    "여의도역 (오피스/공원)": {"coords": (37.5215, 126.9246), "base": 55},
    "홍대입구역 (대학가/유흥)": {"coords": (37.5568, 126.9245), "base": 65},
    "서울숲역 (여가/공원)": {"coords": (37.5443, 127.0440), "base": 45},
    "노원역 (주거/학원가)": {"coords": (37.6542, 127.0568), "base": 50},
    "잠실역 (쇼핑/테마파크)": {"coords": (37.5133, 127.1001), "base": 65},
    "건대입구역 (대학가/유흥)": {"coords": (37.5404, 127.0692), "base": 60}
}

location = st.sidebar.selectbox("📍 지역 선택", list(location_coords.keys()))
day_type = st.sidebar.radio("📅 요일 유형", ["평일 (출퇴근 위주)", "주말/공휴일 (여가 위주)"])
weather_condition = st.sidebar.selectbox("☔ 기상 상태", ["맑음", "비/눈", "미세먼지 나쁨"])
current_hour = st.sidebar.slider("⏰ 시간대", 0, 23, 18)

# 🚦 자동 교통 상황 로직
if "평일" in day_type:
    if current_hour in [8, 9, 18, 19]:
        traffic_condition = "정체 (빨강)"
    elif 10 <= current_hour <= 17:
        traffic_condition = "서행 (노랑)"
    else:
        traffic_condition = "원활 (초록)"
else: # 주말
    if 14 <= current_hour <= 19:
        traffic_condition = "서행 (노랑)"
    else:
        traffic_condition = "원활 (초록)"

lat, lng = location_coords[location]["coords"]
loc_base_demand = location_coords[location]["base"]

# 실시간 데이터 수집
schools, s_status = get_nearby_poi_data(lat, lng, kakao_api_key, "SC4")
subways, sw_status = get_nearby_poi_data(lat, lng, kakao_api_key, "SW8")
school_count = len(schools)
subway_count = len(subways)

# 상단 알림 배너
if traffic_condition == "정체 (빨강)":
    st.error(f"🚨 **[교통 혼잡 알림]** {current_hour}시 현재, 대여소 주변 도로가 매우 혼잡할 것으로 예상됩니다.")
elif traffic_condition == "서행 (노랑)":
    st.warning(f"⚠️ **[교통 서행]** {current_hour}시 주변 도로에 차량이 많습니다.")

st.subheader(f"🗺️ {location.split()[0]} 주변 인프라 및 예상 교통 상황 (반경 1km)")

# Folium 지도 생성
m = folium.Map(location=[lat, lng], zoom_start=15)
folium.Marker([lat, lng], popup="선택한 대여소", icon=folium.Icon(color='black', icon='info-sign')).add_to(m)
folium.Circle([lat, lng], radius=1000, color='blue', fill=True, fill_opacity=0.05).add_to(m)

for s in schools:
    folium.Marker(location=[float(s['y']), float(s['x'])], popup=s['place_name'], icon=folium.Icon(color='orange', icon='graduation-cap', prefix='fa')).add_to(m)
for sw in subways:
    folium.Marker(location=[float(sw['y']), float(sw['x'])], popup=sw['place_name'], icon=folium.Icon(color='blue', icon='subway', prefix='fa')).add_to(m)

# 🗺️ 도로 시각화 업그레이드: 'X'자 대신 '격자(#) 도로망' 시뮬레이션
traffic_colors = {"원활 (초록)": "green", "서행 (노랑)": "orange", "정체 (빨강)": "red"}
t_color = traffic_colors[traffic_condition]

# 격자무늬 도로 좌표 계산용 오프셋 (약 300m 간격)
offset = 0.003 

# 4개의 간선 도로 정의 (# 모양)
grid_roads = [
    # 가로 도로 (East-West)
    [[lat + offset, lng - offset * 1.5], [lat + offset, lng + offset * 1.5]], # 북쪽 도로
    [[lat - offset, lng - offset * 1.5], [lat - offset, lng + offset * 1.5]], # 남쪽 도로
    # 세로 도로 (North-South)
    [[lat + offset * 1.5, lng - offset], [lat - offset * 1.5, lng - offset]], # 서쪽 도로
    [[lat + offset * 1.5, lng + offset], [lat - offset * 1.5, lng + offset]]  # 동쪽 도로
]

# 지도에 도로망 그리기
for road_coords in grid_roads:
    folium.PolyLine(
        road_coords,
        color=t_color,
        weight=10, # 실제 도로처럼 굵게
        opacity=0.7,
        tooltip=f"예상 도로 상황: {traffic_condition}"
    ).add_to(m)

# 지도 출력
st_folium(m, width=1100, height=400)

st.markdown("---")

# 하단 예측 결과 및 차트
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

if traffic_condition == "정체 (빨강)":
    base_demand *= 1.15 

predicted_demand = int(base_demand)

with col1:
    st.subheader("📊 예측 결과 (교통/날씨/인프라 반영)")
    st.markdown(f"### 예상 수요량: **{predicted_demand}대**")
    
    if traffic_condition == "정체 (빨강)":
        st.info("💡 (차량 정체로 인해 자전거 수요가 15% 상승했습니다)")

    if predicted_demand > 100:
        st.error("🚨 매우 혼잡 (자전거 재배치 필요)")
    elif predicted_demand > 50:
        st.warning("🟡 보통")
    else:
        st.success("🟢 널널")

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