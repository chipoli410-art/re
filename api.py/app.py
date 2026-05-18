import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime

# --------------------------------------------------------
# 1. 외부 API 통신 함수 정의
# --------------------------------------------------------
# 카카오 로컬 API를 통해 대여소 반경 1km 내 인프라(학교, 지하철역) 데이터를 가져오는 함수이다.
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
        return [], f"에러코드 {response.status_code}"
    except Exception as e:
        return [], f"요청 실패: {str(e)}"

# 🌟 [수정] Open-Meteo API: 사용자가 선택한 특정 날짜의 날씨 및 미세먼지 예보를 자동 조회하는 함수이다.
@st.cache_data(ttl=600)
def get_weather_by_date(lat, lng, date_str):
    try:
        # 기상청 날씨 예보 API를 호출하여 해당 날짜의 날씨 코드와 최고 기온을 가져온다.
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&start_date={date_str}&end_date={date_str}&daily=weather_code,temperature_2m_max&timezone=Asia/Seoul"
        w_res = requests.get(weather_url).json()
        
        if 'daily' in w_res and w_res['daily']['weather_code']:
            code = w_res['daily']['weather_code'][0]
            temp = w_res['daily']['temperature_2m_max'][0]
        else:
            return "맑음", 20.0, 30.0, "예보 범위 외 날짜(기본값 적용)"
        
        # 대기질 API를 호출하여 해당 날짜의 최대 미세먼지(PM10) 수치를 가져온다.
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lng}&start_date={date_str}&end_date={date_str}&daily=pm10_max&timezone=Asia/Seoul"
        a_res = requests.get(aqi_url).json()
        
        pm10 = 30.0
        if 'daily' in a_res and a_res['daily']['pm10_max'] and a_res['daily']['pm10_max'][0] is not None:
            pm10 = a_res['daily']['pm10_max'][0]
        
        # WMO 기상 코드 기준 50 이상은 강수(비/눈) 상태로 판별한다.
        if code >= 50:
            condition = "비/눈"
        elif pm10 > 80:
            condition = "미세먼지 나쁨"
        else:
            condition = "맑음"
            
        return condition, temp, pm10, "성공"
    except:
        return "맑음", 20.0, 30.0, "날씨 API 조회 실패(기본값 적용)"


# --------------------------------------------------------
# 2. 대시보드 화면 기본 설정
# --------------------------------------------------------
st.set_page_config(page_title="따릉이 수요 예측 대시보드", layout="wide")
st.title("🚲 따릉이 실시간 인프라 분석 & 수요 예측")
kakao_api_key = "09611d17ff9500ed2d94a6d607cf3609"


# --------------------------------------------------------
# 3. 사이드바 (사용자 입력 및 자동 데이터 연산)
# --------------------------------------------------------
st.sidebar.header("🌍 환경 설정 (위치 및 시간)")
location_coords = {
    "강남역 (오피스/환승)": {"coords": (37.4979, 127.0276), "base": 60},
    "여의도역 (오피스/공원)": {"coords": (37.5215, 126.9246), "base": 55},
    "홍대입구역 (대학가/유흥)": {"coords": (37.5568, 126.9245), "base": 65},
    "서울숲역 (여가/공원)": {"coords": (37.5443, 127.0440), "base": 45},
    "노원역 (주거/학원가)": {"coords": (37.6542, 127.0568), "base": 50},
    "잠실역 (쇼핑/테마파크)": {"coords": (37.5133, 127.1001), "base": 65},
    "신도림역 (대형 환승거점)": {"coords": (37.5088, 126.8912), "base": 55},
    "혜화역 (대학로/문화)": {"coords": (37.5823, 127.0019), "base": 50},
    "왕십리역 (다중 환승/대학가)": {"coords": (37.5611, 127.0385), "base": 50},
    "용산역 (KTX/쇼핑)": {"coords": (37.5299, 126.9646), "base": 55},
    "사당역 (경기 남부 환승)": {"coords": (37.4765, 126.9816), "base": 60},
    "건대입구역 (대학가/유흥)": {"coords": (37.5404, 127.0692), "base": 60},
    "신촌역 (대학가)": {"coords": (37.5552, 126.9368), "base": 55}
}

location = st.sidebar.selectbox("📍 지역 선택", list(location_coords.keys()))

# 🌟 [NEW] 날짜 선택 컴포넌트 추가
selected_date = st.sidebar.date_input("📅 날짜 선택", datetime.now().date())
current_hour = st.sidebar.slider("⏰ 시간대", 0, 23, 18)

# 🌟 [NEW] 선택된 날짜의 요일을 분석하여 평일과 주말을 자동으로 구분한다.
# weekday() 결과가 5(토요일), 6(일요일)이면 주말이며 나머지는 평일이다.
if selected_date.weekday() >= 5:
    day_type = "주말/공휴일 (여가 위주)"
else:
    day_type = "평일 (출퇴근 위주)"

st.sidebar.text(f"분석 유형: {day_type}")

lat, lng = location_coords[location]["coords"]
loc_base_demand = location_coords[location]["base"]

# 🌟 [수정] 기상 설정 섹션: 선택된 날짜 데이터를 날씨 API 연동에 매개변수로 전달한다.
st.sidebar.markdown("---")
st.sidebar.header("🌤️ 기상 설정")
weather_mode = st.sidebar.radio("날씨 연동 모드", ["선택 날짜 날씨 자동 연동 🟢", "수동 시뮬레이션 설정 🔴"])

date_str = selected_date.strftime("%Y-%m-%d")

if "선택 날짜" in weather_mode:
    auto_condition, current_temp, current_pm10, w_status = get_weather_by_date(lat, lng, date_str)
    weather_condition = auto_condition
    st.sidebar.success(f"예보 기온: {current_temp}°C\n\n미세먼지: {current_pm10}µg/m³\n\n상태: {weather_condition}\n\n({w_status})")
else:
    weather_condition = st.sidebar.selectbox("수동 기상 상태 선택", ["맑음", "비/눈", "미세먼지 나쁨"])


# --------------------------------------------------------
# 4. 실시간 변수 계산 및 데이터 수집
# --------------------------------------------------------
if "평일" in day_type:
    if current_hour in [8, 9, 18, 19]: traffic_condition = "정체 (빨강)"
    elif 10 <= current_hour <= 17: traffic_condition = "서행 (노랑)"
    else: traffic_condition = "원활 (초록)"
else: 
    if 14 <= current_hour <= 19: traffic_condition = "서행 (노랑)"
    else: traffic_condition = "원활 (초록)"

schools, _ = get_nearby_poi_data(lat, lng, kakao_api_key, "SC4")
subways, _ = get_nearby_poi_data(lat, lng, kakao_api_key, "SW8")
school_count, subway_count = len(schools), len(subways)


# --------------------------------------------------------
# 5. 상단 UI: 교통 알림 배너 및 지도 시각화
# --------------------------------------------------------
if traffic_condition == "정체 (빨강)":
    st.error(f"🚨 **[교통 혼잡 알림]** {current_hour}시 현재, 대여소 주변 주요 도로가 매우 혼잡합니다.")
elif traffic_condition == "서행 (노랑)":
    st.warning(f"⚠️ **[교통 서행]** {current_hour}시 주변 도로에 차량이 많습니다.")

st.subheader(f"🗺️ {location.split()[0]} 주변 인프라 및 교통 상황 (반경 1km)")

m = folium.Map(location=[lat, lng], zoom_start=15)
folium.Marker([lat, lng], popup="선택한 대여소", icon=folium.Icon(color='black', icon='info-sign')).add_to(m)

traffic_colors = {"원활 (초록)": "green", "서행 (노랑)": "orange", "정체 (빨강)": "red"}
t_color = traffic_colors[traffic_condition]

folium.Circle(
    [lat, lng],
    radius=1000,
    color=t_color,
    fill=True,
    fill_color=t_color,
    fill_opacity=0.2,
).add_to(m)

for s in schools:
    folium.Marker(location=[float(s['y']), float(s['x'])], popup=s['place_name'], icon=folium.Icon(color='orange', icon='graduation-cap', prefix='fa')).add_to(m)
for sw in subways:
    folium.Marker(location=[float(sw['y']), float(sw['x'])], popup=sw['place_name'], icon=folium.Icon(color='blue', icon='subway', prefix='fa')).add_to(m)

st_folium(m, width=1100, height=400)
st.markdown("---")


# --------------------------------------------------------
# 6. 수요 예측 알고리즘 적용
# --------------------------------------------------------
base_demand = loc_base_demand + (school_count * 2) + (subway_count * 4) 
if "평일" in day_type:
    if current_hour in [8, 9, 18, 19]: base_demand *= 2.2 
    elif current_hour < 6: base_demand *= 0.2
else:
    if 14 <= current_hour <= 18: base_demand *= 1.8 
    elif current_hour < 8: base_demand *= 0.3

if weather_condition == "비/눈": base_demand *= 0.15 
elif weather_condition == "미세먼지 나쁨": base_demand *= 0.7 
if traffic_condition == "정체 (빨강)": base_demand *= 1.15 

predicted_demand = int(base_demand)


# --------------------------------------------------------
# 7. 하단 UI: 예측 결과 및 데이터 요약 출력
# --------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 예측 결과 (교통/날씨/인프라 반영)")
    st.markdown(f"### 예상 수요량: **{predicted_demand}대**")
    if traffic_condition == "정체 (빨강)": st.info("💡 (차량 정체로 인해 자전거 수요가 15% 상승했습니다)")

    if predicted_demand > 100: st.error("🚨 매우 혼잡 (자전거 재배치 필요)")
    elif predicted_demand > 50: st.warning("🟡 보통")
    else: st.success("🟢 널널")

with col2:
    st.subheader("🔍 수집된 실시간 데이터 요약")
    st.write(f"- 조회 기준 날짜: **{date_str}**")
    st.write(f"- 인근 학교: **{school_count}개** (카카오 API)")
    st.write(f"- 인근 지하철역: **{subway_count}개** (카카오 API)")
    
    if "선택 날짜" in weather_mode:
        st.write(f"- 기상 상태: **{weather_condition} ({current_temp}°C, Open-Meteo 예보 데이터 반영)**")
    else:
        st.write(f"- 기상 상태: **{weather_condition} (수동 시뮬레이션 설정값 반영)**")

st.markdown("---")


# --------------------------------------------------------
# 8. 24시간 예상 수요 패턴 차트
# --------------------------------------------------------
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