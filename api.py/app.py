import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium

# --------------------------------------------------------
# 1. 외부 API 통신 함수 정의
# --------------------------------------------------------
# 카카오 로컬 API를 사용하여 특정 좌표 반경 내의 장소(POI) 목록을 가져오는 함수이다.
# category_code: SC4(학교), SW8(지하철역) 등 카카오 API 지정 코드 사용
def get_nearby_poi_data(lat, lng, api_key, category_code):
    if not api_key:
        return [], "키 미입력"
    
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    # 반경 1000m(1km) 이내, 최대 15개의 장소 데이터를 요청한다.
    params = {"category_group_code": category_code, "y": lat, "x": lng, "radius": 1000, "size": 15}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()['documents'], "성공"
        return [], f"에러코드 {response.status_code}"
    except Exception as e:
        return [], f"요청 실패: {str(e)}"


# --------------------------------------------------------
# 2. 대시보드 화면 기본 설정
# --------------------------------------------------------
# 웹 페이지의 탭 이름과 레이아웃 넓이를 지정한다.
st.set_page_config(page_title="따릉이 수요 예측 대시보드", layout="wide")
st.title("🚲 따릉이 실시간 인프라 분석 & 수요 예측")

# 카카오 REST API 키 (고정값)
kakao_api_key = "09611d17ff9500ed2d94a6d607cf3609"


# --------------------------------------------------------
# 3. 사이드바 (사용자 입력 환경 설정)
# --------------------------------------------------------
st.sidebar.header("🌍 시뮬레이션 환경 설정")

# 서울 주요 대여소의 위도/경도 및 기본 수요량(base) 데이터 딕셔너리이다.
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

# 사용자로부터 지역, 요일, 날씨, 시간대 정보를 입력받는다.
location = st.sidebar.selectbox("📍 지역 선택", list(location_coords.keys()))
day_type = st.sidebar.radio("📅 요일 유형", ["평일 (출퇴근 위주)", "주말/공휴일 (여가 위주)"])
weather_condition = st.sidebar.selectbox("☔ 기상 상태", ["맑음", "비/눈", "미세먼지 나쁨"])
current_hour = st.sidebar.slider("⏰ 시간대", 0, 23, 18)


# --------------------------------------------------------
# 4. 실시간 변수 계산 및 데이터 수집
# --------------------------------------------------------
# 시간대와 요일을 바탕으로 가상의 교통 상황(원활/서행/정체)을 자동 계산한다.
if "평일" in day_type:
    if current_hour in [8, 9, 18, 19]: traffic_condition = "정체 (빨강)"
    elif 10 <= current_hour <= 17: traffic_condition = "서행 (노랑)"
    else: traffic_condition = "원활 (초록)"
else: 
    if 14 <= current_hour <= 19: traffic_condition = "서행 (노랑)"
    else: traffic_condition = "원활 (초록)"

# 선택된 지역의 좌표 및 기본 수요량을 변수에 할당한다.
lat, lng = location_coords[location]["coords"]
loc_base_demand = location_coords[location]["base"]

# 카카오 API를 호출하여 주변 학교와 지하철역 데이터를 수집한다.
schools, _ = get_nearby_poi_data(lat, lng, kakao_api_key, "SC4")
subways, _ = get_nearby_poi_data(lat, lng, kakao_api_key, "SW8")
school_count, subway_count = len(schools), len(subways)


# --------------------------------------------------------
# 5. 상단 UI: 교통 알림 배너 및 지도 시각화
# --------------------------------------------------------
# 교통 상황이 좋지 않을 경우 경고 메시지를 상단에 출력한다.
if traffic_condition == "정체 (빨강)":
    st.error(f"🚨 **[교통 혼잡 알림]** {current_hour}시 현재, 대여소 주변 주요 도로가 매우 혼잡합니다.")
elif traffic_condition == "서행 (노랑)":
    st.warning(f"⚠️ **[교통 서행]** {current_hour}시 주변 도로에 차량이 많습니다.")

st.subheader(f"🗺️ {location.split()[0]} 주변 인프라 및 교통 상황 (반경 1km)")

# Folium 지도 객체를 생성하고 중심점에 마커를 찍는다.
m = folium.Map(location=[lat, lng], zoom_start=15)
folium.Marker([lat, lng], popup="선택한 대여소", icon=folium.Icon(color='black', icon='info-sign')).add_to(m)

# 교통 상황에 따른 반경 1km 원형 오버레이(색상 변경)를 지도에 그린다.
traffic_colors = {"원활 (초록)": "green", "서행 (노랑)": "orange", "정체 (빨강)": "red"}
t_color = traffic_colors[traffic_condition]

folium.Circle(
    [lat, lng],
    radius=1000,
    color=t_color,
    fill=True,
    fill_color=t_color,
    fill_opacity=0.2,
    tooltip=f"예상 교통 상황: {traffic_condition}"
).add_to(m)

# 수집된 학교(주황색)와 지하철역(파란색) 위치에 각각 마커를 추가한다.
for s in schools:
    folium.Marker(location=[float(s['y']), float(s['x'])], popup=s['place_name'], icon=folium.Icon(color='orange', icon='graduation-cap', prefix='fa')).add_to(m)
for sw in subways:
    folium.Marker(location=[float(sw['y']), float(sw['x'])], popup=sw['place_name'], icon=folium.Icon(color='blue', icon='subway', prefix='fa')).add_to(m)

# 완성된 지도를 Streamlit 화면에 렌더링한다.
st_folium(m, width=1100, height=400)

st.markdown("---")


# --------------------------------------------------------
# 6. 수요 예측 알고리즘 적용
# --------------------------------------------------------
# 인프라 점수(학교, 지하철역 가중치)를 더해 기본 수요를 계산한다.
base_demand = loc_base_demand + (school_count * 2) + (subway_count * 4) 

# 요일 및 시간대에 따른 수요량 증감률을 곱한다.
if "평일" in day_type:
    if current_hour in [8, 9, 18, 19]: base_demand *= 2.2 
    elif current_hour < 6: base_demand *= 0.2
else:
    if 14 <= current_hour <= 18: base_demand *= 1.8 
    elif current_hour < 8: base_demand *= 0.3

# 날씨 상황에 따른 패널티를 적용한다.
if weather_condition == "비/눈": base_demand *= 0.15 
elif weather_condition == "미세먼지 나쁨": base_demand *= 0.7 

# 교통 정체 시 자전거 대체 수요가 증가하는 로직을 반영한다.
if traffic_condition == "정체 (빨강)": base_demand *= 1.15 

# 최종 예상 수요량을 정수로 변환한다.
predicted_demand = int(base_demand)


# --------------------------------------------------------
# 7. 하단 UI: 예측 결과 및 데이터 요약 출력
# --------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 예측 결과 (교통/날씨/인프라 반영)")
    st.markdown(f"### 예상 수요량: **{predicted_demand}대**")
    
    if traffic_condition == "정체 (빨강)": 
        st.info("💡 (차량 정체로 인해 자전거 수요가 15% 상승했습니다)")

    # 혼잡도 기준에 따라 알맞은 상태 메시지를 출력한다.
    if predicted_demand > 100: st.error("🚨 매우 혼잡 (자전거 재배치 필요)")
    elif predicted_demand > 50: st.warning("🟡 보통")
    else: st.success("🟢 널널")

with col2:
    st.subheader("🔍 수집된 데이터 요약")
    st.write(f"- 인근 학교: **{school_count}개** (카카오 API)")
    st.write(f"- 인근 지하철역: **{subway_count}개** (카카오 API)")
    st.write(f"- 예상 교통 상황: **{traffic_condition}**")

st.markdown("---")


# --------------------------------------------------------
# 8. 24시간 예상 수요 패턴 차트
# --------------------------------------------------------
st.subheader(f"📈 {day_type.split()[0]} 24시간 예상 수요 패턴")

# 0~23시까지의 시간 배열과 기본 난수 수요량을 생성한다.
hours = np.arange(24)
demands = np.random.randint(10, 30, size=24) + int(loc_base_demand * 0.3)

# 평일/주말 특성에 맞춰 특정 시간대의 수요 데이터를 인위적으로 증폭시킨다.
if "평일" in day_type:
    demands[8:10] += int(loc_base_demand * 1.5) + (subway_count * 3) # 출근 시간
    demands[18:20] += int(loc_base_demand * 1.8) + (subway_count * 3) # 퇴근 시간
else:
    demands[14:19] += int(loc_base_demand * 1.2) + (school_count * 2) # 주말 오후 시간

# DataFrame으로 변환 후 Streamlit의 막대 그래프로 시각화한다.
chart_data = pd.DataFrame({'시간': hours, '수요량': demands})
st.bar_chart(chart_data.set_index('시간'))