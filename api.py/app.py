import streamlit as st
import pandas as pd
import numpy as np
import requests

# API 통신 에러 확인 기능과 5km(5000m) 반경 확장이 적용된 함수
def get_nearby_poi_count(lat, lng, api_key, category_code):
    if not api_key:
        return 0, "키 미입력"
    
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    # 반경을 1000m(1km)에서 5000m(5km)로 대폭 확장!
    params = {"category_group_code": category_code, "y": lat, "x": lng, "radius": 5000}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()['meta']['total_count'], "성공"
        else:
            return 0, f"에러코드 {response.status_code}" # 키가 틀렸거나 권한이 없으면 여기서 에러 코드를 뱉음
    except Exception as e:
        return 0, f"요청 실패: {str(e)}"

st.set_page_config(page_title="따릉이 수요 예측 대시보드", layout="wide")
st.title("🚲 따릉이 다차원 수요 예측 대시보드")

st.sidebar.header("🔑 API 설정")
kakao_api_key = st.sidebar.text_input("카카오 REST API 키", type="password")

st.sidebar.header("🌍 시뮬레이션 환경 설정")

# 지역별 기본 유동 인구 가중치 (API가 0이더라도 혼잡도가 제대로 나오도록 기본값을 대폭 상향!)
location_coords = {
    "강남역 (오피스/환승)": {"coords": (37.4979, 127.0276), "base": 60},
    "여의도역 (오피스/공원)": {"coords": (37.5215, 126.9246), "base": 55},
    "홍대입구역 (대학가/유흥)": {"coords": (37.5568, 126.9245), "base": 65},
    "서울숲 (여가/공원)": {"coords": (37.5443, 127.0440), "base": 45},
    "노원역 (주거/학원가)": {"coords": (37.6542, 127.0568), "base": 50}
}

location = st.sidebar.selectbox("📍 지역 선택", list(location_coords.keys()))
day_type = st.sidebar.radio("📅 요일 유형", ["평일 (출퇴근 위주)", "주말/공휴일 (여가 위주)"])
weather_condition = st.sidebar.selectbox("☔ 기상 상태", ["맑음", "비/눈", "미세먼지 나쁨"])
current_hour = st.sidebar.slider("⏰ 시간대", 0, 23, 18)

lat, lng = location_coords[location]["coords"]
loc_base_demand = location_coords[location]["base"]

# API 호출 결과와 상태 메시지를 동시에 받음
school_count, school_status = get_nearby_poi_count(lat, lng, kakao_api_key, "SC4")
subway_count, subway_status = get_nearby_poi_count(lat, lng, kakao_api_key, "SW8")

# 🧠 수정된 수요 예측 로직 (이제 어느 시간대든 '여유'만 뜨지 않음!)
# 지역 기본 수요 + 학교(개당 2) + 지하철(개당 4)
base_demand = loc_base_demand + (school_count * 2) + (subway_count * 4) 

if "평일" in day_type:
    if current_hour in [8, 9, 18, 19]:
        base_demand *= 2.2 # 출퇴근 시간에 2배 이상 뻥튀기 (강남역이면 150대 넘게 나옴)
    elif current_hour < 6:
        base_demand *= 0.2
else:
    if 14 <= current_hour <= 18:
        base_demand *= 1.8 
    elif current_hour < 8:
        base_demand *= 0.3

if weather_condition == "비/눈":
    base_demand *= 0.15 
elif weather_condition == "미세먼지 나쁨":
    base_demand *= 0.7 

predicted_demand = int(base_demand)

# 🖥️ 화면 출력부
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"📍 {location.split()[0]} 인프라 (반경 5km)")
    if kakao_api_key and school_status != "성공":
        st.error(f"⚠️ API 연동 실패: {school_status}") # API가 실패하면 빨간색 경고창 띄움
    
    st.write(f"- 🎓 검색된 학교 수: **{school_count}개**")
    st.write(f"- 🚇 검색된 지하철역: **{subway_count}개**")

with col2:
    st.subheader("📊 실시간 예측 결과")
    st.markdown(f"### 예상 수요량: **{predicted_demand}대**")
    # 혼잡도 기준 컷 상향 (숫자가 커졌으므로 기준도 올림)
    if predicted_demand > 100:
        st.error("🚨 매우 혼잡 (자전거 재배치 트럭 출동 요망)")
    elif predicted_demand > 50:
        st.warning("🟡 보통 (주의 관찰)")
    else:
        st.success("🟢 널널 (대여소 여유)")

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