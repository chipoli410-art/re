import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import requests
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import platform
import math

warnings = st.cache_resource(lambda: __import__('warnings'))
warnings().filterwarnings('ignore')

# ==========================================
# 🎨 기본 환경 및 폰트 설정
# ==========================================
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="🚲 따릉이 수요 예측 대시보드",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 3em; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 10px; }
    .sub-header { font-size: 1.5em; color: #555; text-align: center; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# 🔑 API KEY 설정 (발급받으신 인증키를 여기에 입력하세요)
SEOUL_API_KEY = "477a746973746a7333374c61524455"

# ==========================================
# ⚙️ 1. MLOps 및 알고리즘 함수
# ==========================================
@st.cache_resource
def load_ml_system():
    model = joblib.load('ttareungi_model_v1.pkl')
    station_meta = pd.read_csv('station_meta.csv')
    profile_db = pd.read_csv('rolling_profile_9m.csv')
    global_mean = profile_db['과거_평균_대여량'].mean()
    return model, station_meta, profile_db, global_mean

try:
    lgb_model, station_meta, profile_db, global_mean = load_ml_system()
    cat_cols = ['대여소_ID_num', '요일', '주말_여부', '비옴_여부']
except Exception as e:
    st.error(f"필수 파일들을 로드하지 못했습니다. 파이프라인 확인 필요: {e}")
    st.stop()

def calculate_simple_distance(lat1, lon1, lat2, lon2):
    """지구 곡률을 제외한 단순 직선거리 연산 (서울 기준 1도당 가중치 적용)"""
    d_lat = (lat1 - lat2) * 111
    d_lon = (lon1 - lon2) * 88
    return math.sqrt(d_lat**2 + d_lon**2)

@st.cache_data(ttl=60)
def get_realtime_bike_status():
    """서울시 실시간 따릉이 거치 상태 API 호출 (전체 대여소 로드)"""
    all_rows = []
    for start_idx in [1, 1001, 2001, 3001]:
        url = f"http://openapi.seoul.go.kr:8088/{SEOUL_API_KEY}/json/bikeList/{start_idx}/{start_idx+999}/"
        try:
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                data = res.json()
                if 'rentBikeStatus' in data and 'row' in data['rentBikeStatus']:
                    all_rows.extend(data['rentBikeStatus']['row'])
        except:
            continue

    if not all_rows:
        return None, "API 응답 실패 또는 데이터가 없습니다."

    df = pd.DataFrame(all_rows)
    df['API_대여소명'] = df['stationName'] # 💡 원본 이름 보존
    df['대여소명'] = df['stationName'].apply(lambda x: x.split('.')[-1].strip() if '.' in str(x) else str(x).strip())
    df['실시간_자전거_수'] = df['parkingBikeTotCnt'].astype(int)
    df['위도'] = df['stationLatitude'].astype(float)
    df['경도'] = df['stationLongitude'].astype(float)
    
    return df, "성공"

# 💡 [신규 추가] 오픈 기상 API 연동 함수
@st.cache_data(ttl=600)
def get_seoul_weather():
    """Open-Meteo API를 사용해 서울의 현재 기상 정보를 가져옵니다."""
    url = "https://api.open-meteo.com/v1/forecast?latitude=37.5665&longitude=126.9780&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&timezone=Asia%2FTokyo"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()['current']
            return {
                '기온': float(data['temperature_2m']),
                '습도': float(data['relative_humidity_2m']),
                '강수량': float(data['precipitation']),
                '풍속': float(data['wind_speed_10m'])
            }, "성공"
        return None, f"API 오류"
    except Exception:
        return None, f"연결 실패"

@st.cache_data
def load_and_adapt_data():
    try:
        df = pd.read_csv('total_clean_data.csv')
        df['date'] = pd.to_datetime(df['대여일자'].astype(str), format='%Y%m%d') + pd.to_timedelta(df['대여시간(시)'], unit='h')
        df.rename(columns={'총_대여건수(Y)': 'demand', '기온': 'temperature', '강수량': 'rainfall', '습도': 'humidity', '대여시간(시)': 'hour'}, inplace=True)
        df['day_of_week'] = df['date'].dt.dayofweek
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_weather_by_date(lat, lng, date_str, target_hour):
    return 22.0, 0.0, 2.0, 60.0, "성공"

# ==========================================
# 🖥️ 2. 메인 화면 및 라우팅
# ==========================================
def main():
    from streamlit_option_menu import option_menu 
    data = load_and_adapt_data()

    with st.sidebar:
        # 타이틀에서 AI 제거
        st.markdown("<h2 style='text-align: center; color: #78c2ff;'>🚲 따릉이 수요예측 시스템</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #FFFFFF; font-size: 14px; opacity: 0.8;'>수요 예측 대시보드</p>", unsafe_allow_html=True)
        st.divider()

        # 메뉴 옵션에서 AI 제거
        page = option_menu(
            menu_title=None, 
            options=["홈 (레이더)", "EDA 분석 대시보드", "수요 예측 조회", "프로젝트 정보"],
            icons=["house-door", "bar-chart-fill", "robot", "info-square"], 
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#78c2ff", "font-size": "18px"},
                "nav-link": {"font-size": "15px", "text-align": "left", "margin": "3px", "--hover-color": "rgba(255, 255, 255, 0.1)", "font-weight": "bold", "color": "#FFFFFF"},
                "nav-link-selected": {"background-color": "#1f77b4", "color": "white", "font-weight": "bold"},
            }
        )

    # 라우팅 명칭 변경
    if page == "홈 (레이더)": show_home(data)
    elif page == "EDA 분석 대시보드": show_eda(data)
    elif page == "수요 예측 조회": show_prediction()
    elif page == "프로젝트 정보": show_project_info()

# ==========================================
# 🏠 3. 홈 화면 (검색창 GPS 매칭 & 명칭 정제 통합 완성본)
# ==========================================
def show_home(data):
    st.markdown('<div class="main-header">🚲 서울시 따릉이 수요예측 대시보드</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    if not data.empty:
        with col1: st.metric("📊 총 누적 데이터", f"{len(data):,.0f} 건")
        with col2: st.metric("🚴 평균 시간당 대여", f"{data['demand'].mean():.1f} 대")
        with col4: st.metric("🗺️ 총 등록 대여소", f"{len(station_meta):,} 개소")
    
    st.divider()
    
    realtime_df, api_status = get_realtime_bike_status()
    if realtime_df is None:
        st.error(f"🚨 **서울시 실시간 API 연동 실패:** {api_status}")
        return 

    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour
    is_weekend = 1 if weekday >= 5 else 0

    if 'selected_station' not in st.session_state:
        st.session_state.selected_station = station_meta['대여소명'].iloc[0]
    if 'temp_selection' not in st.session_state:
        st.session_state.temp_selection = st.session_state.selected_station

    def commit_station():
        st.session_state.selected_station = st.session_state.temp_selection

    st.subheader("🚨 지능형 수요 예측 및 대체 대여소 추천 레이더")
    st.markdown(f"**현재 시간 ({now.strftime('%Y-%m-%d %H:%M')})** 기준 진단판입니다. 검색창에 **대여소 이름을 입력하여 검색**하신 후 확인 버튼을 눌러주세요.")

    # 💡 [통일된 정제 함수] 마침표 기준 뒤쪽 이름 추출
    def clean_station_name(name_str):
        name_str = str(name_str)
        return name_str.split('.')[-1].strip() if '.' in name_str else name_str.strip()

    # 실시간 데이터 넘파이 배열화 (고속 연산용)
    rt_names = realtime_df['대여소명'].values
    rt_api_raw = realtime_df['API_대여소명'].values if 'API_대여소명' in realtime_df.columns else rt_names
    rt_lats = realtime_df['위도'].values
    rt_lngs = realtime_df['경도'].values
    rt_bikes = realtime_df['실시간_자전거_수'].values

    # 1. 텍스트 일치 우선 매핑
    rt_stock_dict = dict(zip(rt_names, rt_bikes))
    rt_clean_name_dict = dict(zip(rt_names, [clean_station_name(x) for x in rt_api_raw]))

    # 2. 검색창 목록을 위해 3000개 대여소 전체 사전 GPS 매핑
    display_name_dict = {}
    for _, row in station_meta.iterrows():
        meta_name = row['대여소명']
        if meta_name in rt_clean_name_dict:
            api_name = rt_clean_name_dict[meta_name]
        else:
            dist_array = np.sqrt(((rt_lats - float(row['위도'])) * 111)**2 + ((rt_lngs - float(row['경도'])) * 88)**2)
            min_idx = np.argmin(dist_array)
            if dist_array[min_idx] <= 0.05:
                api_name = clean_station_name(rt_api_raw[min_idx])
            else:
                api_name = meta_name 
        
        if api_name != meta_name:
            display_name_dict[meta_name] = f"{api_name} ({meta_name})"
        else:
            display_name_dict[meta_name] = meta_name

    station_list = station_meta['대여소명'].tolist()
    try: default_idx = station_list.index(st.session_state.temp_selection)
    except: default_idx = 0

    # 💡 GPS 기반 매핑된 딕셔너리를 드롭다운에 적용
    input_station = st.selectbox(
        "📍 조회할 목적지 검색 (클릭 후 키보드로 텍스트 입력)", 
        station_list, 
        index=default_idx,
        format_func=lambda x: display_name_dict.get(x, x)
    )
    st.session_state.temp_selection = input_station 

    # --- AI 수요 예측 연산부 ---
    target_meta = station_meta[station_meta['대여소명'] == st.session_state.selected_station].iloc[0]
    t_lat = float(target_meta['위도'])
    t_lon = float(target_meta['경도'])
    
    prof = profile_db[(profile_db['대여소_ID_num'] == target_meta['대여소_ID_num']) & (profile_db['요일'] == weekday) & (profile_db['대여시간(시)'] == hour)]
    past_mean = prof['과거_평균_대여량'].values[0] if not prof.empty else global_mean
    
    single_input = pd.DataFrame([{
        '대여소_ID_num': target_meta['대여소_ID_num'], '요일': weekday, '대여시간(시)': hour, '주말_여부': is_weekend,
        '기온': 22.0, '강수량': 0.0, '풍속': 2.0, '습도': 60.0, '비옴_여부': 0,
        '과거_평균_대여량': past_mean, '지하철역_수_1km': target_meta['지하철역_수_1km'], '학교_수_1km': target_meta['학교_수_1km']
    }])
    ordered_cols = ['대여소_ID_num', '요일', '대여시간(시)', '주말_여부', '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', '지하철역_수_1km', '학교_수_1km']
    single_input = single_input[ordered_cols]
    for col in cat_cols: single_input[col] = single_input[col].astype('category')
    
    predicted_demand = int(np.maximum(0, np.round(lgb_model.predict(single_input)[0])))

    # --- 실시간 재고 탐색 ---
    rt_info = realtime_df[realtime_df['대여소명'] == st.session_state.selected_station]
    target_api_name = None
    
    if rt_info.empty:
        realtime_df['거리차이'] = realtime_df.apply(lambda r: calculate_simple_distance(t_lat, t_lon, r['위도'], r['경도']), axis=1)
        closest_match = realtime_df.loc[realtime_df['거리차이'].idxmin()]
        if closest_match['거리차이'] <= 0.05: 
            current_bikes = closest_match['실시간_자전거_수']
            target_api_name = clean_station_name(closest_match.get('API_대여소명', st.session_state.selected_station))
        else: 
            st.warning(f"⚠️ '{st.session_state.selected_station}' 대여소는 현재 정보 미수신 상태입니다.")
            return
    else: 
        current_bikes = rt_info['실시간_자전거_수'].values[0]
        target_api_name = clean_station_name(rt_info.get('API_대여소명', pd.Series([st.session_state.selected_station])).values[0])

    expected_remain = current_bikes - predicted_demand
    best_alt = None
    
    display_target_name = f"{target_api_name} ({st.session_state.selected_station})" if target_api_name and target_api_name != st.session_state.selected_station else st.session_state.selected_station
    
    st.info(f"📋 **현재 진단 중인 대여소:** {display_target_name}")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🚲 현재 거치된 자전거", f"{current_bikes} 대")
    with c2: st.metric("📉 1시간 내 예측 수요", f"{predicted_demand} 대")
    
    if expected_remain <= 1:
        with c3: st.metric("⚠️ 상태 예측", "고갈 위험", delta="- 부족 예상", delta_color="inverse")
        st.error(f"🚨 **고갈 경보:** 부족 현상이 예측됩니다. 아래 지도에서 **초록색 대체 대여소**와 최적 도보 경로를 이용하세요.")
        
        candidates = []
        for _, row in station_meta.iterrows():
            if row['대여소명'] == st.session_state.selected_station: continue
            r_lat = float(row['위도'])
            dist = calculate_simple_distance(t_lat, t_lon, r_lat, float(row['경도']))
            
            if dist <= 1.0: 
                cand_bikes = rt_stock_dict.get(row['대여소명'], None)
                cand_api_name = rt_clean_name_dict.get(row['대여소명'], None)
                
                if cand_bikes is None:
                    dist_array = np.sqrt(((rt_lats - r_lat) * 111)**2 + ((rt_lngs - float(row['경도'])) * 88)**2)
                    min_idx = np.argmin(dist_array)
                    if dist_array[min_idx] <= 0.05: 
                        cand_bikes = rt_bikes[min_idx]
                        cand_api_name = clean_station_name(rt_api_raw[min_idx])
                
                if cand_bikes is not None and cand_bikes >= 5: 
                    display_cand_name = f"{cand_api_name} ({row['대여소명']})" if cand_api_name and cand_api_name != row['대여소명'] else row['대여소명']
                    candidates.append({'이름': row['대여소명'], '표출이름': display_cand_name, '거리_km': dist, '여유분': cand_bikes, '위도': r_lat, '경도': float(row['경도'])})
        
        if candidates:
            candidates = sorted(candidates, key=lambda x: x['거리_km'])
            best_alt = candidates[0]
            st.success(f"✅ **최적 경로 추천:** 목적지 주변 도보 {int(best_alt['거리_km']*15)}분 거리에 {best_alt['여유분']}대 여유가 있는 **'{best_alt['표출이름']}'** 대여소를 권장합니다.")
        else: st.warning("주변 1km 이내에 여유가 있는 대여소가 없습니다.")
    else:
        with c3: st.metric("✅ 상태 예측", "여유 있음", delta=f"+{expected_remain}대 남음 예상", delta_color="normal")
        st.success("😊 **안심하세요!** 수요 분석 결과 자전거가 충분히 남아있을 것으로 예측됩니다.")

    st.divider()

    st.button("🔍 선택 대여소 진단 및 예측 실행", type="primary", on_click=commit_station)
    
    st.subheader(f"🗺️ 대여소 현황 및 레이더 동기화 지도")
    
    m_master = folium.Map(location=[t_lat, t_lon], zoom_start=15, tiles="OpenStreetMap")
    marker_cluster = MarkerCluster().add_to(m_master)
    
    # 💡 [핵심 복구] 지도 마커 생성 전, 전체 대여소에 대해 AI 수요 예측값을 갱신합니다.
    # 예측을 위한 입력 데이터 준비
    batch_inputs = pd.DataFrame({
        '대여소_ID_num': station_meta['대여소_ID_num'], '요일': weekday, '대여시간(시)': hour, '주말_여부': is_weekend,
        '기온': 22.0, '강수량': 0.0, '풍속': 2.0, '습도': 60.0, '비옴_여부': 0,
        '지하철역_수_1km': station_meta['지하철역_수_1km'], '학교_수_1km': station_meta['학교_수_1km']
    })
    # 과거 평균 매칭
    current_prof = profile_db[(profile_db['요일'] == weekday) & (profile_db['대여시간(시)'] == hour)]
    batch_inputs = pd.merge(batch_inputs, current_prof[['대여소_ID_num', '과거_평균_대여량']], on='대여소_ID_num', how='left')
    batch_inputs['과거_평균_대여량'] = batch_inputs['과거_평균_대여량'].fillna(global_mean)
    
    # 컬럼 순서 고정 및 타입 지정
    ordered_cols = ['대여소_ID_num', '요일', '대여시간(시)', '주말_여부', '기온', '강수량', '풍속', '습도', '비옴_여부', '과거_평균_대여량', '지하철역_수_1km', '학교_수_1km']
    batch_inputs = batch_inputs[ordered_cols]
    for col in cat_cols: batch_inputs[col] = batch_inputs[col].astype('category')
        
    # AI 예측 수행 후 station_meta에 반영
    station_meta['현재예측수요'] = np.maximum(0, np.round(lgb_model.predict(batch_inputs))).astype(int)

    # 이제 아래에서 station_meta['현재예측수요']를 안전하게 읽을 수 있습니다.

    for _, row in station_meta.iterrows():
        lat_val = float(row['위도'])
        lng_val = float(row['경도'])
        
        live_bikes = rt_stock_dict.get(row['대여소명'], None)
        api_raw_name = rt_clean_name_dict.get(row['대여소명'], None)
        
        if live_bikes is None:
            dist_array = np.sqrt(((rt_lats - lat_val) * 111)**2 + ((rt_lngs - lng_val) * 88)**2)
            min_idx = np.argmin(dist_array)
            if dist_array[min_idx] <= 0.05: 
                live_bikes = rt_bikes[min_idx]
                api_raw_name = clean_station_name(rt_api_raw[min_idx])
                
        bike_text = f"<span style='color:#2ca02c; font-weight:bold;'>{int(live_bikes)}대</span>" if live_bikes is not None else "<span style='color:gray; font-weight:bold;'>미운영</span>"
        
        # 💡 [핵심 수정 3] 지도 팝업창 타이틀에도 결합된 이름 적용
        display_map_name = f"{api_raw_name} ({row['대여소명']})" if api_raw_name and api_raw_name != row['대여소명'] else row['대여소명']
        
        # 💡 [누락 복구] 각 대여소별 예측 대여량 텍스트 포맷 구성
        predict_text = f"<span style='color:#1f77b4; font-weight:bold;'>{int(row['현재예측수요'])}대</span>"
        
        popup_html = f"""
        <div style='font-family: "Malgun Gothic", sans-serif; width:260px;'>
            <h5 style='color:#1f77b4; margin-bottom:8px; font-size: 14px;'>{display_map_name}</h5>
            <p style='margin:3px 0; font-size:12px;'><b>🚲 실시간 현재 재고:</b> {bike_text}</p>
            <p style='margin:3px 0; font-size:12px;'><b>📈 1시간 내 예측수요:</b> {predict_text}</p> <hr style='margin:5px 0; border:0; border-top:1px solid #eee;'>
            <p style='margin:2px 0; font-size:11px; color:#666;'><b>지하철역(1km):</b> {int(row['지하철역_수_1km'])}개 / <b>학교:</b> {int(row['학교_수_1km'])}개</p>
        </div>
        """
        if row['대여소명'] == st.session_state.selected_station:
            folium.Marker([lat_val, lng_val], popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color='red', icon='info-sign')).add_to(m_master)
        elif best_alt and row['대여소명'] == best_alt['이름']:
            folium.Marker([lat_val, lng_val], popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color='green', icon='bicycle', prefix='fa')).add_to(m_master)
        else:
            folium.Marker([lat_val, lng_val], popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color='blue', icon='bicycle', prefix='fa')).add_to(marker_cluster)

    if best_alt:
        folium.PolyLine([(t_lat, t_lon), (best_alt['위도'], best_alt['경도'])], color="blue", weight=3, opacity=0.8, dash_array='7, 7').add_to(m_master)

    st_folium(m_master, width="100%", height=600, key="master_map", returned_objects=[])
    
# ==========================================
# 📊 4. EDA 분석 페이지 (기존과 동일)
# ==========================================
def show_eda(data):
    if data.empty:
        st.warning("total_clean_data.csv 파일이 없어 시각화 그래프를 표시할 수 없습니다.")
        return
    st.header("📊 탐색적 데이터 분석 (EDA) 종합 리포트")
    st.divider()
    tab1, tab2, tab3 = st.tabs(["⏰ 시간대별 패턴 분석", "📅 요일별 수요 분석", "🌤️ 기온 조건별 분석"])
    with tab1:
        st.subheader("⏰ 시간대별 평균 수요 트렌드")
        hourly_df = data.groupby('hour')['demand'].mean().reset_index()
        hourly_df.columns = ['시간대 (Hour)', '평균 대여건수']
        fig1 = px.bar(hourly_df, x='시간대 (Hour)', y='평균 대여건수', color='평균 대여건수', color_continuous_scale='Blues')
        fig1.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("**📉 평일 vs 주말 시간대별 수요 흐름 비교**")
        weekday_hourly = data[data['day_of_week'] < 5].groupby('hour')['demand'].mean()
        weekend_hourly = data[data['day_of_week'] >= 5].groupby('hour')['demand'].mean()
        chart_data_time = pd.DataFrame({'시간대': range(24), '평일 (출퇴근 패턴)': weekday_hourly.values, '주말 (오후 활동 패턴)': weekend_hourly.values}).melt(id_vars='시간대', var_name='패턴 구분', value_name='평균 대여건수')
        fig2 = px.line(chart_data_time, x='시간대', y='평균 대여건수', color='패턴 구분', color_discrete_map={'평일 (출퇴근 패턴)': '#1f77b4', '주말 (오후 활동 패턴)': '#ff7f0e'}, markers=True)
        fig2.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1), legend_title_text='')
        st.plotly_chart(fig2, use_container_width=True)
    with tab2:
        st.subheader("📅 요일별 평균 수요 분포")
        daily_demand = data.groupby('day_of_week')['demand'].mean().reset_index()
        days = ['월', '화', '수', '목', '금', '토', '일']
        daily_demand['day_of_week'] = daily_demand['day_of_week'].apply(lambda x: days[x])
        daily_demand.columns = ['요일', '평균 대여건수']
        daily_demand['수치 표시'] = daily_demand['평균 대여건수'].apply(lambda x: f"<b>{x:,.1f}대</b>")
        min_demand = daily_demand['평균 대여건수'].min()
        max_demand = daily_demand['평균 대여건수'].max()
        fig3 = px.bar(daily_demand, x='요일', y='평균 대여건수', color='요일', color_discrete_sequence=px.colors.qualitative.Set2, text='수치 표시')
        fig3.update_layout(showlegend=False, yaxis=dict(range=[min_demand * 0.85, max_demand * 1.1]))
        fig3.update_traces(textposition='outside', textfont_size=14, textfont_color='white')
        st.plotly_chart(fig3, use_container_width=True)
    with tab3:
        st.subheader("🌡️ 기온 구간별 수요 민감도 분석")
        temp_bins = [-10, 0, 5, 15, 25, 35, 45]
        temp_labels = ['영하(<0°C)', '추움(0-5°C)', '선선함(5-15°C)', '따뜻함(15-25°C)', '더움(25-35°C)', '폭염(>35°C)']
        data['temp_category'] = pd.cut(data['temperature'], bins=temp_bins, labels=temp_labels)
        temp_demand = data.groupby('temp_category', observed=False)['demand'].mean().reset_index()
        temp_demand.columns = ['기온 구간', '평균 대여건수']
        fig4 = px.bar(temp_demand, x='평균 대여건수', y='기온 구간', color='기온 구간', orientation='h', color_discrete_sequence=px.colors.sequential.Sunset)
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# 🔮 5. AI 수요 예측 & ℹ️ 프로젝트 정보 라우터 (기상 연동 토글 이식)
# ==========================================
def show_prediction():
    st.header("🔮 따릉이 AI 미래 수요 예측 (LightGBM)")
    st.markdown("**최신 9개월의 롤링 트렌드**와 **실제 대여소 위치 좌표**, **기상 시뮬레이션**을 결합하여 정확한 대여 수요를 예측합니다.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌍 대여소 및 일시 선택")
        station_list = station_meta['대여소명'].tolist()
        selected_station = st.selectbox("📍 예측할 대여소를 선택하세요", station_list)
        today_date = datetime.now().date()
        max_date = today_date + timedelta(days=365)
        pred_date = st.date_input("📅 날짜 선택", value=today_date, min_value=datetime(2024,1,1).date(), max_value=max_date)
        pred_hour = st.slider("⏰ 시간대 선택 (0-23시)", 0, 23, 18)
        
    st_info = station_meta[station_meta['대여소명'] == selected_station].iloc[0]
    st_id_num = st_info['대여소_ID_num']
    subway_cnt = st_info['지하철역_수_1km']
    school_cnt = st_info['학교_수_1km']
    lat = float(st_info['위도'] if '위도' in station_meta.columns else st_info['위도'])
    lng = float(st_info['경도'])
    
    with col2:
        st.subheader("🌤️ 기상 조건 설정")
        
        # 💡 [새로 이식된 기능] 기상 실시간 연동 체크박스 토글
        use_realtime_weather = st.checkbox("☑️ 현재 서울 실시간 날씨 연동", value=True)

        if use_realtime_weather:
            weather_data, w_status = get_seoul_weather()
            if weather_data:
                st.success("✅ **자동 호출 완료**")
                def_temp = weather_data['기온']
                def_hum = weather_data['습도']
                def_rain = weather_data['강수량']
                def_wind = weather_data['풍속']
            else:
                st.warning("⚠️ 날씨 호출 실패. 기본값 적용")
                def_temp, def_hum, def_rain, def_wind = 22.0, 60.0, 0.0, 2.0
        else:
            def_temp, def_hum, def_rain, def_wind = 22.0, 60.0, 0.0, 2.0
            
        temperature = st.number_input("기온 (°C)", value=float(def_temp))
        rainfall = st.number_input("강수량 (mm)", value=float(def_rain))
        windspeed = st.number_input("풍속 (m/s)", value=float(def_wind))
        humidity = st.number_input("습도 (%)", value=float(def_hum))
        
    with col3:
        st.subheader("🗺️ 대여소 인프라 정보")
        st.info(f"**{selected_station}**\n\n- 내부 관리 ID: {st_id_num}\n- 실제 위도: {lat:.4f}\n- 실제 경도: {lng:.4f}\n- 주변 1km 지하철역: {subway_cnt}개\n- 주변 1km 학교: {school_cnt}개")
        
    st.divider()
    m = folium.Map(location=[lat, lng], zoom_start=16)
    folium.Marker([lat, lng], popup=selected_station, icon=folium.Icon(color='blue', icon='bicycle', prefix='fa')).add_to(m)
    folium.Circle([lat, lng], radius=1000, color="blue", fill=True, fill_opacity=0.05).add_to(m)
    st_folium(m, width="100%", height=350, returned_objects=[])
    
    st.divider()
    if st.button("🚀수요 예측 실행", use_container_width=True):
        pred_date_dt = pd.Timestamp(pred_date)
        weekday_idx = pred_date_dt.dayofweek
        is_weekend_val = 1 if weekday_idx >= 5 else 0
        prof = profile_db[(profile_db['대여소_ID_num'] == st_id_num) & (profile_db['요일'] == weekday_idx) & (profile_db['대여시간(시)'] == pred_hour)]
        past_mean = prof['과거_평균_대여량'].values[0] if not prof.empty else global_mean
        input_data = pd.DataFrame([{'대여소_ID_num': st_id_num, '요일': weekday_idx, '대여시간(시)': pred_hour, '주말_여부': is_weekend_val, '기온': temperature, '강수량': rainfall, '풍속': windspeed, '습도': humidity, '비옴_여부': 1 if rainfall > 0 else 0, '과거_평균_대여량': past_mean, '지하철역_수_1km': subway_cnt, '학교_수_1km': school_cnt}])
        for col in cat_cols: input_data[col] = input_data[col].astype('category')
        raw_pred = lgb_model.predict(input_data)[0]
        final_prediction = max(0, int(round(raw_pred)))
        st.subheader("📊 AI 예측 결과")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("✨ 예상 대여량", f"{final_prediction} 대")
        with c2: st.metric("🕰️ 과거 동시간대 평균", f"{past_mean:.1f} 대")
        with c3: st.metric("📈 평균 대비 증감량", f"{final_prediction - past_mean:+.1f} 대")

def show_project_info():
    st.header("ℹ️ 프로젝트 정보")
    st.markdown("이 대시보드는 **LightGBM 모델**, **Plotly 데이터 시각화**, **공간 거리 연산 기반 최적화(MLOps)** 파이프라인을 통해 구축되었습니다.")
    team_info = pd.DataFrame({'이름': ['박도영', '최선강', '최연규', '이태윤'], '역할': ['팀장 / 데이터 파이프라인', 'LightGBM 최적화', 'API & MLOps', '웹 대시보드(UI)']})
    st.dataframe(team_info, hide_index=True)

if __name__ == "__main__":
    main()