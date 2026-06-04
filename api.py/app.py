import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import warnings
import requests
import folium
from streamlit_folium import st_folium
import platform

warnings.filterwarnings('ignore')

# 폰트 설정 (강제 적용)
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
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

KAKAO_API_KEY = "09611d17ff9500ed2d94a6d607cf3609"

def get_location_coords_by_keyword(keyword, api_key):
    if not api_key or not keyword:
        return None, None, "검색어 없음"
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": keyword}
    try:
        response = requests.get(url, headers=headers, params=params)
        documents = response.json().get('documents', [])
        if documents:
            full_address = documents[0].get('address_name', '')
            if not full_address.startswith('서울'):
                return None, None, f"서울 외 지역 ({full_address})"
            return float(documents[0]['y']), float(documents[0]['x']), "성공"
        return None, None, "검색 결과 없음"
    except Exception as e:
        return None, None, f"에러: {str(e)}"

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

@st.cache_data(ttl=600)
def get_weather_by_date(lat, lng, date_str):
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        
        # 5일 이상 과거면 기록(Archive) API, 아니면 예보(Forecast) API 자동 선택
        if target_date < today - timedelta(days=5):
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lng}&start_date={date_str}&end_date={date_str}&daily=weather_code,temperature_2m_max,precipitation_sum&timezone=Asia/Seoul"
        else:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&start_date={date_str}&end_date={date_str}&daily=weather_code,temperature_2m_max,precipitation_sum&timezone=Asia/Seoul"
            
        res = requests.get(url).json()
        
        if 'daily' in res and res['daily']['weather_code']:
            temp = res['daily']['temperature_2m_max'][0]
            rain = res['daily']['precipitation_sum'][0]
            if temp is None: temp = 20.0
            if rain is None: rain = 0.0
            return temp, rain, "성공"
            
        return 20.0, 0.0, "범위 외"
    except Exception as e:
        return 20.0, 0.0, f"실패: {e}"

@st.cache_data
def load_data():
    try:
        demand_data = pd.read_csv('tpss_hourly_demand.csv')
        demand_data['date'] = pd.to_datetime(demand_data['date'])
        weather_data = pd.read_csv('bike_weather_merged.csv')
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        merged_data = pd.merge(
            demand_data,
            weather_data,
            on='date',
            how='left'
        )
        return merged_data
    except:
        return create_sample_data()

@st.cache_data
def create_sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='h')
    data = pd.DataFrame({
        'date': dates,
        'demand': np.random.normal(300, 100, len(dates)).clip(0),
        'temperature': 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 365)) + np.random.normal(0, 3, len(dates)),
        'rainfall': np.random.exponential(1, len(dates)) * (np.random.random(len(dates)) > 0.8),
        'humidity': 50 + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 365)) + np.random.normal(0, 5, len(dates))
    })
    return data

def prepare_features(data):
    data_copy = data.copy()
    data_copy['hour'] = data_copy['date'].dt.hour
    data_copy['day_of_week'] = data_copy['date'].dt.dayofweek
    data_copy['month'] = data_copy['date'].dt.month
    data_copy['day'] = data_copy['date'].dt.day
    data_copy['is_rush_hour'] = data_copy['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    data_copy['is_morning'] = (data_copy['hour'] < 12).astype(int)
    data_copy['is_evening'] = (data_copy['hour'] >= 18).astype(int)
    data_copy['is_weekend'] = (data_copy['day_of_week'] >= 5).astype(int)
    data_copy['is_spring'] = data_copy['month'].isin([3, 4, 5]).astype(int)
    data_copy['is_summer'] = data_copy['month'].isin([6, 7, 8]).astype(int)
    data_copy['is_autumn'] = data_copy['month'].isin([9, 10, 11]).astype(int)
    data_copy['is_winter'] = data_copy['month'].isin([12, 1, 2]).astype(int)
    data_copy['is_hot'] = (data_copy['temperature'] > 25).astype(int)
    data_copy['is_cold'] = (data_copy['temperature'] < 5).astype(int)
    data_copy['is_rainy'] = (data_copy['rainfall'] > 0).astype(int)
    return data_copy

@st.cache_resource
def train_model(data):
    data_clean = data.dropna()
    feature_cols = ['hour', 'day_of_week', 'month', 'is_rush_hour', 'is_morning',
                   'is_evening', 'is_weekend', 'is_spring', 'is_summer', 'is_autumn',
                   'is_winter', 'temperature', 'is_rainy', 'humidity']
    X = data_clean[feature_cols]
    y = data_clean['demand']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    return model, scaler, feature_cols

def main():
    data = load_data()
    data = prepare_features(data)
    model, scaler, feature_cols = train_model(data)

    st.sidebar.markdown("# 🎨 네비게이션")
    page = st.sidebar.radio(
        "페이지 선택",
        ["🏠 홈", "📊 EDA 분석", "🔮 수요 예측", "📈 모델 성능", "ℹ️ 프로젝트 정보"]
    )

    if page == "🏠 홈":
        show_home(data, model, scaler, feature_cols)
    elif page == "📊 EDA 분석":
        show_eda(data)
    elif page == "🔮 수요 예측":
        show_prediction(data, model, scaler, feature_cols)
    elif page == "📈 모델 성능":
        show_model_performance(data, model, scaler, feature_cols)
    elif page == "ℹ️ 프로젝트 정보":
        show_project_info()

def show_home(data, model, scaler, feature_cols):
    st.markdown('<div class="main-header">🚲 따릉이 수요 예측</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">서울시 공공자전거 시간당 수요 예측 시스템</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📊 총 데이터 건수",
            f"{len(data):,.0f}",
            "시간 단위"
        )

    with col2:
        st.metric(
            "🚴 평균 시간당 수요",
            f"{data['demand'].mean():.0f}",
            "대여건수"
        )

    with col3:
        st.metric(
            "📈 최고 수요",
            f"{data['demand'].max():.0f}",
            "대여건수"
        )

    with col4:
        st.metric(
            "🌡️ 평균 기온",
            f"{data['temperature'].mean():.1f}°C",
            "섭씨온도"
        )

    st.divider()

    st.subheader("⏰ 시간대별 평균 수요")
    hourly_demand = data.groupby('hour')['demand'].mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(hourly_demand.index, hourly_demand.values, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.set_xlabel('시간대 (Hour)', fontsize=12)
    ax.set_ylabel('평균 대여건수', fontsize=12)
    ax.set_title('시간대별 평균 따릉이 수요', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 요일별 평균 수요")
        daily_demand = data.groupby('day_of_week')['demand'].mean()
        days = ['월', '화', '수', '목', '금', '토', '일']

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#1f77b4'] * 5 + ['#ff7f0e'] * 2
        ax.bar(range(7), daily_demand.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_ylabel('평균 대여건수', fontsize=12)
        ax.set_title('요일별 평균 수요', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("🌡️ 기온별 수요 분석")
        temp_bins = [0, 5, 15, 25, 35]
        temp_labels = ['추움(<5°C)', '선선함(5-15°C)', '따뜻함(15-25°C)', '더움(>25°C)']
        data['temp_category'] = pd.cut(data['temperature'], bins=temp_bins, labels=temp_labels)
        temp_demand = data.groupby('temp_category')['demand'].mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(temp_demand)), temp_demand.values, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(temp_demand)))
        ax.set_yticklabels(temp_demand.index)
        ax.set_xlabel('평균 대여건수', fontsize=12)
        ax.set_title('기온별 수요 분석', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)

def show_eda(data):
    st.header("📊 탐색적 데이터 분석 (EDA)")

    st.subheader("📋 데이터 요약")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**데이터 기간:**")
        st.write(f"{data['date'].min().date()} ~ {data['date'].max().date()}")
        st.write(f"**총 {len(data):,}개 데이터 포인트**")

    with col2:
        st.write("**주요 통계:**")
        st.dataframe({
            '통계': ['평균', '중앙값', '최소값', '최대값', '표준편차'],
            '대여건수': [
                f"{data['demand'].mean():.2f}",
                f"{data['demand'].median():.2f}",
                f"{data['demand'].min():.2f}",
                f"{data['demand'].max():.2f}",
                f"{data['demand'].std():.2f}"
            ]
        })

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["시간대 분석", "요일 분석", "계절 분석", "기상 분석"])

    with tab1:
        st.subheader("⏰ 시간대별 수요 패턴")
        hourly = data.groupby('hour')['demand'].agg(['mean', 'std', 'min', 'max'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        ax1.plot(hourly.index, hourly['mean'], 'o-', linewidth=2, markersize=8, color='#1f77b4', label='평균')
        ax1.fill_between(hourly.index, hourly['min'], hourly['max'], alpha=0.2, color='#1f77b4', label='범위')
        ax1.set_xlabel('시간대 (Hour)', fontsize=12)
        ax1.set_ylabel('대여건수', fontsize=12)
        ax1.set_title('시간대별 수요 분포 (평균 ± 범위)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        rush_hours = [7, 8, 9, 17, 18, 19]
        for hour in rush_hours:
            ax1.axvline(hour, color='red', alpha=0.2, linestyle='--')

        hourly_stats = hourly.round(2)
        ax2.axis('off')
        table_data = []
        for hour in hourly_stats.index:
            table_data.append([
                f"{hour:02d}:00",
                f"{hourly_stats.loc[hour, 'mean']:.1f}",
                f"{hourly_stats.loc[hour, 'std']:.1f}",
                f"{hourly_stats.loc[hour, 'min']:.1f}",
                f"{hourly_stats.loc[hour, 'max']:.1f}"
            ])

        table = ax2.table(
            cellText=table_data,
            colLabels=['시간', '평균', '표준편차', '최소', '최대'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        st.pyplot(fig)

        st.info("""
        💡 **통찰:**
        - **출근시간 (7-9시):** 급격한 수요 증가 → 아침 배치 강화 필요
        - **오후 (12-14시):** 점심시간 소폭 증가
        - **퇴근시간 (17-19시):** 출근시간보다 높은 수요
        - **야간 (20시-06시):** 최저 수요 시간대
        """)

    with tab2:
        st.subheader("📅 요일별 수요 패턴")
        daily = data.groupby('day_of_week')['demand'].agg(['mean', 'std', 'count'])
        days = ['월', '화', '수', '목', '금', '토', '일']

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#1f77b4'] * 5 + ['#ff7f0e'] * 2
        bars = ax.bar(range(7), daily['mean'].values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.errorbar(range(7), daily['mean'].values, yerr=daily['std'].values, fmt='none', color='black', alpha=0.5)

        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_ylabel('평균 대여건수', fontsize=12)
        ax.set_title('요일별 평균 수요 (평균 ± 표준편차)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        ax.bar([], [], color='#1f77b4', alpha=0.7, label='평일', edgecolor='black')
        ax.bar([], [], color='#ff7f0e', alpha=0.7, label='주말', edgecolor='black')
        ax.legend()

        st.pyplot(fig)

        st.dataframe({
            '요일': days,
            '평균': daily['mean'].values.round(2),
            '표준편차': daily['std'].values.round(2),
            '샘플 수': daily['count'].values.astype(int)
        })

        st.info("""
        💡 **통찰:**
        - **평일 vs 주말:** 약 20-30% 수요 차이
        - **금요일:** 평일 중 가장 높은 수요 (퇴근 이동 증가)
        - **주말:** 시간대별 변동성이 적고 일정한 패턴
        """)

    with tab3:
        st.subheader("🌍 계절별 수요 패턴")

        monthly = data.groupby('month')['demand'].mean()
        months = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(1, 13), monthly.values, 'o-', linewidth=3, markersize=10, color='#2ca02c', label='평균')
        ax.fill_between(range(1, 13), monthly.values * 0.8, monthly.values * 1.2, alpha=0.2, color='#2ca02c')

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, rotation=45)
        ax.set_ylabel('평균 대여건수', fontsize=12)
        ax.set_title('월별 수요 트렌드', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        data['season'] = data['month'].apply(lambda x: 
            '겨울' if x in [12, 1, 2] else
            '봄' if x in [3, 4, 5] else
            '여름' if x in [6, 7, 8] else
            '가을'
        )

        seasonal = data.groupby('season')['demand'].agg(['mean', 'std', 'min', 'max'])
        st.dataframe({
            '계절': seasonal.index,
            '평균': seasonal['mean'].values.round(2),
            '표준편차': seasonal['std'].values.round(2),
            '최소': seasonal['min'].values.round(2),
            '최대': seasonal['max'].values.round(2)
        })

        st.info("""
        💡 **통찰:**
        - **여름 (6-8월):** 최고 수요 시기 (야외활동 증가)
        - **봄/가을:** 중간 정도의 안정적 수요
        - **겨울 (12-2월):** 최저 수요 시기 (추위, 눈)
        - **계절성 변동:** ±20-25% 범위
        """)

    with tab4:
        st.subheader("🌤️ 기상 조건별 수요 분석")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**기온별 수요:**")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(data['temperature'], data['demand'], 
                               c=data['demand'], cmap='RdYlBu_r', alpha=0.5, s=20)
            ax.set_xlabel('기온 (°C)', fontsize=12)
            ax.set_ylabel('대여건수', fontsize=12)
            ax.set_title('기온 vs 수요', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='대여건수')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.write("**강우량별 수요:**")
            fig, ax = plt.subplots(figsize=(10, 6))

            rainy = data[data['rainfall'] > 0]
            dry = data[data['rainfall'] == 0]

            ax.boxplot([dry['demand'], rainy['demand']], labels=['맑음', '비옴'], patch_artist=True)
            ax.set_ylabel('대여건수', fontsize=12)
            ax.set_title('강우 여부에 따른 수요', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            st.pyplot(fig)

        st.info("""
        💡 **통찰:**
        - **기온 효과:** 5°C 이상 25°C 이하에서 최적 수요
        - **극한 기후:** 극저온(<0°C), 극고온(>30°C) 시 수요 20-30% 감소
        - **강우 효과:** 비 오는 날 약 20-25% 수요 감소
        - **습도 효과:** 습도 70% 이상에서 미세한 수요 감소 경향
        """)

def show_prediction(data, model, scaler, feature_cols):
    st.header("🔮 따릉이 수요 예측")

    st.markdown("""
    특정 날짜와 시간의 따릉이 수요를 예측합니다.
    기상 조건, 요일, 시간대 등을 입력하세요.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌍 장소 및 날짜 설정")
        search_keyword = st.text_input("🔍 서울 내 장소 검색", "서울역")
        lat, lng, search_status = get_location_coords_by_keyword(search_keyword, KAKAO_API_KEY)
        
        if search_status != "성공":
            if "서울 외 지역" in search_status:
                st.error(f"❌ {search_status}")
            else:
                st.error("❌ 장소를 찾을 수 없습니다.")
            st.stop()
            
        # ==========================================
        # [수정된 부분] 미래 예측을 위해 최대 날짜를 1년(365일) 뒤로 시원하게 늘려줍니다!
        # ==========================================
        min_date = data['date'].min().date()
        today_date = datetime.now().date()
        max_date = today_date + timedelta(days=365) # 미래 날짜 제한 해제!
        
        if min_date <= today_date <= max_date:
            default_date = today_date
        else:
            default_date = max_date

        pred_date = st.date_input(
            "날짜 선택",
            value=default_date,
            min_value=min_date,
            max_value=max_date
        )
        pred_hour = st.slider("시간대 선택 (0-23시)", 0, 23, 12)

    with col2:
        st.subheader("🌤️ 기상 조건")
        date_str = pred_date.strftime("%Y-%m-%d")
        
        w_temp, w_rain, status = get_weather_by_date(lat, lng, date_str)
        
        if status == "성공":
            temperature = w_temp
            rainfall = w_rain
            humidity = 60
            st.success(f"🌦️ {date_str} 기상 데이터 자동 연동\n\n- 기온: {w_temp}°C\n- 강수량: {w_rain}mm")
        else:
            # 아주 먼 미래(예: 내년)를 골라 예보 API가 실패해도 터지지 않고 기본값 제공
            st.warning(f"⚠️ {date_str} 예보를 불러올 수 없습니다. (기본값 적용)")
            temperature = 20.0
            rainfall = 0.0
            humidity = 60
            st.info(f"- 자동 적용 기온: {temperature}°C\n- 자동 적용 강수량: {rainfall}mm")

    with col3:
        st.subheader("🗺️ 주변 인프라")
        schools, _ = get_nearby_poi_data(lat, lng, KAKAO_API_KEY, "SC4")
        subways, _ = get_nearby_poi_data(lat, lng, KAKAO_API_KEY, "SW8")
        st.write(f"- 인근 학교 수: {len(schools)}개")
        st.write(f"- 인근 지하철역 수: {len(subways)}개")

    st.divider()

    st.subheader(f"🗺️ '{search_keyword}' 주변 인프라 지도 시각화")
    m = folium.Map(location=[lat, lng], zoom_start=15)
    folium.Marker([lat, lng], popup=search_keyword, icon=folium.Icon(color='black')).add_to(m)
    folium.Circle([lat, lng], radius=1000, color="blue", fill=True, fill_opacity=0.1).add_to(m)
    
    for s in schools:
        folium.Marker([float(s['y']), float(s['x'])], icon=folium.Icon(color='orange', icon='book')).add_to(m)
    for sw in subways:
        folium.Marker([float(sw['y']), float(sw['x'])], icon=folium.Icon(color='blue', icon='train')).add_to(m)
        
    st_folium(m, width="100%", height=350, returned_objects=[])

    st.divider()

    if st.button("🔍 수요 예측 실행", use_container_width=True):
        pred_date_dt = pd.Timestamp(pred_date)
        day_of_week = pred_date_dt.dayofweek
        month = pred_date_dt.month
        day = pred_date_dt.day

        features_dict = {
            'hour': pred_hour,
            'day_of_week': day_of_week,
            'month': month,
            'day': day,
            'is_rush_hour': 1 if pred_hour in [7, 8, 9, 17, 18, 19] else 0,
            'is_morning': 1 if pred_hour < 12 else 0,
            'is_evening': 1 if pred_hour >= 18 else 0,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_spring': 1 if month in [3, 4, 5] else 0,
            'is_summer': 1 if month in [6, 7, 8] else 0,
            'is_autumn': 1 if month in [9, 10, 11] else 0,
            'is_winter': 1 if month in [12, 1, 2] else 0,
            'temperature': temperature,
            'is_rainy': 1 if rainfall > 0 else 0,
            'humidity': humidity
        }

        X_pred = np.array([features_dict[col] for col in feature_cols]).reshape(1, -1)
        X_pred_scaled = scaler.transform(X_pred)

        base_prediction = model.predict(X_pred_scaled)[0]
        infra_weight = (len(schools) * 2) + (len(subways) * 4)
        prediction = max(0, base_prediction + infra_weight)

        st.subheader("📊 예측 결과")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🚴 예상 수요",
                f"{prediction:.0f}",
                "대여건수"
            )

        with col2:
            avg_demand = data[data['hour'] == pred_hour]['demand'].mean()
            diff_pct = ((prediction - avg_demand) / avg_demand * 100) if avg_demand > 0 else 0

            st.metric(
                "📈 시간대 평균 대비",
                f"{diff_pct:+.1f}%",
                f"(평균: {avg_demand:.0f})"
            )

        with col3:
            std_demand = data[data['hour'] == pred_hour]['demand'].std()
            confidence = 100 - abs(prediction - avg_demand) / (std_demand + 1) * 100
            confidence = max(0, min(100, confidence))

            st.metric(
                "🎯 예측 신뢰도",
                f"{confidence:.1f}%"
            )

        st.divider()

        st.subheader("📋 입력 정보 요약")

        summary_data = {
            '항목': [
                '날짜',
                '시간',
                '요일',
                '기온',
                '강우량',
                '습도',
                '러시아워 여부',
                '기후'
            ],
            '값': [
                f"{pred_date_dt.strftime('%Y-%m-%d')}",
                f"{pred_hour:02d}:00",
                ['월', '화', '수', '목', '금', '토', '일'][day_of_week],
                f"{temperature:.1f}°C",
                f"{rainfall:.1f}mm",
                f"{humidity}%",
                "예" if features_dict['is_rush_hour'] else "아니오",
                "맑음" if rainfall == 0 else "흐림/비"
            ]
        }

        st.dataframe(summary_data, use_container_width=True, hide_index=True)

        st.subheader("🔍 비교 분석")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**같은 시간대 수요 분포:**")
            same_hour_data = data[data['hour'] == pred_hour]['demand']

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(same_hour_data, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
            ax.axvline(prediction, color='red', linestyle='--', linewidth=2, label=f'예측값 ({prediction:.0f})')
            ax.axvline(same_hour_data.mean(), color='green', linestyle='--', linewidth=2, label=f'평균 ({same_hour_data.mean():.0f})')
            ax.set_xlabel('대여건수', fontsize=12)
            ax.set_ylabel('빈도', fontsize=12)
            ax.set_title(f'{pred_hour:02d}시 수요 분포', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)

        with col2:
            st.write("**요일별 같은 시간대 수요:**")
            days = ['월', '화', '수', '목', '금', '토', '일']
            daily_hourly = []
            for d in range(7):
                mask = (data['day_of_week'] == d) & (data['hour'] == pred_hour)
                avg = data[mask]['demand'].mean() if mask.sum() > 0 else 0
                daily_hourly.append(avg)

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#1f77b4'] * 5 + ['#ff7f0e'] * 2
            ax.bar(range(7), daily_hourly, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(7))
            ax.set_xticklabels(days)
            ax.set_ylabel('평균 대여건수', fontsize=12)
            ax.set_title(f'{pred_hour:02d}시 요일별 평균 수요', fontsize=14, fontweight='bold')

            if day_of_week < 7:
                ax.patches[day_of_week].set_edgecolor('red')
                ax.patches[day_of_week].set_linewidth(3)

            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)

        st.subheader("💡 운영 권장사항")

        if prediction > data['demand'].quantile(0.75):
            st.success("""
            ✅ **높은 수요 예상**
            - 자전거 배치 증가 권장
            - 대여 불가 상황 방지를 위해 추가 배치 고려
            - 반납 수거 강화 필요
            """)
        elif prediction > data['demand'].quantile(0.25):
            st.info("""
            ℹ️ **중간 수준 수요**
            - 일반적인 배치 수준 유지
            - 특별한 조치 불필요
            """)
        else:
            st.warning("""
            ⚠️ **낮은 수요 예상**
            - 자전거 배치 감소 가능
            - 유지보수 시간 활용 가능
            - 다른 지역으로 배치 이동 검토
            """)

def show_model_performance(data, model, scaler, feature_cols):
    st.header("📈 모델 성능 분석")

    data_clean = data.dropna()
    X = data_clean[feature_cols]
    y = data_clean['demand']
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    st.subheader("📊 주요 성능 지표")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("R² 스코어", f"{r2:.4f}", "설명력")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}", "오차 범위 (±대여건수)")
    with col3:
        st.metric("MAE", f"{mae:.2f}", "절대 오차")
    with col4:
        st.metric("MSE", f"{mse:.2f}", "제곱 오차")
    with col5:
        st.metric("MAPE", f"{mape:.2f}%", "평균 퍼센트 오차")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["예측 정확도", "잔차 분석", "특성 중요도"])

    with tab1:
        st.subheader("예측값 vs 실제값")

        sample_idx = np.random.choice(len(y), size=min(5000, len(y)), replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        scatter = ax1.scatter(y.iloc[sample_idx], y_pred[sample_idx], alpha=0.5, s=10)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='완벽한 예측')
        ax1.set_xlabel('실제 수요', fontsize=12)
        ax1.set_ylabel('예측 수요', fontsize=12)
        ax1.set_title('예측값 vs 실제값', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = axes[1]
        plot_range = slice(0, min(500, len(y)))
        ax2.plot(range(len(y[plot_range])), y.iloc[plot_range].values, 'o-', alpha=0.6, label='실제값', linewidth=2)
        ax2.plot(range(len(y_pred[plot_range])), y_pred[plot_range], 's-', alpha=0.6, label='예측값', linewidth=2)
        ax2.set_xlabel('시간순서', fontsize=12)
        ax2.set_ylabel('대여건수', fontsize=12)
        ax2.set_title('예측 추이 (처음 500개 데이터)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        st.pyplot(fig)

    with tab2:
        st.subheader("잔차 분석")

        residuals = y - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('잔차', fontsize=12)
        ax1.set_ylabel('빈도', fontsize=12)
        ax1.set_title('잔차 분포', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        ax2 = axes[0, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q 플롯 (정규성 검증)', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        ax3 = axes[1, 0]
        ax3.scatter(y_pred[sample_idx], residuals.iloc[sample_idx], alpha=0.5, s=10)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('예측값', fontsize=12)
        ax3.set_ylabel('잔차', fontsize=12)
        ax3.set_title('잔차 vs 예측값', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)

        ax4 = axes[1, 1]
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(residuals.iloc[:1000], ax=ax4)
        ax4.set_title('잔차의 자기상관함수 (ACF)', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)

        st.pyplot(fig)

        st.info(f"""
        💡 **잔차 분석 결과:**
        - 평균 잔차: {residuals.mean():.4f} (0에 가까울수록 좋음)
        - 표준편차: {residuals.std():.2f}
        - 95% 신뢰 구간: [{residuals.quantile(0.025):.2f}, {residuals.quantile(0.975):.2f}]
        """)

    with tab3:
        st.subheader("특성 중요도")

        coefficients = model.coef_
        importance = np.abs(coefficients)
        importance_sorted_idx = np.argsort(importance)[-10:]

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = [feature_cols[i] for i in importance_sorted_idx]
        top_importance = importance[importance_sorted_idx]

        ax.barh(range(len(top_features)), top_importance, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('중요도 (절대 계수값)', fontsize=12)
        ax.set_title('상위 10개 중요 특성', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        st.pyplot(fig)

        st.dataframe({
            '특성': top_features,
            '중요도': top_importance.round(4),
            '계수': model.coef_[importance_sorted_idx].round(4)
        })

def show_project_info():
    st.header("ℹ️ 프로젝트 정보")

    st.subheader("🚲 따릉이 수요 예측 및 운영 최적화")

    st.markdown("""
    **프로젝트 설명:**

    서울시의 공공자전거 '따릉이'는 이용객이 급증하면서 대여소 간의 자전거 불균형이 발생하고 있습니다.
    예를 들어, 출근시간에는 강남역에서 자전거 부족 현상이 발생하는 반면, 
    한적한 주택가 대여소에서는 자전거가 남아돕니다.

    이 프로젝트는 **머신러닝**을 활용하여 다음을 수행합니다:
    - 📊 과거 1년간의 따릉이 대여 기록 분석
    - 🌡️ 실시간 기상 데이터와의 연계
    - 🗺️ 지역 특성(학교, 역, 편의점) 반영
    - 🔮 시간별, 지역별 수요 예측
    - 💡 자전거 배치 최적화 권장
    """)

    st.divider()

    st.subheader("👥 팀 구성")

    team_info = {
        '이름': ['박도영', '최선강', '최연규', '이태윤'],
        '역할': [
            '팀장 / 데이터 엔지니어링',
            '머신러닝 모델링',
            '기상 API & 데이터 수집',
            '공간 데이터 분석'
        ],
        '담당 업무': [
            '인프라 데이터 수집, 데이터 병합',
            '선형회귀, RandomForest, XGBoost',
            '기상청 API, 실시간 데이터',
            '학교/역/편의점 정보화'
        ]
    }

    st.dataframe(team_info, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("🛠️ 기술 스택")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **데이터 처리**
        - Pandas
        - NumPy
        """)

    with col2:
        st.markdown("""
        **머신러닝**
        - Scikit-learn
        - XGBoost
        - Random Forest
        """)

    with col3:
        st.markdown("""
        **시각화 & 배포**
        - Matplotlib
        - Seaborn
        - Streamlit
        """)

    st.divider()

    st.subheader("✨ 주요 기능")

    features = {
        '🔮 수요 예측': '특정 날짜/시간/기상 조건에 대한 따릉이 수요 예측',
        '📊 EDA 분석': '시간대별, 요일별, 계절별, 기상별 상세 분석',
        '📈 모델 성능': '머신러닝 모델의 성능 지표 및 특성 중요도 분석',
        '💡 운영 권장사항': '예측 결과 기반 자전거 배치 및 운영 전략',
        '🗺️ 지역 분석': '대여소 위치별 특성 및 수요 패턴 분석'
    }

    for feature, description in features.items():
        st.markdown(f"**{feature}**")
        st.write(description)
        st.write("")

    st.divider()

    st.subheader("🎯 프로젝트 목표")

    goals = {
        '예측 정확도 (R²)': '> 0.80 (현재: 0.63)',
        '오차 범위 (RMSE)': '< 1.0대 (현재: 1.55대)',
        'API 통합': '3개 API 실시간 연동',
        '대시보드': '실제 운영팀 활용 가능한 웹 앱'
    }

    for goal, target in goals.items():
        st.metric(goal, target)

    st.divider()

    st.subheader("📅 향후 계획")

    roadmap = {
        '1단계': 'XGBoost 하이퍼파라미터 최적화 (목표: R² > 0.75)',
        '2단계': '공간 데이터 통합 (학교, 역, 편의점)',
        '3단계': '신경망 모델 도입 (LSTM)',
        '4단계': '실시간 배포 및 운영 시스템 구축'
    }

    for phase, plan in roadmap.items():
        st.write(f"**{phase}:** {plan}")

    st.divider()

    st.subheader("📌 GitHub")
    st.markdown("""
    **프로젝트 저장소:**
    [https://github.com/chipoli410-art/re](https://github.com/chipoli410-art/re)

    모든 코드와 데이터는 GitHub에 공개되어 있습니다.
    """)

    st.subheader("📚 데이터 출처")
    st.markdown("""
    - **따릉이 대여 데이터:** [서울시 열린데이터광장](https://data.seoul.go.kr/)
    - **기상 데이터:** 기상청 OpenAPI
    - **지역 정보:** OpenStreetMap, Kakao Map API
    """)

if __name__ == "__main__":
    main()
