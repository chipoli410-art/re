# 🚲 서울시 따릉이 다차원 분석 및 실시간 수요 예측 시스템
> **공공데이터와 실시간 외부 API(기상, 인프라, 교통)를 융합한 지능형 자전거 재배치 솔루션**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=flat-square&logo=scikit-learn&logoColor=white)

## 📝 프로젝트 개요
서울시 공공자전거 '따릉이'의 수요는 날씨, 시간대, 요일, 그리고 주변 인프라(학교, 지하철역 등)와 도로 교통 상황에 따라 급격하게 변동합니다. 
본 프로젝트는 **1,000만 건 이상의 과거 대여 기록(1년 치)**을 분석하고, **실시간 외부 API**를 연동하여 특정 대여소의 예상 수요를 정확히 예측합니다. 이를 통해 관리자가 효율적으로 자전거를 재배치할 수 있도록 돕는 **'데이터 기반 의사결정 대시보드'**를 구축했습니다.

## 🌟 주요 기능 (Key Features)

### 1. 🔮 실시간 수요 예측 시뮬레이터
* **멀티 API 실시간 연동:** * `Kakao Local API`: 대여소 반경 1km 내 학교/지하철역 개수 수집
  * `Open-Meteo API`: 선택한 날짜의 기온, 날씨 상태, 미세먼지(PM10) 예보 데이터 수집
  * `OpenStreetMap (OSM)`: 실시간 도로망 좌표를 수집하여 지도 위에 렌더링
* **인터랙티브 지도 시각화 (`Folium`):** 수집된 인프라 마커와 예상 교통 상황에 따른 도로 정체망을 시각적으로 구현
* **ML 알고리즘 스위칭:** Rule-based, Random Forest(안정성 중심), XGBoost(외부 변수 민감도 중심) 등 상황에 맞는 예측 모델 전환 및 테스트 기능

### 2. 📊 과거 데이터 분석 (EDA) 대시보드
* 주요 거점별 일평균 대여량/반납량 및 출퇴근 집중도 통계 제공
* 맑은 날, 비 오는 날, 주말 등 기상 및 요일 조건에 따른 24시간 수요 패턴 시각화
* 핵심 지표 요약 카드(Metrics) 및 데이터 그리드 제공

## 🏗️ 시스템 아키텍처 및 데이터 파이프라인
1. **Data Source:** 서울시 공공데이터포털 (1,000만 행+), Kakao API, Open-Meteo API, OSM Overpass API
2. **Data Processing:** Python, Pandas, NumPy (결측치 처리, 파생 변수 생성)
3. **Modeling:** Linear Regression (Baseline 모델 구축 완료), LightGBM/XGBoost (고도화 예정)
4. **Deployment:** Streamlit Community Cloud (프론트엔드 UI 및 인터랙티브 대시보드)

## 📈 모델링 진행 상태 (Model Performance)
* **Dataset:** 1년 치 대여 기록 (약 1,000만 행)
* **Baseline Model (Linear Regression):** * `MSE`: 7.92
  * `RMSE`: 2.81 (평균 오차 2.8대 수준으로 매우 안정적인 초기 성능 확보)
* **Next Steps:** 요일, 시간대, 휴일 여부 등 파생 변수 추가 및 `LightGBM`을 활용한 대규모 데이터 학습 최적화 진행 중

## 💻 실행 방법 (How to Run)
```bash
# 1. 저장소 클론
git clone [https://github.com/](https://github.com/)[본인 깃허브 아이디]/[레포지토리 이름].git

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 3. Streamlit 앱 실행
streamlit run app.py
