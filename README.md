# 🚲 따릉이 수요 예측 및 지능형 관제 대시보드 (Ttareungi Demand Prediction)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-ff69b4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-77B829?style=for-the-badge&logo=leaflet&logoColor=white)

> "데이터 기반의 똑똑한 공공자전거 재배치(Rebalancing), 예측부터 관제까지 한 번에!"

## 📢 프로젝트 소개 (Project Overview)
서울시 공공자전거 '따릉이'의 대여소별 시간당 수요를 예측하고, 현장 실무자가 직관적으로 모니터링할 수 있는 지능형 관제 웹 대시보드입니다. 
기존 정적(Static) 예측 모델이 가지는 시계열 데이터 편이(Data Drift) 현상을 극복하기 위해 Walk-Forward Validation(전진 검증) 기법과 LightGBM 알고리즘을 도입하여, 최신 이용 트렌드 변화에 모델이 스스로 적응하도록 고도화했습니다.

## ✨ 주요 기능 (Key Features)

* 🔮 동적 수요 예측 (Dynamic Demand Prediction)
  * 직전 9개월 데이터를 롤링(Rolling) 학습하여 미래 수요 예측 (과적합 방지 및 성능 극대화)
  * 과거 평균 대여량, 실시간 기상(Open-Meteo), 공간 인프라(Kakao Local) 등 12종 핵심 피처(Feature) 융합
* 🗺️ 실시간 웹 대시보드 (Interactive Dashboard)
  * `Streamlit` 기반의 실무자 맞춤형 UI 제공
  * `Folium`을 활용한 검색 지역 반경 1km 내 유동 인구 유발 인프라(학교, 지하철역) 지도 시각화
* 💡 운영 권장사항 산출 (Actionable Insights)
  * 실시간 예측 수요량에 따른 즉각적인 재배치 액션 플랜(자전거 배치 증가 / 유지 / 감소) 가이드 제공

## 🛠 기술 스택 (Tech Stack)

### Data Processing & ML
* Pandas / NumPy: 대용량 데이터(약 1,000만 건) Chunking 최적화 및 피처 엔지니어링
* Scikit-learn: 데이터 스케일링 및 모델 성능 평가 (RMSE, R²)
* LightGBM: 대규모 비선형 시계열 데이터 최적화 및 수요 예측 고도화

### Frontend & Visualization
* Streamlit: 파이썬 기반 반응형 웹 대시보드 구축
* Folium / Streamlit-folium: 공간 데이터 기반 맵 시각화
* Matplotlib / Seaborn: 시간/기상별 대여량 EDA 및 특성 중요도 그래프 시각화

### Open API
* Open-Meteo API: 과거 및 실시간 미래 기상 예보 데이터 자동 연동
* Kakao Local API: 공간 인프라(반경 1km 내 특정 시설 개수) 데이터 수집

## 📈 모델 성능 (Model Performance)

Walk-Forward Validation (직전 9개월 학습 ➡️ 타겟 계절 예측) 방식 적용 결과, 장기 예측 시 발생하는 성능 저하 문제를 해결하고 사계절 모두 안정적이고 높은 예측 성능을 달성했습니다.

| 타겟 계절 (기간) | RMSE | R² Score | 분산 (Variance) |
| :--- | :--- | :--- | :--- |
| ❄️ 겨울 (24.12~25.02) | 1.4271 | 0.4140 | 3.4753 |
| 🌸 봄 (25.03~25.05) | 2.0097 | 0.5968 | 10.0160 |
| ☀️ 여름 (25.06~25.08) | 1.9039 | 0.6457 | 10.2304 |
| 🍁 가을 (25.09~25.11) | 1.8820 | **0.6558** | 10.2905 |

## 🚀 실행 방법 (How to Run)

```bash
# 1. 저장소 클론
git clone [https://github.com/chipoli410-art/re.git](https://github.com/chipoli410-art/re.git)
cd re

# 2. 필수 라이브러리 설치
pip install -r requirements.txt

# 3. Streamlit 대시보드 실행
streamlit run app.py
