깃허브(GitHub) README 또는 포트폴리오용
🚲 따릉이 수요 예측 및 운영 최적화 프로그램 (Ttareungi Demand Forecaster)
📌 Project Overview
서울시 열린데이터광장의 공공자전거 이용 정보와 기상청 오픈 API를 결합하여, 따릉이 대여소별 시간당 수요를 예측하는 머신러닝 모델입니다. 복잡한 도심의 수요 패턴을 분석하여 자전거 배치의 불균형 문제를 해결합니다.

👥 Team 6
박도영 (팀장/데이터 엔지니어링): 주변 시설 인프라 데이터 수집 및 데이터 병합, 가설 검증

최연규: 유동 인구 및 기상청 API 데이터 수집

최선강: 교통 체증 데이터 조사 및 예측 결과 시각화

이태윤: 대여소 주변 교육/편의 시설 정보 수치화

💡 Key Features
다중 변수 기반 예측: 시간대, 요일 등 시계열 데이터뿐만 아니라 날씨(기온, 강수량 등) 변수를 통합 학습

공간 인프라 특성 반영 (차별점): 대여소 주변의 '학교 수' 등 공간 인프라 데이터를 모델에 주입하여 예측 정확도(R²) 대폭 향상

단계적 모델 고도화: 선형 회귀(Baseline)에서 시작하여 비선형 패턴을 잡는 랜덤 포레스트(Random Forest) 모델을 거쳐, 향후 XGBoost로 최적화 진행 예정

🛠 Tech Stack
Language: Python

Data & ML: Pandas, NumPy, Scikit-learn

Visualization: Matplotlib, Seaborn

API: 기상청 Open API, 서울 열린데이터
