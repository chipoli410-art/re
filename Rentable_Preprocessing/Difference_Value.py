import pandas as pd

# =========================
# 파일 경로
# =========================
actual_file = "../Rentable_Preprocessing/preprocessed_data.csv"
predict_file = "../Data_Preprocessing/prediction.csv"

# =========================
# CSV 읽기
# =========================
print("파일 읽는 중...")

actual_df = pd.read_csv(actual_file)
predict_df = pd.read_csv(predict_file)

# =========================
# 컬럼명 통일
# =========================
predict_df = predict_df.rename(columns={
    "대요소_ID_num": "대여소번호",
    "대여시간(시)": "시간대",
    "총_대여건수": "예측수량"
})

# =========================
# 데이터 병합
# =========================
print("데이터 병합 중...")

merged = pd.merge(
    actual_df,
    predict_df,
    on=["주차", "월", "요일", "대여소번호", "시간대"],
    how="inner"
)

print(f"병합된 행 수: {len(merged)}")

# =========================
# 부족수량 계산
# 부족수량 = 예측수량 - 거치대수량
# 음수는 0 처리
# =========================
merged["부족수량"] = (
    merged["예측수량"] - merged["거치대수량"]
).clip(lower=0)

# =========================
# 부족수량 5 이하 추출
# =========================
result = merged[merged["부족수량"] <= 5]

print(f"부족수량 5 이하 데이터 수: {len(result)}")

# =========================
# 저장
# =========================
merged.to_csv(
    "../Data_Preprocessing/all_result.csv",
    index=False,
    encoding="utf-8-sig"
)

result.to_csv(
    "../Data_Preprocessing/shortage_under_5.csv",
    index=False,
    encoding="utf-8-sig"
)

print("저장 완료")
print("전체 결과: all_result.csv")
print("부족수량 5 이하: shortage_under_5.csv")