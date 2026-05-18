import requests

key = "09611d17ff9500ed2d94a6d607cf3609"
url = "https://dapi.kakao.com/v2/local/search/category.json"
headers = {"Authorization": f"KakaoAK {key}"}
params = {"category_group_code": "SC4", "y": 37.4979, "x": 127.0276, "radius": 5000}

res = requests.get(url, headers=headers, params=params)
print(res.status_code)
print(res.text)