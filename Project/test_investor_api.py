import requests

url = "http://127.0.0.1:5000/recommend_startups"

data = {
 "industry": "AI/ML",
 "funding_round": "Seed",
 "region": "Asia"
}

response = requests.post(url, json=data)

print(response.json())