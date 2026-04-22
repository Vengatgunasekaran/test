import requests

url = "http://127.0.0.1:5000/predict_funding"

data = {
 "industry": "AI/ML",
 "funding_round": "Seed",
 "region": "Asia",
 "employee_count": 50,
 "estimated_revenue_usd": 500000,
 "founded_year": 2022
}

response = requests.post(url, json=data)

print(response.json())