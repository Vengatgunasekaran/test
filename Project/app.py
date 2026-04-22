import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sklearn.metrics.pairwise import cosine_similarity
import random
import spacy
import pycountry
import os
from dotenv import load_dotenv

load_dotenv()

USD_TO_INR = 83
# -------------------------------
# Flask App Setup
# -------------------------------

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "bird_secret_key")

# Database Config
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URI", "sqlite:///instance/users.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = os.getenv("SQLALCHEMY_TRACK_MODIFICATIONS", False)

db = SQLAlchemy(app)

# -------------------------------
# Load ML Models
# -------------------------------

model = joblib.load("models/investor_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

success_model = joblib.load("models/startup_success_model.pkl")
success_feature_columns = joblib.load("models/success_feature_columns.pkl")

funding_model = joblib.load("models/funding_prediction_model.pkl")
funding_feature_columns = joblib.load("models/funding_feature_columns.pkl")
nlp = spacy.load("en_core_web_sm")
# Dataset
startup_data = pd.read_csv("data/cleaned_startup_dataset.csv")

startup_features = startup_data[
    ["industry","funding_round","region","employee_count","estimated_revenue_usd"]
]

startup_features = pd.get_dummies(startup_features)

# -------------------------------
# Database Model
# -------------------------------

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100))

    email = db.Column(db.String(100), unique=True)

    password = db.Column(db.String(100))

    role = db.Column(db.String(20))

# -------------------------------
# Helper Functions
# -------------------------------
import re

def extract_startup_info(text):

    text = text.lower()

    data = {
        "industry": None,
        "funding_round": None,
        "region": None,
        "employee_count": None,
        "estimated_revenue_usd": None,
        "founded_year": None
    }

    industries = ["ai","fintech","healthcare","saas","blockchain"]
    funding = ["pre-seed","seed","semi-seed","series a","series b"]

    for i in industries:
        if i in text:
            data["industry"] = i

    for f in funding:
        if f in text:
            data["funding_round"] = f

    # detect region
    region = detect_region_from_country(text)
    if region:
        data["region"] = region

    # detect numbers inside sentences
    numbers = re.findall(r'\d+', text)

    for num in numbers:

        n = int(num)

        if 1990 <= n <= 2030:
            data["founded_year"] = n

        elif "employee" in text or "employees" in text:
            data["employee_count"] = n

        elif "revenue" in text or "income" in text:
            data["estimated_revenue_usd"] = n

    return data

def login_required():
    if "user_id" not in session:
        return redirect(url_for("login"))

def find_missing_fields(data):

    order = [
        "industry",
        "funding_round",
        "region",
        "employee_count",
        "estimated_revenue_usd",
        "founded_year"
    ]

    for field in order:
        if data.get(field) is None:
            return [field]

    return []

def detect_region_from_country(text):

    text = text.lower()

    for country in pycountry.countries:

        name = country.name.lower()

        if name in text:

            if country.alpha_2 in [
                "IN","ID","CN","JP","SG","TH","MY","KR"
            ]:
                return "asia"

            if country.alpha_2 in [
                "US","CA","MX"
            ]:
                return "north america"

            if country.alpha_2 in [
                "DE","FR","IT","ES","NL","GB"
            ]:
                return "europe"

    return None

def calculate_startup_score(startup):

    score = 0

    # Revenue score (max 40)
    revenue = startup.get("estimated_revenue_usd", 0)
    if revenue > 1000000:
        score += 40
    elif revenue > 500000:
        score += 30
    elif revenue > 100000:
        score += 20
    else:
        score += 10

    # Employee score (max 30)
    employees = startup.get("employee_count", 0)
    if employees > 1000:
        score += 30
    elif employees > 200:
        score += 20
    elif employees > 50:
        score += 10
    else:
        score += 5

    # Funding stage score (max 30)
    stage = startup.get("funding_round", "").lower()

    if stage == "series c":
        score += 30
    elif stage == "series b":
        score += 25
    elif stage == "series a":
        score += 20
    elif stage == "seed":
        score += 15
    else:
        score += 10

    return score
# -------------------------------
# ML Prediction Functions
# -------------------------------

def calculate_compatibility(data, investor):

    score = 0

    investor_industries = {

        "Sequoia": ["AI/ML", "Fintech", "SaaS"],
        "Accel": ["AI/ML", "SaaS", "E-commerce"],
        "Tiger Global": ["E-commerce", "Fintech"],
        "Andreessen Horowitz": ["AI/ML", "Blockchain"],
        "Index Ventures": ["SaaS", "Fintech"],
        "SoftBank": ["AI/ML", "Logistics"],
        "Y Combinator": ["AI/ML", "Healthcare"]

    }

    if data["industry"] in investor_industries.get(investor, []):
        score += 0.4

    if data["funding_round"] in ["Seed", "Pre-Seed"]:
        score += 0.3

    if data["region"] in ["Asia", "North America"]:
        score += 0.2

    if data["estimated_revenue_usd"] > 100000:
        score += 0.1

    return round(score,2)


# -------------------------------
# Funding Prediction
# -------------------------------

def predict_funding_amount(data, investor):

    input_data = data.copy()

    # add investor feature
    input_data["lead_investor"] = investor

    df = pd.DataFrame([input_data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=funding_feature_columns, fill_value=0)

    prediction = funding_model.predict(df)[0]

    return int(prediction)

def predict_startup_success(data):

    # Normalize text
    data["industry"] = str(data["industry"]).strip().title()
    data["funding_round"] = str(data["funding_round"]).strip().title()
    data["region"] = str(data["region"]).strip().title()

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=success_feature_columns, fill_value=0)

    probability = success_model.predict_proba(df)[0][1]

    return round(probability * 100,2)

# -------------------------------
# Investor Recommendation
# -------------------------------

def predict_investor(data):

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=feature_columns, fill_value=0)

    probabilities = model.predict_proba(df)[0]

    investors = label_encoder.classes_

    # Calculate startup success probability
    success_probability = predict_startup_success(data)

    results = []

    for investor, prob in zip(investors, probabilities):

        compatibility = calculate_compatibility(data, investor)

        final_score = (prob * 0.6) + (compatibility * 0.4)

        # funding prediction
        funding_amount = predict_funding_amount(data, investor)

        results.append({

            "investor": investor,
            "match_probability": int(prob * 100),
            "compatibility_score": int(compatibility * 100),
            "final_score": int(final_score * 100),
            "startup_success_probability": success_probability,
            "funding_amount": int(funding_amount * USD_TO_INR)

        })

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    max_score = results[0]["final_score"]

    for r in results:
        r["normalized_score"] = int((r["final_score"] / max_score) * 100)

    return results[:3]


# -------------------------------
# Startup Recommendation
# -------------------------------

def recommend_startups(data):

    investor_profile = pd.DataFrame([{

        "industry": data.get("industry"),
        "funding_round": data.get("funding_round"),
        "region": data.get("region"),
        "employee_count": data.get("employee_count", 100),
        "estimated_revenue_usd": data.get("estimated_revenue_usd", 1000000)

    }])

    investor_profile = pd.get_dummies(investor_profile)

    investor_profile = investor_profile.reindex(columns=startup_features.columns, fill_value=0)

    similarity_scores = cosine_similarity(investor_profile, startup_features)[0]

    startup_data["similarity_score"] = similarity_scores

    top = startup_data.sort_values(by="similarity_score", ascending=False).head(5)

    results = []

    for _, row in top.iterrows():

        startup = {

            "industry": row["industry"],
            "funding_round": row["funding_round"],
            "region": row["region"],
            "employee_count": row["employee_count"],
            "estimated_revenue_usd": row["estimated_revenue_usd"],
            "similarity_score": row["similarity_score"]

        }

        startup["startup_score"] = calculate_startup_score(startup)

        # generic funding prediction
        funding_usd = predict_funding_amount(startup, "Sequoia")
        startup["funding_amount"] = int(funding_usd * USD_TO_INR)

        results.append(startup)

    return results


# -------------------------------
# Authentication Routes
# -------------------------------

@app.route("/")
def landing():

    return render_template("landing.html")

@app.route("/register",methods=["GET","POST"])
def register():

    if request.method == "POST":

        existing = User.query.filter_by(email=request.form["email"]).first()

        if existing:
            return "User already exists"

        user = User(

            name=request.form["name"],
            email=request.form["email"],
            password=request.form["password"],
            role=request.form["role"]

        )

        db.session.add(user)
        db.session.commit()

        session["user_id"] = user.id
        session["role"] = user.role

        if user.role == "Startup":
            return redirect("/startup_form")
        else:
            return redirect("/investor_form")

    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():

    # If already logged in → redirect
    if "user_id" in session:
        return redirect(url_for("dashboard_page"))

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email, password=password).first()

        if user:

            session["user_id"] = user.id
            session["role"] = user.role

            # redirect based on role
            if user.role == "Startup":
                return redirect(url_for("startup_form"))
            else:
                return redirect(url_for("investor_form"))

        return "Invalid Login"

    return render_template("login.html")

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")

# -------------------------------
# Protected Pages
# -------------------------------

@app.route("/dashboard_page")
def dashboard_page():

    if "user_id" not in session:
        return redirect("/login")

    return render_template("dashboard.html")

@app.route("/startup_form")
def startup_form():

    check = login_required()
    if check:
        return check

    if session.get("role") != "Startup":
        return redirect(url_for("login"))

    return render_template("startup_form.html")

@app.route("/investor_form")
def investor_form():

    check = login_required()
    if check:
        return check

    if session.get("role") != "Investor":
        return redirect(url_for("login"))

    return render_template("investor_form.html")

@app.route("/chatbot")
def chatbot():

    if "user_id" not in session:
        return redirect("/login")

    return render_template("chatbot.html")

@app.route("/investor_chatbot")
def investor_chatbot():

    if "user_id" not in session:
        return redirect("/login")

    return render_template("investor_chatbot.html")

@app.route("/feed_page")
def feed_page():

    if "user_id" not in session:
        return redirect("/login")

    return render_template("feed.html")
    
# -------------------------------
# ML API Routes
# -------------------------------

@app.route("/startup_match", methods=["POST"])
def startup_match():

    USD_TO_INR = 83
    revenue_inr = float(request.form["estimated_revenue_inr"])

    data = {
        "industry": request.form["industry"],
        "funding_round": request.form["funding_round"],
        "region": request.form["region"],
        "employee_count": int(request.form["employee_count"]),
        "estimated_revenue_usd": revenue_inr / USD_TO_INR
    }

    recommendations = predict_investor(data)

    return render_template(
        "results.html",
        recommendations=recommendations
    )

@app.route("/investor_match",methods=["POST"])
def investor_match():

    data = {

        "industry":request.form["industry"],
        "funding_round":request.form["funding_round"],
        "region":request.form["region"]

    }

    startups = recommend_startups(data)

    return render_template(
        "startup_results.html",
        startups=startups
    )

@app.route("/chatbot_match", methods=["POST"])
def chatbot_match():

    data = request.json

    recommendations = predict_investor(data)

    return jsonify({
        "recommendations": recommendations
    })

@app.route("/investor_chatbot_match",methods=["POST"])
def investor_chatbot_match():

    data = request.json

    startups = recommend_startups(data)

    return jsonify({
        "startups":startups
    })

# -------------------------------
# Feed API
# -------------------------------

@app.route("/feed")
def feed():

    random_rows = startup_data.sample(3)

    feed=[]

    for _,row in random_rows.iterrows():

        feed.append({

            "message":f"🔥 Startup in {row['industry']} seeking {row['funding_round']} funding in {row['region']}"

        })

    investor=random.choice(startup_data["lead_investor"].unique())

    feed.append({

        "message":f"⭐ Investor {investor} actively investing this week"

    })

    return jsonify(feed)

@app.route("/smart_chat", methods=["POST"])
def smart_chat():

    message = request.json["message"]

    # initialize session storage
    if "startup_profile" not in session:
        session["startup_profile"] = {
            "industry": None,
            "funding_round": None,
            "region": None,
            "employee_count": None,
            "estimated_revenue_usd": None,
            "founded_year": None
        }

    profile = session["startup_profile"]

    extracted = extract_startup_info(message)

    # update profile with new extracted data
    for key in profile:
        if extracted.get(key):
            profile[key] = extracted[key]

    session["startup_profile"] = profile

    # check missing fields
    missing = find_missing_fields(profile)

    if missing:

        question_map = {

"industry":"Which industry is your startup in?",

"funding_round":"What funding stage are you currently in?",

"region":"Which region is your startup located in?",

"employee_count":"How many employees work in your startup?",

"estimated_revenue_usd":"What is your estimated annual revenue (INR)?",

"founded_year":"In which year was your startup founded?"

}

        return jsonify({
            "reply": question_map[missing[0]],
            "recommendations": []
        })

    # run prediction
    investors = predict_investor(profile)

    # reset session after prediction
    session.pop("startup_profile", None)

    return jsonify({
        "reply": "Great! Based on your startup profile, these investors match best.",
        "recommendations": investors
    })

# -------------------------------

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)