# app.py
import streamlit as st
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# -----------------------------
# Step 1: Generate Fake Areas
# -----------------------------
areas = [
    {"area_id": i+1, "name": name, "latitude": round(random.uniform(24.8, 25.1), 5),
     "longitude": round(random.uniform(66.9, 67.2), 5)}
    for i, name in enumerate([
        "Clifton", "DHA", "Gulshan-e-Iqbal", "Korangi", "Nazimabad",
        "North Nazimabad", "PECHS", "Saddar", "Lyari", "Malir",
        "Landhi", "Karachi Cantt", "Shah Faisal", "Orangi Town", "Karachi University Area"
    ])
]
areas_df = pd.DataFrame(areas)

# -----------------------------
# Step 2: Generate Fake Tutors
# -----------------------------
subjects = ["Computer Science", "English", "Maths", "Chemistry", "Physiscs", "Biology", "Urdu", "Islamic Studies"]
tutors = []
for i in range(30):
    area = random.choice(areas_df['name'])
    tutor = {
        "tutor_id": i+1,
        "name": f"Tutor_{i+1}",
        "subjects": random.sample(subjects, k=random.randint(1, 3)),
        "area": area,
        "latitude": float(areas_df[areas_df['name'] == area]['latitude']),
        "longitude": float(areas_df[areas_df['name'] == area]['longitude']),
        "experience_years": random.randint(1, 10),
        "rating": round(random.uniform(3.0, 5.0), 1),
        "price_per_hour": random.randint(500, 2000),
        # Add monthly price to match monthly budget UI (optional)
        "price_per_month": random.randint(10000, 80000),
        # Optional: tutor availability in days per week (1-7)
        "available_days_per_week": random.randint(1, 7)
    }
    tutors.append(tutor)
tutors_df = pd.DataFrame(tutors)

# -----------------------------
# Step 3: Generate Fake Parent Requests
# -----------------------------
requests = []
for i in range(10):
    area = random.choice(areas_df['name'])
    req = {
        "request_id": i+1,
        "subject": random.choice(subjects),
        "area": area,
        "latitude": float(areas_df[areas_df['name'] == area]['latitude']),
        "longitude": float(areas_df[areas_df['name'] == area]['longitude']),
        # Keep budget meaning as monthly budget (UI expects monthly)
        "budget": random.randint(5000, 60000),
        "preferred_experience": random.randint(1, 5),
        # IMPORTANT: include days_per_week here so historical data is complete
        "days_per_week": random.randint(1, 7)
    }
    requests.append(req)
requests_df = pd.DataFrame(requests)

# -----------------------------
# Step 4: Haversine Distance
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# -----------------------------
# Step 5: Simulate Historical Matches for ML
# -----------------------------
historical_data = []
for req in requests:
    potential_tutors = tutors_df.sample(3)
    chosen_tutor = potential_tutors.sample(1).iloc[0]
    for _, tutor in potential_tutors.iterrows():
        historical_data.append({
            "subject_match": int(req['subject'] in tutor['subjects']),
            "experience_diff": tutor['experience_years'] - req['preferred_experience'],
            "rating": tutor['rating'],
            "distance": haversine(req['latitude'], req['longitude'], tutor['latitude'], tutor['longitude']),
            # include days_per_week from the request (numeric)
            "days_per_week": req["days_per_week"],
            "selected": int(tutor['tutor_id'] == chosen_tutor['tutor_id'])
        })

hist_df = pd.DataFrame(historical_data)

# In case tiny dataset causes issues, ensure no NaNs and types correct
hist_df = hist_df.fillna(0)
hist_df[['subject_match','experience_diff','rating','distance','days_per_week','selected']] = \
    hist_df[['subject_match','experience_diff','rating','distance','days_per_week','selected']].astype(float)

X = hist_df[['subject_match','experience_diff','rating','distance','days_per_week']]
y = hist_df['selected'].astype(int)

# If y has only one class (unlikely here), GradientBoostingClassifier will raise an error.
# To be robust, check and handle that edge case.
if len(y.unique()) == 1:
    # create a trivial synthetic negative example by flipping one sample (only if necessary)
    # this is a quick fix for very tiny or degenerate simulated datasets
    y.iloc[0] = 1 - y.iloc[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# -----------------------------
# Step 6: Recommendation Function
# -----------------------------
def recommend_tutors_ml(request, tutors_df, model, top_n=3):
    candidates = tutors_df.copy()
    # features required by model
    candidates['subject_match'] = candidates['subjects'].apply(lambda x: int(request['subject'] in x))
    candidates['experience_diff'] = candidates['experience_years'] - request['preferred_experience']
    # set days_per_week for each candidate equal to request (could be modified to match tutor availability)
    candidates['days_per_week'] = request['days_per_week']
    candidates['distance'] = candidates.apply(
        lambda row: haversine(request['latitude'], request['longitude'], row['latitude'], row['longitude']),
        axis=1
    )

    features = candidates[['subject_match','experience_diff','rating','distance','days_per_week']].astype(float)

    # predict_proba requires same feature order as used for training
    candidates['pred_score'] = model.predict_proba(features)[:, 1]
    recommended = candidates.sort_values(by='pred_score', ascending=False).head(top_n)
    return recommended

# -----------------------------
# Step 7: Streamlit UI
# -----------------------------
st.title("Tutor Recommendation System (ML + Distance)")

st.sidebar.header("Parent Request Input")
selected_subject = st.sidebar.selectbox("Subject", subjects)
selected_area = st.sidebar.selectbox("Area", areas_df['name'])
preferred_experience = st.sidebar.slider("Preferred Tutor Experience (Years)", 1, 10, 3)
# Monthly budget slider (as you requested)
budget = st.sidebar.slider("Monthly Budget", 5000, 60000, 20000)
days_per_week = st.sidebar.slider("Days per Week", 1, 7, 3)

request_input = {
    "subject": selected_subject,
    "area": selected_area,
    "latitude": float(areas_df[areas_df['name'] == selected_area]['latitude']),
    "longitude": float(areas_df[areas_df['name'] == selected_area]['longitude']),
    "preferred_experience": preferred_experience,
    "budget": budget,
    "days_per_week": days_per_week
}

st.subheader("Top Recommended Tutors")
recommended_tutors_ml = recommend_tutors_ml(request_input, tutors_df, gb_model)
st.dataframe(recommended_tutors_ml[[
    'tutor_id','name','subjects','area','experience_years','rating','price_per_hour','price_per_month','available_days_per_week','distance','pred_score'
]])

# -----------------------------
# Step 8: Analytics Dashboard
# -----------------------------
st.subheader("Analytics Dashboard")
st.markdown("**Most Popular Subjects in Requests:**")
st.bar_chart(requests_df['subject'].value_counts())

st.markdown("**Average Tutor Rating by Area:**")
st.bar_chart(tutors_df.groupby('area')['rating'].mean())
st.markdown("**Tutor Distribution by Experience:**")
st.bar_chart(tutors_df['experience_years'].value_counts().sort_index())