Tutor Recommendation System

This repository contains a proof-of-concept Tutor Recommendation System built using Python, Streamlit, and a simple Machine Learning model.
The prototype demonstrates how local tutoring agencies can use real operational data to deliver intelligent, data-driven tutor recommendations.
Since real commercial datasets were not available, the system currently uses dummy generated data to validate the workflow and recommendation logic.

Features
1. Parent Request Input (via UI)

 Data Requirement from Prents :
 
Subject

Area

Monthly budget

Preferred experience

Required days per week

2. Tutor Profiles (Synthetic)

The system generates fake tutors with:

Subjects taught

Experience (1–10 years)

Hourly & monthly pricing

Ratings

Areas & coordinates

Availability

3. Machine Learning Recommendation Engine

A Gradient Boosting Classifier is trained on synthetic historical match data.
It ranks tutors using features such as:

Subject match

Experience difference

Tutor rating

Distance (Haversine formula)

Days per week requirement

The output is a match probability score, and the system displays the Top 3 recommended tutors.

4. Analytics Dashboard

Built-in analytics include:

Most requested subjects

Average tutor rating by area

Tutor distribution by experience

Project Structure
│── app.py                 # Main Streamlit application
│── README.md              # Documentation (this file)

Why Dummy Data?

Local tutoring agencies usually maintain private datasets such as:

Tutor details

Parent demand history

Actual coordinate data

Pricing and experience records

These datasets are confidential; therefore, this project uses synthetic/dummy data to demonstrate the concept.
Once real datasets are provided, the recommendation engine will improve significantly in:

Accuracy

Relevance

Commercial usability

This prototype is fully prepared to integrate agency datasets without architectural changes.

Installation
1. Clone the Repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <Tutor-Recommendation-System>

2. Install Dependencies

pip install streamlit pandas numpy scikit-learn

Running the Application

Start the Streamlit app using one of the two valid commands:

streamlit run app.py


or:

python -m streamlit run app.py


After launching, open:

http://localhost:8501

 
UI screens

<img width="1358" height="626" alt="image" src="https://github.com/user-attachments/assets/f06d5d45-17b6-42f3-b39c-279bf6bf72cd" />


<img width="861" height="215" alt="image" src="https://github.com/user-attachments/assets/5ec4784f-d230-4e9d-a019-fc475e22c753" />

Analytics charts
<img width="809" height="439" alt="image" src="https://github.com/user-attachments/assets/5ab065a5-ef07-42cd-95a5-6c87229a9705" />

<img width="816" height="386" alt="image" src="https://github.com/user-attachments/assets/2167a9e0-f5e1-4bc1-a5fc-f20fbea94d85" />

<img width="830" height="446" alt="image" src="https://github.com/user-attachments/assets/6c543861-54d2-488e-b88c-b630aa99b9ca" />


Future Improvements

Once real tutoring agency data is available, the system can be enhanced with:

Better ML model trained on real parent–tutor match history

Filtering by gender, home tuition vs online, grade level

Distance-based travel time using Google Maps API

Tutor verification and student feedback loop

Deployment on cloud (AWS, GCP, Azure)
