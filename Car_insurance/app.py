from pathlib import Path
import joblib
import streamlit as st

# Load Pre-trained End-to-End Pipeline
@st.cache_resource
def load_pipeline():
    model_path = Path(__file__).parent / "model_pipeline.joblib"
    return joblib.load(model_path)


pipeline = load_pipeline()

st.title("Auto Insurance Claim Risk Assessment")
st.write("Enter policyholder details to calculate estimated claim probability.")

# Input Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Income ($)", min_value=0, value=50000)
    home_val = st.number_input("Home Value ($)", min_value=0, value=150000)
    mvr_pts = st.slider("MVR Violation Points", 0, 15, 0)
    clm_freq = st.slider("Claims in Past 5 Years", 0, 10, 0)
    kids_driv = st.number_input("Driving Children (KIDSDRIV)", min_value=0, max_value=5, value=0)

with col2:
    urbanicity = st.selectbox("Urbanicity", ["Highly Urban/ Urban", "Highly Rural/ Rural"])
    car_use = st.selectbox("Car Use", ["Private", "Commercial"])
    revoked = st.selectbox("License Revoked (Past 7 Years)", ["No", "Yes"])
    education = st.selectbox("Education Level", ["<High School", "High School", "Bachelors", "Masters", "PhD"])
    car_type = st.selectbox("Car Type", ["Minivan", "Van", "Panel Truck", "Pickup", "Sports Car", "SUV"])
    home_kids = st.number_input("Children at Home (HOMEKIDS)", min_value=0, max_value=10, value=0)

# Input Formatting (matches all 22 expected pipeline columns)
input_data = pd.DataFrame([{
    'KIDSDRIV': kids_driv,
    'AGE': age,
    'HOMEKIDS': home_kids,
    'YOJ': 5,
    'INCOME': income,
    'PARENT1': 'No',
    'HOME_VAL': home_val,
    'MSTATUS': 'Yes',
    'GENDER': 'M',
    'EDUCATION': education,
    'OCCUPATION': 'Professional',
    'TRAVTIME': 30,
    'CAR_USE': car_use,
    'BLUEBOOK': 15000,
    'TIF': 4,
    'CAR_TYPE': car_type,
    'OLDCLAIM': 0,
    'CLM_FREQ': clm_freq,
    'REVOKED': revoked,
    'MVR_PTS': mvr_pts,
    'CAR_AGE': 5,
    'URBANICITY': urbanicity
}])

if st.button("Calculate Claim Risk"):
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader(f"Calculated Claim Probability: {probability:.1%}")
    if prediction == 1:
        st.error("Result: HIGH RISK - Policyholder is likely to file a claim.")
    else:
        st.success("Result: LOW RISK - Policyholder is unlikely to file a claim.")