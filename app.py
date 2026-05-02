import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")

@st.cache_resource
def train_model():
    df = pd.read_csv("hr_data.csv")
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    pipeline = ImbPipeline(steps=[
        ("scaler",     StandardScaler()),
        ("smote",      SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline

with st.spinner("🔧 Training model, please wait..."):
    pipeline = train_model()

st.title("👥 HR Attrition Predictor")
st.markdown("Fill in the employee details below and click **Predict** to check attrition risk.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    age                  = st.slider("Age", 18, 60, 30)
    gender_male          = st.selectbox("Gender", ["Female", "Male"])
    marital_status       = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
    education            = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                        format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
    education_field      = st.selectbox("Education Field", [
                                "Human Resources", "Life Sciences", "Marketing",
                                "Medical", "Other", "Technical Degree"])
    distance_from_home   = st.slider("Distance From Home (km)", 1, 30, 5)

with col2:
    st.subheader("Job Details")
    department           = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
    job_role             = st.selectbox("Job Role", [
                                "Healthcare Representative", "Human Resources",
                                "Laboratory Technician", "Manager",
                                "Manufacturing Director", "Research Director",
                                "Research Scientist", "Sales Executive",
                                "Sales Representative"])
    job_level            = st.slider("Job Level", 1, 5, 2)
    job_involvement      = st.slider("Job Involvement (1-4)", 1, 4, 3)
    job_satisfaction     = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    business_travel      = st.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
    overtime             = st.selectbox("OverTime", ["No", "Yes"])

with col3:
    st.subheader("Compensation & Experience")
    monthly_income       = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
    daily_rate           = st.number_input("Daily Rate", 100, 1500, 800, step=50)
    hourly_rate          = st.number_input("Hourly Rate", 30, 100, 65, step=5)
    monthly_rate         = st.number_input("Monthly Rate", 2000, 27000, 14000, step=500)
    percent_salary_hike  = st.slider("Percent Salary Hike", 11, 25, 15)
    performance_rating   = st.selectbox("Performance Rating", [3, 4],
                                        format_func=lambda x: {3:"Excellent", 4:"Outstanding"}[x])
    stock_option_level   = st.slider("Stock Option Level", 0, 3, 1)

st.divider()
col4, col5 = st.columns(2)

with col4:
    st.subheader("Work History")
    total_working_years      = st.slider("Total Working Years", 0, 40, 10)
    num_companies_worked     = st.slider("Num Companies Worked", 0, 9, 2)
    years_at_company         = st.slider("Years At Company", 0, 40, 5)
    years_in_current_role    = st.slider("Years In Current Role", 0, 18, 3)
    years_since_last_promo   = st.slider("Years Since Last Promotion", 0, 15, 2)
    years_with_curr_manager  = st.slider("Years With Current Manager", 0, 17, 3)
    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 3)

with col5:
    st.subheader("Satisfaction Scores")
    environment_satisfaction  = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
    work_life_balance         = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    employee_number           = st.number_input("Employee Number", min_value=1, value=1)

st.divider()

def build_input():
    return pd.DataFrame([{
        "Age":                                  age,
        "DailyRate":                            daily_rate,
        "DistanceFromHome":                     distance_from_home,
        "Education":                            education,
        "EmployeeCount":                        1,
        "EmployeeNumber":                       employee_number,
        "EnvironmentSatisfaction":              environment_satisfaction,
        "HourlyRate":                           hourly_rate,
        "JobInvolvement":                       job_involvement,
        "JobLevel":                             job_level,
        "JobSatisfaction":                      job_satisfaction,
        "MonthlyIncome":                        monthly_income,
        "MonthlyRate":                          monthly_rate,
        "NumCompaniesWorked":                   float(num_companies_worked),
        "PercentSalaryHike":                    percent_salary_hike,
        "PerformanceRating":                    performance_rating,
        "RelationshipSatisfaction":             relationship_satisfaction,
        "StandardHours":                        80,
        "StockOptionLevel":                     float(stock_option_level),
        "TotalWorkingYears":                    float(total_working_years),
        "TrainingTimesLastYear":                float(training_times_last_year),
        "WorkLifeBalance":                      work_life_balance,
        "YearsAtCompany":                       years_at_company,
        "YearsInCurrentRole":                   float(years_in_current_role),
        "YearsSinceLastPromotion":              float(years_since_last_promo),
        "YearsWithCurrManager":                 float(years_with_curr_manager),
        # One-hot encoded columns
        "BusinessTravel_Travel_Frequently":     business_travel == "Travel_Frequently",
        "BusinessTravel_Travel_Rarely":         business_travel == "Travel_Rarely",
        "Department_Research & Development":    department == "Research & Development",
        "Department_Sales":                     department == "Sales",
        "EducationField_Life Sciences":         education_field == "Life Sciences",
        "EducationField_Marketing":             education_field == "Marketing",
        "EducationField_Medical":               education_field == "Medical",
        "EducationField_Other":                 education_field == "Other",
        "EducationField_Technical Degree":      education_field == "Technical Degree",
        "Gender_Male":                          gender_male == "Male",
        "JobRole_Human Resources":              job_role == "Human Resources",
        "JobRole_Laboratory Technician":        job_role == "Laboratory Technician",
        "JobRole_Manager":                      job_role == "Manager",
        "JobRole_Manufacturing Director":       job_role == "Manufacturing Director",
        "JobRole_Research Director":            job_role == "Research Director",
        "JobRole_Research Scientist":           job_role == "Research Scientist",
        "JobRole_Sales Executive":              job_role == "Sales Executive",
        "JobRole_Sales Representative":         job_role == "Sales Representative",
        "MaritalStatus_Married":                marital_status == "Married",
        "MaritalStatus_Single":                 marital_status == "Single",
        "OverTime_Yes":                         overtime == "Yes",
    }])

if st.button("🔍 Predict Attrition", type="primary", use_container_width=True):
    input_df    = build_input()
    prediction  = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    st.divider()
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error("## ⚠️ High Attrition Risk")
            st.markdown(f"**Probability of leaving:** `{probability[1]*100:.1f}%`")
            st.markdown("This employee is likely to leave. Consider retention strategies.")
        else:
            st.success("## ✅ Low Attrition Risk")
            st.markdown(f"**Probability of staying:** `{probability[0]*100:.1f}%`")
            st.markdown("This employee is likely to stay with the company.")

    with res_col2:
        st.metric("Stay Probability",   f"{probability[0]*100:.1f}%")
        st.metric("Attrit Probability", f"{probability[1]*100:.1f}%")

    with st.expander("📋 View input data sent to model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))
