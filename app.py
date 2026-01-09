import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Page config
st.set_page_config(page_title="Health Insurance Cost Prediction", layout="wide")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    df = df.dropna()
    return df

@st.cache_data
def prepare_model(df):
    df_ml = df.copy()
    le = LabelEncoder()
    df_ml['gender'] = le.fit_transform(df_ml['gender'])
    df_ml['diabetic'] = le.fit_transform(df_ml['diabetic'])
    df_ml['smoker'] = le.fit_transform(df_ml['smoker'])
    df_ml['region'] = le.fit_transform(df_ml['region'])
    
    X = df_ml[['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker', 'region']]
    y = df_ml['claim']
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model, le

# Main app
st.title("🏥 Health Insurance Cost Prediction")

df = load_data()
model, le = prepare_model(df)

# Sidebar for prediction
st.sidebar.header("Predict Insurance Cost")
age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
bp = st.sidebar.slider("Blood Pressure", 80, 140, 90)
diabetic = st.sidebar.selectbox("Diabetic", ["No", "Yes"])
children = st.sidebar.slider("Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

if st.sidebar.button("Predict Cost"):
    # Encode inputs
    gender_enc = 1 if gender == "male" else 0
    diabetic_enc = 1 if diabetic == "Yes" else 0
    smoker_enc = 1 if smoker == "Yes" else 0
    region_map = {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}
    region_enc = region_map[region]
    
    prediction = model.predict([[age, gender_enc, bmi, bp, diabetic_enc, children, smoker_enc, region_enc]])
    st.sidebar.success(f"Predicted Cost: ${prediction[0]:.2f}")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "📋 Dataset", "🤖 Model Info"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
        st.pyplot(fig)
        
        st.subheader("Smoker vs Cost")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='smoker', y='claim', ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("BMI vs Cost")
        fig, ax = plt.subplots()
        ax.scatter(df['bmi'], df['claim'], alpha=0.6)
        st.pyplot(fig)
        
        st.subheader("Region Distribution")
        fig, ax = plt.subplots()
        df['region'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

with tab2:
    st.subheader("Insurance Dataset")
    st.dataframe(df)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name="insurance_data.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Model Information")
    st.write("**Model:** Random Forest Regressor")
    st.write(f"**Dataset Size:** {len(df)} records")
    st.write("**Features:** Age, Gender, BMI, Blood Pressure, Diabetic, Children, Smoker, Region")