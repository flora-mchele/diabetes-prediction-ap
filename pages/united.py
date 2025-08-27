# pages/united.py
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
# from db import c, conn # This import is not needed for this page.

# --- Authentication Check ---
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("Please log in to access this page.")
    st.info("Redirecting to login...")
    st.switch_page("main.py")

# --- Page Config ---
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    h1 {color: #2E86C1; text-align: center;}
    .stButton>button {background-color: #2E86C1; color: white;}
    .stSidebar {background-color: #d6eaf8;}
    </style>
""", unsafe_allow_html=True)

# --- App Title and User Info ---
st.title("🩺 Diabetes Prediction Dashboard")
st.markdown("Professional health analytics for diabetes prediction and modeling")
st.markdown(f"**Logged in as:** {st.session_state.get('username', 'Guest')}")

# --- Logout Button ---
if st.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.success("You have been logged out.")
    st.switch_page("main.py")


# --- Sidebar for Settings ---
st.sidebar.header("⚙ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
model_option = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "KNN", "Random Forest"])

if model_option == "KNN":
    n_neighbors = st.sidebar.slider("Number of neighbors (KNN)", 1, 20, 5)
elif model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Number of trees (Random Forest)", 10, 200, 100)

# --- Tabs Interface ---
tabs = st.tabs([
    "📄 Dataset",
    "🚀 Train Model",
    "🌡 Visualizations",
    "🧪 Predict Patient"
])

# Use a session state to store the processed dataframe
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None

# --- Dataset Tab ---
with tabs[0]:
    st.subheader("📄 Dataset Preview & Info")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_processed = df.copy() # Store a copy for later use
        st.dataframe(df.head())
        st.write("Dataset Info:")
        st.dataframe(pd.DataFrame(df.info()))
        st.write("Missing values check:")
        st.dataframe(df.isnull().sum())
    else:
        st.info("Upload a CSV dataset in the sidebar to proceed.")


# --- Train Model Tab (Combined Data Cleaning) ---
with tabs[1]:
    st.subheader("🚀 Model Training & Evaluation")
    if st.session_state.df_processed is not None:
        df_clean = st.session_state.df_processed.copy()
        
        # --- Handle Missing Values ---
        st.write("### Data Cleaning")
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64','int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        st.success("✅ Missing values handled.")
        
        # --- Interactive Outlier Removal ---
        st.write("### Interactive Outlier Removal")
        numeric_cols = df_clean.select_dtypes(include=['float64','int64']).columns.tolist()
        
        iqr_factors = {}
        for col in numeric_cols:
            iqr_factors[col] = st.slider(f"IQR factor for {col}", 1.0, 3.0, 1.5, 0.1, key=f"iqr_{col}")
        
        if st.button("Apply Cleaning & Prepare Data"):
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - iqr_factors[col]*IQR
                upper = Q3 + iqr_factors[col]*IQR
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            st.session_state.df_processed = df_clean
            st.success("✅ Outliers removed.")
            st.dataframe(st.session_state.df_processed.head())
            
        st.write("---")
        
        if 'Outcome' in st.session_state.df_processed.columns:
            st.write("### Model Training")
            X = st.session_state.df_processed.drop("Outcome", axis=1)
            y = st.session_state.df_processed["Outcome"]

            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler

            # Select model
            if model_option == "Logistic Regression":
                model = LogisticRegression()
            elif model_option == "KNN":
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
            else:
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            if st.button("Train Model", key="train_button"):
                model.fit(X_train_scaled, y_train)
                st.session_state.model = model
                y_pred = model.predict(X_test_scaled)
                st.success("✅ Model trained successfully!")

                # Metrics Cards
                accuracy = accuracy_score(y_test, y_pred)
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{accuracy*100:.2f}%")
                col2.write("Confusion Matrix")
                col2.write(confusion_matrix(y_test, y_pred))

                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                
                # Save model and scaler
                model_filename = "diabetes_model.pkl"
                scaler_filename = "scaler.pkl"
                joblib.dump(st.session_state.model, model_filename)
                joblib.dump(st.session_state.scaler, scaler_filename)
                st.success(f"Model saved as {model_filename} and scaler as {scaler_filename}")
        else:
            st.error("The dataset must contain an 'Outcome' column for training.")
            
    else:
        st.info("Upload dataset to train the model.")

# --- Visualizations Tab ---
with tabs[2]:
    st.subheader("🌡 Exploratory Data Analysis")
    if st.session_state.df_processed is not None:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(st.session_state.df_processed.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        if 'Outcome' in st.session_state.df_processed.columns:
            st.write("Outcome Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(x='Outcome', data=st.session_state.df_processed, ax=ax)
            st.pyplot(fig)
    else:
        st.info("Upload dataset to see visualizations.")


# --- Prediction Tab ---
with tabs[3]:
    st.subheader("🧪 Predict Diabetes for New Patient")
    if st.session_state.df_processed is not None:
        input_data = {}
        for col in st.session_state.df_processed.drop("Outcome", axis=1, errors='ignore').columns:
            val = st.number_input(f"Enter {col}", value=float(st.session_state.df_processed[col].mean()))
            input_data[col] = val
        
        if st.button("Predict"):
            if "model" in st.session_state and "scaler" in st.session_state:
                loaded_model = st.session_state.model
                loaded_scaler = st.session_state.scaler
                
                input_df = pd.DataFrame([input_data])
                
                # Ensure columns are in the same order as training
                training_cols = list(st.session_state.df_processed.drop("Outcome", axis=1, errors='ignore').columns)
                input_df = input_df[training_cols]
                
                input_scaled = loaded_scaler.transform(input_df)
                prediction = loaded_model.predict(input_scaled)
                
                if prediction[0] == 1:
                    st.error("Prediction: The patient is likely to have diabetes. 😔")
                else:
                    st.success("Prediction: The patient is unlikely to have diabetes. 😊")
            else:
                st.warning("Train the model first before predicting.")
    else:
        st.info("Upload dataset to make predictions.")