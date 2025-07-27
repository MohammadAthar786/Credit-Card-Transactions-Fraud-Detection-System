import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(layout="wide")

# Load your final model and pipeline
with open("fraud_model_smote.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessing_pipeline_smote.pkl", "rb") as f:
    pipeline = pickle.load(f)


# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("realistic_transactions.csv")
    return df


df = load_data()

# Layout Sidebar
st.sidebar.title("Fraud Detection Dashboard")
pages = ["Overview", "Fraud Analysis", "Model Performance", "Live Prediction"]
choice = st.sidebar.radio("Go to", pages)

# Page 1: Overview
if choice == "Overview":
    st.title("🔍 Fraud Detection Overview")
    st.markdown("### Dataset Snapshot")
    st.dataframe(df.head())

    st.markdown("### Class Distribution")
    class_dist = df["is_fraudulent"].value_counts().rename({0: "Not Fraud", 1: "Fraud"})
    fig = px.pie(
        values=class_dist.values, names=class_dist.index, title="Fraud vs Not Fraud"
    )
    st.plotly_chart(fig)

    st.markdown("### Feature Summary")
    st.write(df.describe(include="all"))

# Page 2: Fraud Analysis
elif choice == "Fraud Analysis":
    st.title("📊 Fraud Analysis")
    fraud = df[df["is_fraudulent"] == 1]
    legit = df[df["is_fraudulent"] == 0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Fraud by Merchant")
        fig = px.histogram(fraud, x="merchant", title="Fraud Count per Merchant")
        st.plotly_chart(fig)

    with col2:
        st.markdown("#### Fraud by Location")
        fig = px.histogram(
            fraud, x="customer_location", title="Fraud Count per Location"
        )
        st.plotly_chart(fig)

    st.markdown("#### Fraud Amount Distribution")
    fig = px.box(fraud, y="transaction_amount", title="Fraud Transaction Amounts")
    st.plotly_chart(fig)

# Page 3: Model Performance
elif choice == "Model Performance":
    st.title("📈 Model Evaluation")
    X = df.drop("is_fraudulent", axis=1)
    y = df["is_fraudulent"]

    # Transform using pipeline
    X_processed = pipeline.transform(X)

    # Predict
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)[:, 1]

    # Classification Report
    st.markdown("### Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
    )
    st.pyplot(fig)

    # ROC Curve
    st.markdown("### ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC Curve (AUC = {roc_auc:.2f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
    )
    st.plotly_chart(fig)

# Page 4: Live Prediction
elif choice == "Live Prediction":
    st.title("🎯 Live Fraud Prediction")

    st.markdown("Enter transaction details below:")

    customer_location = st.selectbox(
        "Customer Location", df["customer_location"].unique()
    )
    merchant = st.selectbox("Merchant", df["merchant"].unique())
    transaction_purpose = st.selectbox(
        "Transaction Purpose", df["transaction_purpose"].unique()
    )
    card_type = st.selectbox("Card Type", df["card_type"].unique())
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
    fraud_score = st.slider("Fraud Score (0-1)", 0.0, 1.0, step=0.01)

    input_data = pd.DataFrame(
        [
            {
                "customer_location": customer_location,
                "merchant": merchant,
                "transaction_purpose": transaction_purpose,
                "card_type": card_type,
                "transaction_amount": transaction_amount,
                "fraud_score": fraud_score,
            }
        ]
    )

    if st.button("Predict"):
        input_processed = pipeline.transform(input_data)
        prediction = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0][1]

        st.markdown("### Prediction Result:")
        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Transaction Looks Legitimate (Confidence: {1 - prob:.2f})")
