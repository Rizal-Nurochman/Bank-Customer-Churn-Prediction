import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import utils

data_path = 'Data/Churn_Modelling.csv'

st.set_page_config("Zyntra - Bank Customer Churn Predictor", layout="wide")

st.title("Zyntra - Bank Customer Churn Predictor")
st.markdown("""
Welcome to **Zyntra**!  
This application predicts bank customer churn.
""")

data = utils.load_data(data_path)
model, scaler, encoder, num_cols = utils.load_assets()

X = data.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])
y = data['Exited']

cat_cols = X[['Geography', 'Gender']]
encoded = encoder.transform(cat_cols)
encoded_cols = encoder.get_feature_names_out(['Geography', 'Gender'])
df_encoded = pd.DataFrame(encoded, columns=encoded_cols)

X = X.drop(columns=['Geography', 'Gender'])
X = pd.concat([X, df_encoded], axis=1)

X[num_cols] = X[num_cols].astype(float)
X[num_cols] = scaler.transform(X[num_cols])

final_columns = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain', 'Gender_Male'
]

X = X[final_columns]

tab1, tab2, tab3 = st.tabs([
    "Eksplorasi Data",
    "Prediksi Churn Customer",
    "Evaluasi Model"
])

with tab1:
    st.header("Eksplorasi Dataset")
    st.dataframe(data.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Dimensi:", data.shape)
    with col2:
        st.write("Distribusi Target:")
        st.write(data['Exited'].value_counts())

with tab2:
    st.header("Form Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 92, 35)
        tenure = st.number_input("Tenure", 0, 10, 5)

    with col2:
        balance = st.number_input("Balance", value=0.0, step=100.0)
        num_products = st.number_input("Num Products", 1, 4, 1)

        has_card_label = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_label = st.selectbox("Is Active Member?", ["Yes", "No"])

        has_card = 1 if has_card_label == "Yes" else 0
        is_active = 1 if is_active_label == "Yes" else 0

        salary = st.number_input("Salary", value=100000.0, step=1000.0)

    submit = st.button("Prediksi", type="primary")

if submit:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [salary]
    })

    cat_cols = input_data[['Geography', 'Gender']]
    encoded = encoder.transform(cat_cols)
    encoded_cols = encoder.get_feature_names_out(['Geography', 'Gender'])
    df_encoded = pd.DataFrame(encoded, columns=encoded_cols)

    input_data = input_data.drop(columns=['Geography', 'Gender'])
    input_data = pd.concat([input_data, df_encoded], axis=1)

    input_data[num_cols] = input_data[num_cols].astype(float)
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    input_data = input_data[final_columns]

    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0]

    st.divider()

    if prediction[0] == 1:
        st.error("⚠️ CHURN")
        st.info(f"Probabilitas Churn: {proba[1]*100:.2f}%")
    else:
        st.success("✅ LOYAL")
        st.info(f"Probabilitas Loyal: {proba[0]*100:.2f}%")

with tab3:
    st.header("Evaluasi Kinerja Model")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"**Akurasi Model: {accuracy * 100:.2f}%**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)

    with col2:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            annot_kws={"size": 10}
        )

        ax.set_ylabel('Aktual')
        ax.set_xlabel('Prediksi')

        st.pyplot(fig)