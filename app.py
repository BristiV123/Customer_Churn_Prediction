import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .high-risk {
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }

    .medium-risk {
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }

    .low-risk {
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-powered customer retention and churn analysis</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("customer_churn.csv")

    df.columns = df.columns.astype(str).str.strip()

    for col in df.columns:

        if (
            df[col].dtype == "object"
            or str(df[col].dtype).startswith("string")
        ):

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

    return df


try:

    data = load_data()

except Exception as e:

    st.error("❌ customer_churn.csv could not be loaded.")

    st.code(str(e))

    st.info(
        "Make sure customer_churn.csv and app.py are in the same folder."
    )

    st.stop()


# ============================================================
# CHECK CHURN COLUMN
# ============================================================

if "Churn" not in data.columns:

    st.error("❌ 'Churn' column was not found.")

    st.write(
        "Available columns:"
    )

    st.write(
        list(data.columns)
    )

    st.stop()


# ============================================================
# CLEAN CHURN
# ============================================================

data["Churn"] = (
    data["Churn"]
    .astype(str)
    .str.strip()
)


# ============================================================
# SHOW ORIGINAL CHURN VALUES
# ============================================================

churn_values = data["Churn"].value_counts(
    dropna=False
)


if data["Churn"].nunique() < 2:

    st.error(
        "❌ Churn column must contain at least two classes."
    )

    st.write(churn_values)

    st.stop()


# ============================================================
# TARGET ENCODING
# ============================================================

target_encoder = LabelEncoder()

data["Churn"] = target_encoder.fit_transform(
    data["Churn"]
)


# ============================================================
# PREPARE DATA FOR ML
# ============================================================

processed_data = data.copy()

feature_encoders = {}


for col in processed_data.columns:

    if col == "Churn":
        continue

    if (
        processed_data[col].dtype == "object"
        or str(processed_data[col].dtype).startswith("string")
    ):

        encoder = LabelEncoder()

        processed_data[col] = encoder.fit_transform(
            processed_data[col].astype(str)
        )

        feature_encoders[col] = encoder


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

for col in processed_data.columns:

    processed_data[col] = pd.to_numeric(
        processed_data[col],
        errors="coerce"
    )


processed_data = processed_data.fillna(0)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = processed_data.drop(
    columns=["Churn"]
)

y = processed_data["Churn"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

except ValueError:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )


# ============================================================
# MODEL CREATION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

decision_tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)


# ============================================================
# MODEL TRAINING
# ============================================================

try:

    logistic_model.fit(
        X_train,
        y_train
    )

    decision_tree_model.fit(
        X_train,
        y_train
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

except Exception as e:

    st.error("❌ Model training failed.")

    st.exception(e)

    st.stop()


# ============================================================
# MODEL METRICS FUNCTION
# ============================================================

def get_metrics(model):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        auc = roc_auc_score(
            y_test,
            probabilities
        )

    except Exception:

        auc = 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    }


# ============================================================
# CALCULATE METRICS
# ============================================================

logistic_metrics = get_metrics(
    logistic_model
)

decision_tree_metrics = get_metrics(
    decision_tree_model
)

random_forest_metrics = get_metrics(
    random_forest_model
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "📊 Dashboard",
        "📋 Dataset Analysis",
        "📈 Visualizations",
        "🤖 Model Performance",
        "🆚 Model Comparison",
        "📌 Feature Importance",
        "👥 Customer Segmentation",
        "🔮 Churn Prediction",
        "📜 Prediction History",
        "📂 Bulk CSV Prediction"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.header("📊 Customer Churn Dashboard")

    total_customers = len(data)

    churned_customers = int(
        data["Churn"].sum()
    )

    retained_customers = (
        total_customers
        - churned_customers
    )

    churn_rate = (
        churned_customers
        / total_customers
        * 100
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Customers",
        total_customers
    )

    col2.metric(
        "❌ Churned Customers",
        churned_customers
    )

    col3.metric(
        "✅ Retained Customers",
        retained_customers
    )

    col4.metric(
        "📉 Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader("📌 Customer Dataset")

    st.dataframe(
        data,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📊 Churn Distribution")

    churn_counts = data["Churn"].value_counts()

    labels = []

    values = []

    for value in sorted(
        churn_counts.index
    ):

        if value == 0:
            labels.append("No Churn")
        else:
            labels.append("Churn")

        values.append(
            churn_counts[value]
        )

    fig, ax = plt.subplots()

    ax.bar(
        labels,
        values
    )

    ax.set_title(
        "Customer Churn Distribution"
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    st.pyplot(fig)


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "📋 Dataset Analysis":

    st.header("📋 Dataset Analysis")

    # --------------------------------------------------------
    # SHAPE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows",
            data.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            data.shape[1]
        )

    st.divider()

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.subheader(
        "🔍 Missing Values"
    )

    missing = data.isnull().sum()

    missing_df = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Values": missing.values
        }
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    st.subheader(
        "🔤 Data Types"
    )

    dtype_df = pd.DataFrame(
        {
            "Column": data.columns,
            "Data Type": [
                str(dtype)
                for dtype in data.dtypes
            ]
        }
    )

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.subheader(
        "📊 Statistical Summary"
    )

    st.dataframe(
        data.describe(
            include="all"
        ).T,
        use_container_width=True
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

elif page == "📈 Visualizations":

    st.header(
        "📈 Data Visualizations"
    )

    # --------------------------------------------------------
    # CHURN
    # --------------------------------------------------------

    st.subheader(
        "1️⃣ Churn Distribution"
    )

    counts = data["Churn"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        ["No Churn", "Churn"],
        [
            counts.get(0, 0),
            counts.get(1, 0)
        ]
    )

    ax.set_title(
        "Churn Distribution"
    )

    ax.set_ylabel(
        "Customers"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_columns = [
        col
        for col in numeric_columns
        if col != "Churn"
    ]

    if len(numeric_columns) > 0:

        st.subheader(
            "2️⃣ Numerical Feature vs Churn"
        )

        selected_column = st.selectbox(
            "Select Feature",
            numeric_columns
        )

        no_churn_data = data[
            data["Churn"] == 0
        ][selected_column].dropna()

        churn_data = data[
            data["Churn"] == 1
        ][selected_column].dropna()

        fig, ax = plt.subplots()

        ax.boxplot(
            [
                no_churn_data,
                churn_data
            ],
            labels=[
                "No Churn",
                "Churn"
            ]
        )

        ax.set_title(
            f"{selected_column} vs Churn"
        )

        ax.set_ylabel(
            selected_column
        )

        st.pyplot(fig)

    else:

        st.info(
            "No numerical columns are available."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.header(
        "🤖 Model Performance"
    )

    selected_model = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    if selected_model == "Logistic Regression":

        model = logistic_model

    elif selected_model == "Decision Tree":

        model = decision_tree_model

    else:

        model = random_forest_model


    metrics = get_metrics(
        model
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.2f}"
    )

    c2.metric(
        "Precision",
        f"{metrics['Precision']:.2f}"
    )

    c3.metric(
        "Recall",
        f"{metrics['Recall']:.2f}"
    )

    c4.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.2f}"
    )

    c5.metric(
        "ROC-AUC",
        f"{metrics['ROC-AUC']:.2f}"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "📌 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        ["No Churn", "Churn"]
    )

    ax.set_yticklabels(
        ["No Churn", "Churn"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "📄 Classification Report"
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Churn",
            "Churn"
        ],
        zero_division=0
    )

    st.code(
        report
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "🆚 Model Comparison":

    st.header(
        "🆚 Model Comparison"
    )

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                **logistic_metrics
            },
            {
                "Model": "Decision Tree",
                **decision_tree_metrics
            },
            {
                "Model": "Random Forest",
                **random_forest_metrics
            }
        ]
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.subheader(
        "📊 Accuracy Comparison"
    )

    fig, ax = plt.subplots()

    ax.bar(
        comparison_df["Model"],
        comparison_df["Accuracy"]
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        "Model Accuracy Comparison"
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader(
        "📈 ROC Curve"
    )

    fig, ax = plt.subplots()

    models = [
        (
            "Logistic Regression",
            logistic_model
        ),
        (
            "Decision Tree",
            decision_tree_model
        ),
        (
            "Random Forest",
            random_forest_model
        )
    ]

    for name, model in models:

        try:

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            fpr, tpr, _ = roc_curve(
                y_test,
                probabilities
            )

            ax.plot(
                fpr,
                tpr,
                label=name
            )

        except Exception:

            pass

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve Comparison"
    )

    ax.legend()

    st.pyplot(fig)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "📌 Feature Importance":

    st.header(
        "📌 Feature Importance"
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance":
                random_forest_model.feature_importances_
        }
    )

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    fig, ax = plt.subplots()

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    ax.invert_yaxis()

    st.pyplot(fig)


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "👥 Customer Segmentation":

    st.header(
        "👥 Customer Segmentation"
    )

    segment_data = data.copy()

    # --------------------------------------------------------
    # CUSTOMER SEGMENT
    # --------------------------------------------------------

    if "Tenure" in segment_data.columns:

        segment_data["Customer Segment"] = np.select(
            [
                segment_data["Tenure"] <= 12,
                segment_data["Tenure"] <= 36,
                segment_data["Tenure"] > 36
            ],
            [
                "New Customer",
                "Regular Customer",
                "Loyal Customer"
            ],
            default="Regular Customer"
        )

    else:

        segment_data["Customer Segment"] = (
            "General Customer"
        )


    # --------------------------------------------------------
    # VALUE SEGMENT
    # --------------------------------------------------------

    if "MonthlyCharges" in segment_data.columns:

        low_limit = segment_data[
            "MonthlyCharges"
        ].quantile(0.33)

        high_limit = segment_data[
            "MonthlyCharges"
        ].quantile(0.66)

        segment_data["Value Segment"] = np.select(
            [
                segment_data["MonthlyCharges"] <= low_limit,
                segment_data["MonthlyCharges"] <= high_limit,
                segment_data["MonthlyCharges"] > high_limit
            ],
            [
                "Low Value",
                "Medium Value",
                "High Value"
            ],
            default="Medium Value"
        )

    else:

        segment_data["Value Segment"] = (
            "General Value"
        )


    # --------------------------------------------------------
    # SEGMENT DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📊 Customer Segment Distribution"
    )

    segment_counts = (
        segment_data["Customer Segment"]
        .value_counts()
    )

    st.dataframe(
        segment_counts.to_frame(
            "Customers"
        ),
        use_container_width=True
    )

    fig, ax = plt.subplots()

    ax.bar(
        segment_counts.index,
        segment_counts.values
    )

    ax.set_ylabel(
        "Customers"
    )

    ax.set_title(
        "Customer Segments"
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # VALUE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "💰 Value Segment Distribution"
    )

    value_counts = (
        segment_data["Value Segment"]
        .value_counts()
    )

    st.dataframe(
        value_counts.to_frame(
            "Customers"
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # CHURN BY SEGMENT
    # --------------------------------------------------------

    st.subheader(
        "📉 Churn by Customer Segment"
    )

    churn_segment = pd.crosstab(
        segment_data["Customer Segment"],
        segment_data["Churn"]
    )

    churn_segment = churn_segment.rename(
        columns={
            0: "No Churn",
            1: "Churn"
        }
    )

    st.dataframe(
        churn_segment,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    selected_segment = st.selectbox(
        "Select Customer Segment",
        segment_data[
            "Customer Segment"
        ].unique()
    )

    filtered_data = segment_data[
        segment_data["Customer Segment"]
        == selected_segment
    ]

    st.dataframe(
        filtered_data,
        use_container_width=True
    )


    # --------------------------------------------------------
    # RETENTION STRATEGY
    # --------------------------------------------------------

    st.subheader(
        "💡 Retention Strategy"
    )

    if selected_segment == "New Customer":

        st.info(
            "Use onboarding support, welcome offers, "
            "early engagement and customer assistance."
        )

    elif selected_segment == "Regular Customer":

        st.info(
            "Use personalized offers, loyalty rewards "
            "and engagement campaigns."
        )

    elif selected_segment == "Loyal Customer":

        st.info(
            "Use loyalty programs, premium services "
            "and long-term relationship benefits."
        )

    else:

        st.info(
            "Use personalized customer engagement."
        )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    segment_csv = (
        segment_data
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Customer Segmentation",
        segment_csv,
        "customer_segments.csv",
        "text/csv"
    )


# ============================================================
# CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.header(
        "🔮 Customer Churn Prediction"
    )

    st.info(
        "Enter customer details to calculate churn risk."
    )

    input_values = {}


    # --------------------------------------------------------
    # CUSTOMER ID
    # --------------------------------------------------------

    if "CustomerID" in data.columns:

        customer_id = st.text_input(
            "Customer ID",
            value="CUST001"
        )

    else:

        customer_id = "Customer"


    # --------------------------------------------------------
    # CREATE INPUT FORM
    # --------------------------------------------------------

    for col in X.columns:

        if col == "CustomerID":
            continue

        original_column = data[col]

        # Categorical column
        if col in feature_encoders:

            options = (
                original_column
                .astype(str)
                .unique()
                .tolist()
            )

            input_values[col] = st.selectbox(
                col,
                options
            )

        # Numeric column
        else:

            numeric_values = pd.to_numeric(
                original_column,
                errors="coerce"
            )

            default_value = numeric_values.median()

            if pd.isna(default_value):

                default_value = 0

            input_values[col] = st.number_input(
                col,
                value=float(default_value)
            )


    st.divider()


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Churn Risk",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [input_values]
        )


        # ----------------------------------------------------
        # ENCODE CATEGORICAL INPUT
        # ----------------------------------------------------

        for col in input_df.columns:

            if col in feature_encoders:

                encoder = feature_encoders[col]

                value = str(
                    input_df.loc[0, col]
                )

                if value in encoder.classes_:

                    input_df[col] = encoder.transform(
                        [value]
                    )

                else:

                    input_df[col] = 0


        # ----------------------------------------------------
        # ADD MISSING FEATURES
        # ----------------------------------------------------

        for col in X.columns:

            if col not in input_df.columns:

                input_df[col] = 0


        # ----------------------------------------------------
        # KEEP SAME COLUMN ORDER
        # ----------------------------------------------------

        input_df = input_df[
            X.columns
        ]


        # ----------------------------------------------------
        # NUMERIC
        # ----------------------------------------------------

        input_df = input_df.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(0)


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = random_forest_model.predict(
            input_df
        )[0]


        try:

            probability = (
                random_forest_model
                .predict_proba(
                    input_df
                )[0][1]
            )

        except Exception:

            probability = float(
                prediction
            )


        risk_score = round(
            probability * 100,
            2
        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if risk_score >= 70:

            risk_level = "High Risk"

            recommendation = (
                "Contact the customer immediately "
                "and provide a personalized retention offer."
            )

        elif risk_score >= 40:

            risk_level = "Medium Risk"

            recommendation = (
                "Send personalized offers, "
                "loyalty benefits and engagement messages."
            )

        else:

            risk_level = "Low Risk"

            recommendation = (
                "Continue regular engagement "
                "and maintain good customer service."
            )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Result"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

        c2.metric(
            "Risk Level",
            risk_level
        )

        c3.metric(
            "Prediction",
            "Will Churn"
            if prediction == 1
            else "Will Stay"
        )


        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if risk_level == "High Risk":

            st.error(
                "🚨 HIGH CHURN RISK"
            )

        elif risk_level == "Medium Risk":

            st.warning(
                "⚠️ MEDIUM CHURN RISK"
            )

        else:

            st.success(
                "✅ LOW CHURN RISK"
            )


        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "💡 Retention Recommendation"
        )

        st.info(
            recommendation
        )


        # ----------------------------------------------------
        # WHY CUSTOMER MAY CHURN
        # ----------------------------------------------------

        st.subheader(
            "🔍 Why Customer May Churn"
        )

        reasons = []


        if "Tenure" in input_values:

            if input_values["Tenure"] <= 12:

                reasons.append(
                    "Customer has relatively low tenure."
                )


        if "MonthlyCharges" in input_values:

            if input_values["MonthlyCharges"] >= 70:

                reasons.append(
                    "Monthly charges are relatively high."
                )


        if "Contract" in input_values:

            contract_value = str(
                input_values["Contract"]
            ).lower()

            if contract_value == "month-to-month":

                reasons.append(
                    "Customer is on a month-to-month contract."
                )


        if len(reasons) == 0:

            reasons.append(
                "No strong rule-based churn factor was identified."
            )


        for reason in reasons:

            st.write(
                "• " + reason
            )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history_record = input_values.copy()

        history_record["CustomerID"] = (
            customer_id
        )

        history_record["Risk Score"] = (
            risk_score
        )

        history_record["Risk Level"] = (
            risk_level
        )

        history_record["Prediction"] = (
            "Churn"
            if prediction == 1
            else "No Churn"
        )

        history_record["Recommendation"] = (
            recommendation
        )

        st.session_state.prediction_history.append(
            history_record
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📜 Prediction History":

    st.header(
        "📜 Prediction History"
    )

    if len(
        st.session_state.prediction_history
    ) == 0:

        st.info(
            "No predictions have been made yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )


        history_csv = (
            history_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            "⬇️ Download Prediction History",
            history_csv,
            "prediction_history.csv",
            "text/csv"
        )


        if st.button(
            "🗑️ Clear History"
        ):

            st.session_state.prediction_history = []

            st.rerun()


# ============================================================
# BULK CSV PREDICTION
# ============================================================

elif page == "📂 Bulk CSV Prediction":

    st.header(
        "📂 Bulk Customer Churn Prediction"
    )

    st.write(
        "Upload a CSV file containing customer information."
    )


    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            bulk_data = pd.read_csv(
                uploaded_file
            )

            bulk_data.columns = (
                bulk_data.columns
                .astype(str)
                .str.strip()
            )


            st.subheader(
                "📌 Uploaded Dataset"
            )

            st.dataframe(
                bulk_data,
                use_container_width=True
            )


            processed_bulk = bulk_data.copy()


            # ------------------------------------------------
            # ENCODE CATEGORICAL FEATURES
            # ------------------------------------------------

            for col in X.columns:

                if col not in processed_bulk.columns:

                    processed_bulk[col] = 0

                    continue


                if col in feature_encoders:

                    encoder = feature_encoders[col]

                    mapping = dict(
                        zip(
                            encoder.classes_,
                            encoder.transform(
                                encoder.classes_
                            )
                        )
                    )

                    processed_bulk[col] = (
                        processed_bulk[col]
                        .astype(str)
                        .map(mapping)
                        .fillna(0)
                    )


            # ------------------------------------------------
            # SAME COLUMN ORDER
            # ------------------------------------------------

            processed_bulk = processed_bulk[
                X.columns
            ]


            # ------------------------------------------------
            # NUMERIC
            # ------------------------------------------------

            processed_bulk = processed_bulk.apply(
                pd.to_numeric,
                errors="coerce"
            ).fillna(0)


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            bulk_predictions = (
                random_forest_model.predict(
                    processed_bulk
                )
            )


            try:

                bulk_probabilities = (
                    random_forest_model
                    .predict_proba(
                        processed_bulk
                    )[:, 1]
                )

            except Exception:

                bulk_probabilities = (
                    bulk_predictions
                    .astype(float)
                )


            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            result_data = bulk_data.copy()


            result_data[
                "Churn Probability"
            ] = (
                bulk_probabilities * 100
            ).round(2)


            result_data[
                "Risk Score"
            ] = (
                bulk_probabilities * 100
            ).round(2)


            result_data[
                "Risk Level"
            ] = np.where(
                bulk_probabilities >= 0.70,
                "High Risk",
                np.where(
                    bulk_probabilities >= 0.40,
                    "Medium Risk",
                    "Low Risk"
                )
            )


            result_data[
                "Prediction"
            ] = np.where(
                bulk_predictions == 1,
                "Churn",
                "No Churn"
            )


            st.subheader(
                "📊 Bulk Prediction Results"
            )

            st.dataframe(
                result_data,
                use_container_width=True
            )


            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            result_csv = (
                result_data
                .to_csv(index=False)
                .encode("utf-8")
            )


            st.download_button(
                "⬇️ Download Churn Prediction Report",
                result_csv,
                "churn_prediction_report.csv",
                "text/csv"
            )


        except Exception as e:

            st.error(
                "❌ Error while processing the uploaded CSV."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Developed by Bristi Ray | "
    "Customer Churn Prediction using Machine Learning"
)