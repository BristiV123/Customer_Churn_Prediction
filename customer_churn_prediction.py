import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load Dataset
try:
    data = pd.read_csv("customer_churn.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: customer_churn.csv file not found.")
    exit()

# Display first 5 rows
print(data.head())

# Check dataset information
print(data.info())

# Check missing values
print(data.isnull().sum())

# Dataset Information
print(data.info())

# Missing Values
print(data.isnull().sum())

# Statistical Summary
print(data.describe())

#Data Cleaning & Encoding
# Fill missing values
data=data.ffill()

# Convert categorical columns into numbers
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for column in data.select_dtypes(include="object").columns:
    data[column] = encoder.fit_transform(data[column])

print("Data Cleaning and Encoding Completed!")
print(data.head())

#Train-Test Split and Model Traning
# Features and Target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully!")

#Train-Test Split and Logistic Regression Model
# Split dataset
from sklearn.model_selection import train_test_split

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model Trained Successfully!")

# Features and Target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Customer Churn Confusion Matrix")
plt.savefig("IMAGES/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Customer Churn Count
plt.figure(figsize=(6,4))

data["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.savefig("IMAGES/churn_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Monthly Charges vs Churn
plt.figure(figsize=(6,4))

plt.scatter(data["MonthlyCharges"], data["Churn"])

plt.title("Monthly Charges vs Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Churn")

plt.savefig("IMAGES/monthlycharges_vs_churn.png", dpi=300, bbox_inches="tight")
plt.show()