import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

# Suppress future warnings for cleaner console output
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load dataset
df = pd.read_excel(r"C:\Users\Pranjal Bhardwaj\Downloads\track4_mid_day_meal_procurement.xlsx")
print("--- Dataset Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())
print("\nTotal Missing Values:", df.isnull().sum().sum())

# Clean column names
df.columns = df.columns.str.strip()

# 2. Data Cleaning & Preprocessing
# Parse dates with mixed formats, coercing errors
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Clean total_cost
def clean_cost(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    val_clean = re.sub(r'[^\d.]', '', val_str.replace(',', ''))
    try:
        return float(val_clean)
    except:
        return np.nan

df["clean_cost"] = df["total_cost"].apply(clean_cost)

# Clean quantity
def clean_quantity(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    val_clean = re.sub(r'[^\d.]', '', val_str.replace(',', ''))
    try:
        return float(val_clean)
    except:
        return np.nan

df["clean_quantity"] = df["quantity"].apply(clean_quantity)

# Standardize grain types into main categories
def standardize_grain(val):
    val_str = str(val).lower()
    if 'wheat' in val_str or 'gehun' in val_str or 'atta' in val_str:
        return 'Wheat'
    elif 'rice' in val_str or 'chawal' in val_str:
        return 'Rice'
    elif 'dal' in val_str or 'pulses' in val_str or 'lentils' in val_str:
        return 'Dal'
    elif 'oil' in val_str or 'tel' in val_str:
        return 'Oil'
    else:
        return 'Other'

df["grain_category"] = df["grain_type"].apply(standardize_grain)

# Clean target variable: payment_status into Binary Classification (1 = Paid/Cleared, 0 = Pending/Due)
def clean_payment_status(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if val_str in ['paid', 'cleared']:
        return 1
    elif val_str in ['pending', 'due']:
        return 0
    else:
        return np.nan

df["target"] = df["payment_status"].apply(clean_payment_status)

# Remove rows without valid target or date
df.dropna(subset=["target", "date"], inplace=True)

# Feature Engineering from date
df["hour"] = df["date"].dt.hour
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["year"] = df["date"].dt.year

# 3. Exploratory Data Analysis & Visualizations
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="target", hue="target", palette="Set2", legend=False)
plt.title("Distribution of Target Variable (Payment Status)")
plt.xlabel("Payment Status (0 = Pending/Due, 1 = Paid/Cleared)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation Analysis
corr_cols = ["clean_quantity", "clean_cost", "hour", "day", "month", "year", "target"]
df_corr = df[corr_cols].fillna(df[corr_cols].median())
corr = df_corr.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 4. Machine Learning Classification Setup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Encode categorical columns
le_vendor = LabelEncoder()
le_grain = LabelEncoder()
df["vendor_encoded"] = le_vendor.fit_transform(df["vendor_name"].astype(str))
df["grain_encoded"] = le_grain.fit_transform(df["grain_category"].astype(str))

X = df[['clean_quantity', 'clean_cost', 'vendor_encoded', 'grain_encoded', 'hour', 'day', 'month', 'day_of_week', 'year']]
y = df['target']

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Crucial for KNN, SVM, and Logistic Regression)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Import Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine (SVM)": SVC(random_state=42)
}

results = []

for name, model in models.items():
    # Use scaled data for Logistic Regression, KNN, and SVM; unscaled is fine for Trees/Naive Bayes
    if name in ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"]:
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
    else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })
    
    print(f"\n--- {name} Results ---")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

# 5. Model Comparison & Summary
results_df = pd.DataFrame(results)
print("\n--- Model Comparison Summary ---")
print(results_df)

best_model = results_df.loc[results_df["Accuracy"].idxmax()]
print("\nBest Model:", best_model["Model"])
print("Best Accuracy:", best_model["Accuracy"])

# 6. Visualization: Model Accuracy Comparison
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy", hue="Model", palette="viridis", legend=False)
plt.title("Classification Model Accuracy Comparison")
plt.xlabel("Classifier Model")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
