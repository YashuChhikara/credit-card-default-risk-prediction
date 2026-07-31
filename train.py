import pandas as pd

df = pd.read_csv("data/dataset.csv", header=1)

print("Rows and columns:", df.shape)
print(df.head())


df = df.rename(
    columns={"default payment next month": "default_next_month"}
)

df = df.drop(columns="ID")

# Collapse undocumented/sentinel codes into the existing "other" category
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # 4 = "others"
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                # 3 = "other"

print("\nDefault counts:")
print(df["default_next_month"].value_counts())

print("\nDefault proportions:")
print(df["default_next_month"].value_counts(normalize=True))

print("\nTotal missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print(df.info())
print(df.describe(include="all").T)
print(df.nunique().sort_values())


categorical_columns = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

for column in categorical_columns:
    print(f"\n{column} value counts:")
    print(df[column].value_counts().sort_index())


for column in categorical_columns:
    print(f"\nDefault rate by {column}:")

    default_rate = (
        df.groupby(column)["default_next_month"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print(default_rate.round(2))

financial_columns = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

print("\nAverage financial values by default outcome:")
print(
    df.groupby("default_next_month")[financial_columns]
    .mean()
    .T
    .round(2)
)


import matplotlib.pyplot as plt

pay0_default_rate = (
    df.groupby("PAY_0")["default_next_month"]
    .mean()
    .mul(100)
)

"""plt.figure(figsize=(10, 5))
plt.bar(pay0_default_rate.index.astype(str), pay0_default_rate.values)

plt.title("Default Rate by Most Recent Repayment Status")
plt.xlabel("PAY_0 repayment status")
plt.ylabel("Default rate (%)")
plt.show()"""


numeric_correlation = (
    df.select_dtypes(include="number")
    .corr()["default_next_month"]
    .sort_values(ascending=False)
)

print("\nCorrelation with default:")
print(numeric_correlation)


from sklearn.model_selection import train_test_split

X = df.drop(columns="default_next_month")
y = df["default_next_month"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

print("\nTraining default rate:", y_train.mean().round(4))
print("Test default rate:", y_test.mean().round(4))


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

numeric_features = [
    column for column in X.columns
    if column not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]
)

logistic_model.fit(X_train, y_train)

print("Logistic regression model trained successfully.")


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

y_pred = logistic_model.predict(X_test)
y_prob = logistic_model.predict_proba(X_test)[:, 1]

print("Classification report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Default", "Default"]
)

"""display.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()"""


from sklearn.ensemble import RandomForestClassifier

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

print("Random Forest classification report:")
print(classification_report(y_test, rf_pred))

print("Random Forest ROC-AUC:", round(roc_auc_score(y_test, rf_prob), 3))


feature_names = random_forest_model.named_steps[
    "preprocessor"
].get_feature_names_out()

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": random_forest_model.named_steps["model"].feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 most important features:")
print(feature_importance.head(15))

top_features = feature_importance.head(15).sort_values("importance")

"""plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.title("Top 15 Random Forest Default-Risk Drivers")
plt.xlabel("Feature importance")
plt.show()"""


from sklearn.model_selection import RandomizedSearchCV

parameter_options = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}

rf_search = RandomizedSearchCV(
    estimator=random_forest_model,
    param_distributions=parameter_options,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_search.fit(X_train, y_train)

best_random_forest = rf_search.best_estimator_

print("Best settings:")
print(rf_search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(round(rf_search.best_score_, 3))


best_rf_prob = best_random_forest.predict_proba(X_test)[:, 1]

print("Tuned Random Forest ROC-AUC:")
print(round(roc_auc_score(y_test, best_rf_prob), 3))



precision, recall, pr_thresholds = precision_recall_curve(y_test, best_rf_prob)

f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)

best_f1_idx = f1_scores.argmax()
print(f"\nBest F1 threshold: {pr_thresholds[best_f1_idx]:.3f}")
print(f"  precision={precision[best_f1_idx]:.3f}  recall={recall[best_f1_idx]:.3f}  f1={f1_scores[best_f1_idx]:.3f}")

print("\nthreshold  precision  recall     f1")
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    idx = (pr_thresholds >= t).argmax()
    p, r = precision[idx], recall[idx]
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"{pr_thresholds[idx]:.2f}       {p:.3f}      {r:.3f}     {f1:.3f}")


final_threshold = 0.424

best_rf_pred = (best_rf_prob >= final_threshold).astype(int)

print(f"\nTuned Random Forest report at threshold = {final_threshold}:")
print(classification_report(y_test, best_rf_pred))

final_cm = confusion_matrix(y_test, best_rf_pred)

ConfusionMatrixDisplay(
    confusion_matrix=final_cm,
    display_labels=["No Default", "Default"]
).plot()

"""plt.title("Tuned Random Forest Confusion Matrix")
plt.show()"""


from pathlib import Path
import joblib

Path("models").mkdir(exist_ok=True)

model_package = {
    "model": best_random_forest,
    "threshold": final_threshold,
    "feature_columns": list(X.columns),
    "model_name": "Tuned Random Forest"
}

joblib.dump(
    model_package,
    "models/credit_default_model.joblib"
)

print("\nModel saved to models/credit_default_model.joblib")
print(f"Saved threshold: {model_package['threshold']}")
