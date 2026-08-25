import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

from pathlib import Path
import joblib


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/dataset.csv", header=1)

print("Rows and columns:", df.shape)
print(df.head())


# ============================================================
# 2. BASIC CLEANING
# ============================================================

df = df.rename(
    columns={"default payment next month": "default_next_month"}
)

# ID is only an identifier and should not be used as a predictor
df = df.drop(columns="ID")


# Collapse undocumented codes into the existing "other" category
df["EDUCATION"] = df["EDUCATION"].replace({
    0: 4,
    5: 4,
    6: 4
})  # 4 = "others"

df["MARRIAGE"] = df["MARRIAGE"].replace({
    0: 3
})  # 3 = "other"


# ============================================================
# 3. BASIC DATA CHECKS
# ============================================================

print("\nDefault counts:")
print(df["default_next_month"].value_counts())

print("\nDefault proportions:")
print(
    df["default_next_month"]
    .value_counts(normalize=True)
    .round(4)
)

print("\nTotal missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all").T)

print("\nNumber of unique values:")
print(df.nunique().sort_values())


# ============================================================
# 4. CATEGORICAL / DISCRETE EDA
# ============================================================

pay_features = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

nominal_features = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

categorical_columns = nominal_features + pay_features


for column in categorical_columns:
    print(f"\n{column} value counts:")
    print(
        df[column]
        .value_counts()
        .sort_index()
    )


for column in categorical_columns:
    print(f"\nDefault rate by {column}:")

    default_rate = (
        df.groupby(column)["default_next_month"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
    )

    print(default_rate.round(2))


# ============================================================
# 5. FINANCIAL VARIABLE EDA
# ============================================================

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


# ============================================================
# 6. EXAMPLE EDA PLOT: PAY_0 DEFAULT RATE
# ============================================================

pay0_default_rate = (
    df.groupby("PAY_0")["default_next_month"]
    .mean()
    .mul(100)
)

# plt.figure(figsize=(10, 5))
# plt.bar(
#     pay0_default_rate.index.astype(str),
#     pay0_default_rate.values
# )
# plt.title("Default Rate by Most Recent Repayment Status")
# plt.xlabel("PAY_0 repayment status")
# plt.ylabel("Default rate (%)")
# plt.show()


# ============================================================
# 7. CORRELATION FOR QUANTITATIVE VARIABLES
# ============================================================

# Pearson correlation is shown only for genuinely quantitative
# variables rather than arbitrary category codes.

correlation_columns = financial_columns + ["default_next_month"]

numeric_correlation = (
    df[correlation_columns]
    .corr()["default_next_month"]
    .sort_values(ascending=False)
)

print("\nCorrelation with default:")
print(numeric_correlation)


# ============================================================
# 8. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(columns="default_next_month")
y = df["default_next_month"]


# ============================================================
# 9. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First hold out 20% as the final test set.

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split the remaining 80% into:
# 60% training and 20% validation overall.

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("\nTraining data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)

print("\nTraining default rate:", round(y_train.mean(), 4))
print("Validation default rate:", round(y_val.mean(), 4))
print("Test default rate:", round(y_test.mean(), 4))


# ============================================================
# 10. LOGISTIC REGRESSION PREPROCESSING
# ============================================================

# PAY variables are discrete repayment-status codes.
#
# For logistic regression, they are one-hot encoded so that the
# model does not impose a single linear log-odds effect across
# codes such as -2, -1, 0, 1, 2, etc.

logistic_categorical_features = nominal_features + pay_features

logistic_numeric_features = [
    column
    for column in X.columns
    if column not in logistic_categorical_features
]


logistic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            logistic_numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            logistic_categorical_features
        )
    ]
)


# ============================================================
# 11. LOGISTIC REGRESSION BASELINE
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            logistic_preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


logistic_model.fit(X_train, y_train)

print("\nLogistic regression model trained successfully.")


# Validation performance
logistic_val_pred = logistic_model.predict(X_val)

logistic_val_prob = (
    logistic_model
    .predict_proba(X_val)[:, 1]
)

print("\nLogistic Regression validation report:")
print(
    classification_report(
        y_val,
        logistic_val_pred
    )
)

print(
    "Logistic Regression validation ROC-AUC:",
    round(
        roc_auc_score(
            y_val,
            logistic_val_prob
        ),
        3
    )
)


# ============================================================
# 12. RANDOM FOREST PREPROCESSING
# ============================================================

# For the Random Forest:
#
# SEX, EDUCATION and MARRIAGE are nominal categories
# and are one-hot encoded.
#
# PAY_* variables retain their ordered integer representation
# because trees can exploit meaningful ordered splits such as
# PAY_0 <= 0 versus PAY_0 > 0.
#
# Scaling is NOT required for tree-based models.

rf_categorical_features = nominal_features

rf_numeric_features = [
    column
    for column in X.columns
    if column not in rf_categorical_features
]


rf_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            rf_numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            rf_categorical_features
        )
    ]
)


# ============================================================
# 13. BASELINE RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            rf_preprocessor
        ),
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


rf_val_pred = random_forest_model.predict(X_val)

rf_val_prob = (
    random_forest_model
    .predict_proba(X_val)[:, 1]
)


print("\nRandom Forest validation report:")
print(
    classification_report(
        y_val,
        rf_val_pred
    )
)

print(
    "Random Forest validation ROC-AUC:",
    round(
        roc_auc_score(
            y_val,
            rf_val_prob
        ),
        3
    )
)


# ============================================================
# 14. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

parameter_options = {
    "model__n_estimators": [200, 300, 500],

    "model__max_depth": [
        5,
        10,
        15,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


# Stratified CV preserves the default / non-default proportion
# in each fold.

cv_strategy = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


rf_search = RandomizedSearchCV(
    estimator=random_forest_model,
    param_distributions=parameter_options,
    n_iter=10,
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,
    random_state=42,
    verbose=1
)


# Hyperparameter search uses TRAINING DATA ONLY.
rf_search.fit(X_train, y_train)


best_random_forest = rf_search.best_estimator_


print("\nBest Random Forest settings:")
print(rf_search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(
    round(
        rf_search.best_score_,
        3
    )
)


# ============================================================
# 15. VALIDATION PERFORMANCE OF TUNED RANDOM FOREST
# ============================================================

best_rf_val_prob = (
    best_random_forest
    .predict_proba(X_val)[:, 1]
)

print("\nTuned Random Forest validation ROC-AUC:")

print(
    round(
        roc_auc_score(
            y_val,
            best_rf_val_prob
        ),
        3
    )
)


# ============================================================
# 16. CHOOSE CLASSIFICATION THRESHOLD USING VALIDATION DATA
# ============================================================

precision, recall, pr_thresholds = precision_recall_curve(
    y_val,
    best_rf_val_prob
)


# precision_recall_curve returns one more precision/recall
# value than threshold values, hence [:-1].

f1_scores = (
    2
    * precision[:-1]
    * recall[:-1]
    /
    (
        precision[:-1]
        + recall[:-1]
        + 1e-12
    )
)


best_f1_idx = f1_scores.argmax()

final_threshold = pr_thresholds[best_f1_idx]


print(
    f"\nBest validation F1 threshold: "
    f"{final_threshold:.3f}"
)

print(
    f"precision={precision[best_f1_idx]:.3f}  "
    f"recall={recall[best_f1_idx]:.3f}  "
    f"f1={f1_scores[best_f1_idx]:.3f}"
)


# Display the precision-recall trade-off at several thresholds.

print("\nthreshold  precision  recall     f1")

for t in [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]:

    # Find the stored PR threshold closest to t.
    idx = abs(pr_thresholds - t).argmin()

    p = precision[idx]
    r = recall[idx]

    f1 = (
        2 * p * r / (p + r)
        if (p + r) > 0
        else 0
    )

    print(
        f"{pr_thresholds[idx]:.2f}       "
        f"{p:.3f}      "
        f"{r:.3f}     "
        f"{f1:.3f}"
    )


# ============================================================
# 17. FINAL TEST-SET EVALUATION
# ============================================================

# The test set has not been used for:
# - model fitting
# - hyperparameter tuning
# - threshold selection
#
# It is now used for final evaluation only.

best_rf_test_prob = (
    best_random_forest
    .predict_proba(X_test)[:, 1]
)


best_rf_test_pred = (
    best_rf_test_prob >= final_threshold
).astype(int)


print(
    f"\nFinal Tuned Random Forest report "
    f"at threshold = {final_threshold:.3f}:"
)

print(
    classification_report(
        y_test,
        best_rf_test_pred
    )
)


print(
    "Final test ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            best_rf_test_prob
        ),
        3
    )
)


# ============================================================
# 18. FINAL CONFUSION MATRIX
# ============================================================

final_cm = confusion_matrix(
    y_test,
    best_rf_test_pred
)


display = ConfusionMatrixDisplay(
    confusion_matrix=final_cm,
    display_labels=[
        "No Default",
        "Default"
    ]
)

# display.plot()
# plt.title("Tuned Random Forest Confusion Matrix")
# plt.show()


# ============================================================
# 19. FEATURE IMPORTANCE FROM TUNED RANDOM FOREST
# ============================================================

feature_names = (
    best_random_forest
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


feature_importance = pd.DataFrame({
    "feature": feature_names,

    "importance":
        best_random_forest
        .named_steps["model"]
        .feature_importances_
})


feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)


print("\nTop 15 most important features:")
print(
    feature_importance
    .head(15)
)


top_features = (
    feature_importance
    .head(15)
    .sort_values("importance")
)


# plt.figure(figsize=(10, 6))
# plt.barh(
#     top_features["feature"],
#     top_features["importance"]
# )
# plt.title("Top 15 Random Forest Default-Risk Drivers")
# plt.xlabel("Feature importance")
# plt.show()


# ============================================================
# 20. SAVE FINAL MODEL PACKAGE
# ============================================================

Path("models").mkdir(
    exist_ok=True
)


model_package = {
    "model": best_random_forest,

    "threshold": float(final_threshold),

    "feature_columns": list(X.columns),

    "model_name": "Tuned Random Forest"
}


joblib.dump(
    model_package,
    "models/credit_default_model.joblib"
)


print(
    "\nModel saved to "
    "models/credit_default_model.joblib"
)

print(
    f"Saved threshold: "
    f"{model_package['threshold']:.3f}"
)
