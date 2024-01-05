# %% [markdown]
# # Ahmet Emre Usta
#
# ## 2200765036
#

# %% [markdown]
# # Necessary Imports
#

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# %%
working_dir = "/".join(os.getcwd().split("/")[:-1])
DATASETS_PATH = os.path.join(working_dir, "datasets")
RAW_DATASET_PATH = os.path.join(DATASETS_PATH, "raw")
RAW_DATASET = os.path.join(RAW_DATASET_PATH, "data.csv")

# %% [markdown]
# # EDA
#

# %%
df = pd.read_csv(RAW_DATASET)
df.head()

# %%
df.info()

# %%
# Since first row all zero drop it
df.drop(df.index[0], inplace=True)

# %% [markdown]
# There are some ? in the dataset
#

# %%
df = df.replace("?", np.nan)

# %%
# look if there is any null values
df.isnull().sum()

# %% [markdown]
# 5 unique classes in the dataset. There isn't any class imbalance.
#
# 14 unique user
#

# %%
for col in df.columns:
    print(f"Column: {col}")
    print(df[col].value_counts().to_string())
    print()

# %% [markdown]
# There is some ? in the datasets which we need to manipulate.
#
# - X3: Contains 690 '?' values.
# - Y3: Contains 690 '?' values.
# - Z3: Contains 690 '?' values.
# - X4: Contains 3120 '?' values.
# - Y4: Contains 3120 '?' values.
# - Z4: Contains 3120 '?' values.
# - X5: Contains 13023 '?' values.
# - Y5: Contains 13023 '?' values.
# - Z5: Contains 13023 '?' values.
# - X6: Contains 25848 '?' values.
# - Y6: Contains 25848 '?' values.
# - Z6: Contains 25848 '?' values.
# - X7: Contains 39152 '?' values.
# - Y7: Contains 39152 '?' values.
# - Z7: Contains 39152 '?' values.
# - X8: Contains 47532 '?' values.
# - Y8: Contains 47532 '?' values.
# - Z8: Contains 47532 '?' values.
# - X9: Contains 54128 '?' values.
# - Y9: Contains 54128 '?' values.
# - Z9: Contains 54128 '?' values.
# - X10: Contains 63343 '?' values.
# - Y10: Contains 63343 '?' values.
# - Z10: Contains 63343 '?' values.
# - X11: Contains 78064 '?' values.
# - Y11: Contains 78064 '?' values.
# - Z11: Contains 78064 '?' values.
#
# These columns have missing or uncertain values denoted by '?'.
#
# Since some rows almost completely empty, droping them is make sense to making dataset more robust.
#

# %%
# Drop columns with '?' values
cols_to_drop = [
    "X6",
    "Y6",
    "Z6",
    "X7",
    "Y7",
    "Z7",
    "X8",
    "Y8",
    "Z8",
    "X9",
    "Y9",
    "Z9",
    "X10",
    "Y10",
    "Z10",
    "X11",
    "Y11",
    "Z11",
]
df = df.drop(columns=cols_to_drop)

# %%
df.isnull().sum()

# %%
# convert them to float
df = df.astype(float)

# %%
# fill the empty values with mean of the column
df = df.fillna(df.mean())

# %%
categorical = ["Class", "User"]

fig = plt.figure(figsize=(6, 3))

for i, col in enumerate(categorical):
    value_counts = df[col].value_counts()
    plt.subplot(1, 2, i + 1)
    value_counts.plot(kind="bar")
    plt.title(f"Unique Values Count for {col}")
    plt.xlabel("Unique Values")
    plt.ylabel("Count")

# plt.subplots_adjust(hspace=10)
plt.show()

# %%
fig = plt.figure(figsize=(4, 4))

correlation = df.drop(categorical, axis=1).corr()

# plt.figure(figsize=(6, 6))
sns.heatmap(correlation, annot=True, cbar=True, linewidths=0.5, fmt=".2f")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=45, va="top")

plt.show()

# %% [markdown]
# # Set Dataset
#

# %%
# Divide X and y
X = df.drop(["Class"], axis=1)
y = df["Class"]

# %%
# One-hot encode the 'User' column
X = pd.get_dummies(X, columns=["User"])

# Extract columns to be scaled
columns_to_scale = X.columns.difference(["User"])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply Min-Max scaling to the selected columns
X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

# %%
X.head()

# %%
# use label encoder to encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# %%
np.unique(y)

# %%
model_performances = pd.DataFrame(
    columns=[
        "Model Name",
        "Minimum Cross-Validation Score",
        "Maximum Cross-Validation Score",
        "Mean Cross-Validation Score",
        "Cross-Validation Standard Deviation",
    ]
)

# %% [markdown]
# Given its ability to provide more robust and reliable results compared to the traditional train-test split approach, I prefer utilizing cross-validation for its enhanced performance evaluation.
#

# %% [markdown]
# # Random Forest Classifier
#

# %%
# 5 class classification using cross validation
rf = RandomForestClassifier(random_state=42)

rf_cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print(f"Random Forest Classifier CV Scores: {rf_cv_scores}")
print(f"Random Forest Classifier CV Mean Score: {rf_cv_scores.mean()}")
print(f"Random Forest Classifier CV Standard Deviation: {rf_cv_scores.std()}")

# %%
new_row = {
    "Model Name": "Random Forest",
    "Minimum Cross-Validation Score": rf_cv_scores.min(),
    "Maximum Cross-Validation Score": rf_cv_scores.max(),
    "Mean Cross-Validation Score": rf_cv_scores.mean(),
    "Cross-Validation Standard Deviation": rf_cv_scores.std(),
}
model_performances.loc[len(model_performances)] = new_row

# %% [markdown]
# # Naive Bayes
#

# %%
# 5 class classification using cross validation with Naive Bayes
nb = GaussianNB()

nb_cv_scores = cross_val_score(nb, X, y, cv=5, scoring="accuracy")
print(f"Naive Bayes Classifier CV Scores: {nb_cv_scores}")
print(f"Naive Bayes Classifier CV Mean Score: {nb_cv_scores.mean()}")
print(f"Naive Bayes Classifier CV Standard Deviation: {nb_cv_scores.std()}")

# %%
new_row_nb = {
    "Model Name": "Naive Bayes",
    "Minimum Cross-Validation Score": nb_cv_scores.min(),
    "Maximum Cross-Validation Score": nb_cv_scores.max(),
    "Mean Cross-Validation Score": nb_cv_scores.mean(),
    "Cross-Validation Standard Deviation": nb_cv_scores.std(),
}
model_performances.loc[len(model_performances)] = new_row_nb

# %% [markdown]
# # XGBoost
#

# %%
# 5 class classification using cross validation with XGBoost
xgb_model = XGBClassifier(random_state=42)

xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring="accuracy")
print(f"XGBoost Classifier CV Scores: {xgb_cv_scores}")
print(f"XGBoost Classifier CV Mean Score: {xgb_cv_scores.mean()}")
print(f"XGBoost Classifier CV Standard Deviation: {xgb_cv_scores.std()}")

# %%
new_row_xgb = {
    "Model Name": "XGBoost",
    "Minimum Cross-Validation Score": xgb_cv_scores.min(),
    "Maximum Cross-Validation Score": xgb_cv_scores.max(),
    "Mean Cross-Validation Score": xgb_cv_scores.mean(),
    "Cross-Validation Standard Deviation": xgb_cv_scores.std(),
}
model_performances.loc[len(model_performances)] = new_row_xgb

# %% [markdown]
# # Comprassion
#

# %%
model_performances.head()

# %%
# Data preparation
models = model_performances["Model Name"]
mean_scores = model_performances["Mean Cross-Validation Score"]
std_deviation = model_performances["Cross-Validation Standard Deviation"]

# Plotting
fig, ax = plt.subplots(figsize=(4, 4))

ax.bar(models, mean_scores, yerr=std_deviation, capsize=5, color="skyblue")
ax.set_ylabel("Mean Cross-Validation Score")
ax.set_title("Performance Comparison of Different Models")

# Show the plot
plt.show()

# %% [markdown]
# The Random Forest Classifier demonstrates a robust performance with a mean cross-validation score of approximately 73.04%, showcasing consistent accuracy across different folds, albeit with a standard deviation of 3.63%.
#
# On the other hand, the Naive Bayes Classifier exhibits a lower mean cross-validation score of around 24.50%, indicating comparatively weaker predictive performance. The standard deviation of 5.34% suggests notable variability in performance across folds.
#
# XGBoost Classifier performs competitively with a mean cross-validation score of approximately 71.06%, demonstrating stable accuracy. The standard deviation of 4.50% indicates moderate variability in performance across different folds.
#
# In summary, the Random Forest and XGBoost classifiers outperform the Naive Bayes Classifier in terms of predictive accuracy, with the Random Forest Classifier showing the highest and most consistent performance. The standard deviations provide insights into the stability and reliability of these models across various cross-validation folds.
#

# %%
# remove the unnecessary variables
del df, X, y, rf, nb, xgb_model, model_performances
