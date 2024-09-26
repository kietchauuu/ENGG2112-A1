import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

"""
Loads the data, preprocesses and uses KNN, logistical regression and random forest model
It will take a long time to finish training the models so we need to boost performance somehow
Or we can delegate each model into a separate python file
"""
# Load data
clinical_data = pd.read_csv('datasets/clinical_dataset.csv')
lifestyle_data = pd.read_csv('datasets/lifestyle_dataset.csv')

# View basic information
print(clinical_data.head())
print(lifestyle_data.head())

# Preprocessing the lifestyle dataset
Xlifestyle = lifestyle_data.drop('Heart Attack Risk', axis=1)
Ylifestyle = lifestyle_data['Heart Attack Risk']

# Preprocessing the clinical dataset
Xclinical = clinical_data.drop('output', axis=1)
Yclinical = clinical_data['output']

# Convert categorical columns to numeric using one-hot encoding before splitting
Xlifestyle_encoded = pd.get_dummies(Xlifestyle, drop_first=True)
Xclinical_encoded = pd.get_dummies(Xclinical, drop_first=True)

# Re-split the datasets after encoding
Xlife_train, Xlife_test, ylife_train, ylife_test = train_test_split(Xlifestyle_encoded, Ylifestyle, test_size=0.2, random_state=42)
Xclin_train, Xclin_test, yclin_train, yclin_test = train_test_split(Xclinical_encoded, Yclinical, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
Xlife_train_scaled = scaler.fit_transform(Xlife_train)
Xlife_test_scaled = scaler.transform(Xlife_test)
Xclin_train_scaled = scaler.fit_transform(Xclin_train)
Xclin_test_scaled = scaler.transform(Xclin_test)

# -------------MODELLING----------------

# Initialise models
logistic_model = LogisticRegression()
knn_model = KNeighborsClassifier()
random_forest_model = RandomForestClassifier()

# Perform 5-fold cross-validation for each model ----- LIFESTYLE DATA
life_logistic_cv = cross_val_score(logistic_model, Xlife_train_scaled, ylife_train, cv=5, scoring='accuracy')
life_knn_cv = cross_val_score(knn_model, Xlife_train_scaled, ylife_train, cv=5, scoring='accuracy')
life_rf_cv = cross_val_score(random_forest_model, Xlife_train_scaled, ylife_train, cv=5, scoring='accuracy')

# Perform 5-fold cross-validation for each model ----- CLINICAL DATA
clin_logistic_cv = cross_val_score(logistic_model, Xclin_train_scaled, yclin_train, cv=5, scoring='accuracy')
clin_knn_cv = cross_val_score(knn_model, Xclin_train_scaled, yclin_train, cv=5, scoring='accuracy')
clin_rf_cv = cross_val_score(random_forest_model, Xclin_train_scaled, yclin_train, cv=5, scoring='accuracy')

# Print cross-validation results
print("Accuracy results for Lifestyle data\n")
print(f'Logistic Regression CV Accuracy: {life_logistic_cv.mean()}')
print(f'KNN CV Accuracy: {life_knn_cv.mean()}')
print(f'Random Forest CV Accuracy: {life_rf_cv.mean()}')

print("\nAccuracy results for Clinical data\n")
print(f'Logistic Regression CV Accuracy: {clin_logistic_cv.mean()}')
print(f'KNN CV Accuracy: {clin_knn_cv.mean()}')
print(f'Random Forest CV Accuracy: {clin_rf_cv.mean()}')

# Test set evaluation
logistic_model = LogisticRegression()
logistic_model.fit(Xlife_train_scaled, ylife_train)
y_pred_life_log = logistic_model.predict(Xlife_test_scaled)
print(f'Logistic Regression Test Accuracy (Lifestyle): {accuracy_score(ylife_test, y_pred_life_log)}')

logistic_model.fit(Xclin_train_scaled, yclin_train)
y_pred_clin_log = logistic_model.predict(Xclin_test_scaled)
print(f'Logistic Regression Test Accuracy (Clinical): {accuracy_score(yclin_test, y_pred_clin_log)}')