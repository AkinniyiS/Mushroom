import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

#Initial dataset
file_path = 'C:\\Users\\samue\\Downloads\\mushroom.csv' #Put file path here, zip isnt a problem as long as there is one file inside
data = pd.read_csv(file_path, encoding='utf-8')

#Target is class e or p

print(f"Number of rows before filtering: {data.shape[0]}")

#Cleaning and Quality checking
print(f"Number of duplicate rows: {data.duplicated().sum()}")
data_sample = data.drop_duplicates()

print("\nUnique values per column:")
print(data_sample.nunique())

#chose to remove veil type column based on its vagueness
if "veil-type" in data_sample.columns:
    data_sample = data_sample.drop(columns=["veil-type"])

print("Filling missing values...")
data_sample = data_sample.fillna("missing")

# Encode all categorical data
encoder = LabelEncoder()
for column in data_sample.columns:
    if data_sample[column].dtype == 'object':
        data_sample[column] = encoder.fit_transform(data_sample[column])

corr_matrix = data_sample.corr(numeric_only=True)
target_corr = corr_matrix['class'].sort_values(ascending=False)
top_5 = target_corr.index[1:11]

filtered_corr = corr_matrix.loc[top_5,top_5]

plt.figure(figsize=(8,6))

dataplot = sb.heatmap(filtered_corr,cmap="YlGnBu",annot=True)
plt.title('Top 10 features that correlate to the target')
plt.show()

# Displaying heatmap
mp.show()

dataplot = sb.heatmap(data_sample.corr(numeric_only=True))
mp.show()

#drop all columns that arent the selected features

selected_features = ["ring-type","stem-root","has-ring","cap-color","spore-print-color","gill-spacing","does-bruise-or-bleed","cap-surface","habitat","veil-color"]

X = data_sample[selected_features]
y = data_sample["class"]

#Test split
print("Splitting data into train and test sets.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalize both the training data and the test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

# Perform 5-fold cross-validation on training data and test on testing data
for model_name, model in models.items():
    print(f"\nPerforming 5-Fold Cross-Validation for {model_name}...")
    
    # Cross-validation on training data
    scores = cross_val_score(model, X_train_scaled if model_name in ["KNN", "SVM"] else X_train, y_train, cv=5)
    print(f"{model_name} Training - Mean Accuracy: {scores.mean():.2f}, Std Dev: {scores.std():.2f}")
    
    # Train and evaluate on testing data
    model.fit(X_train_scaled if model_name in ["KNN", "SVM"] else X_train, y_train)
    y_pred = model.predict(X_test_scaled if model_name in ["KNN", "SVM"] else X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Testing Accuracy: {test_accuracy:.2f}")

#Confusion Matrices
for model_name, model in models.items():
    if model_name in ["KNN", "SVM"]:
        X_test_input = X_test_scaled
    else:
        X_test_input = X_test
    
    y_pred = model.predict(X_test_input)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    
    # Plot the confusion matrix
    print(f"Confusion Matrix for {model_name}:")
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show(block=False)

# ROC and AUC curve stuff
plt.figure(figsize=(10, 5))

for model_name, model in models.items():
    if model_name in ["KNN", "SVM"]:  #Scaling still required, same used as in the cross validation section
        X_test_input = X_test_scaled
    else:
        X_test_input = X_test
    
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test_input)[:, 1]
    else:
        y_pred_prob = model.decision_function(X_test_input)
        y_pred_prob = (y_pred_prob - y_pred_prob.min()) / (y_pred_prob.max() - y_pred_prob.min())

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--", label="Random Guess")

# Finalize the plot
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(alpha=0.5)
plt.show(block=False)
