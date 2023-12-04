import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import dtreeviz
import category_encoders as ce
from sklearn import tree
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns
import matplotlib as mpl
import numpy as np

def add_noise(df):
    random_state = np.random.RandomState()
    n_samples, n_features = df.shape
    x = np.concatenate([df, random_state.randn(n_samples, 200*n_features)], axis=1)
    return x

column_names = ['date', 'precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
features_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']

dataset_clean = pd.read_csv(r"C:\Users\sengj\Downloads\seattle-weather.csv", header=0, names=column_names)
#y = add_noise(dataset_clean)
dataset_10 = pd.read_csv(r"C:\Users\sengj\OneDrive\Documents\seattle-weather-10.csv", header=0, names=column_names)
dataset_20 = pd.read_csv(r"C:\Users\sengj\OneDrive\Documents\seattle-weather-20.csv", header=0, names=column_names)
dataset_30 = pd.read_csv(r"C:\Users\sengj\OneDrive\Documents\seattle-weather-30.csv", header=0, names=column_names)

# Features
x_clean = dataset_clean[features_cols]
x_10 = dataset_10[features_cols]
x_20 = dataset_20[features_cols]
x_30 = dataset_30[features_cols]

# Target variable
y_clean = dataset_clean.weather
y_10 = dataset_10.weather
y_20 = dataset_20.weather
y_30 = dataset_30.weather

# 70% training and 30% test
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(x_clean, y_clean, test_size=0.3, random_state=1)
X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(x_10, y_10, test_size=0.3, random_state=1)
X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(x_20, y_20, test_size=0.3, random_state=1)
X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(x_30, y_30, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf_clean = DecisionTreeClassifier()
clf_10 = DecisionTreeClassifier()
clf_20 = DecisionTreeClassifier()
clf_30 = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf_clean = clf_clean.fit(X_train_clean, y_train_clean)
clf_10 = clf_10.fit(X_train_10, y_train_10)
clf_20 = clf_20.fit(X_train_20, y_train_20)
clf_30 = clf_30.fit(X_train_30, y_train_30)

# Predict the response for test dataset
y_pred_clean = clf_clean.predict(X_test_clean)
y_pred_10 = clf_10.predict(X_test_10)
y_pred_20 = clf_20.predict(X_test_20)
y_pred_30 = clf_30.predict(X_test_30)

# Accuracy of the classify 
print("Accuracy:", metrics.accuracy_score(y_test_clean, y_pred_clean))
print("Accuracy - 10%:", metrics.accuracy_score(y_test_10, y_pred_10))
print("Accuracy - 20%:", metrics.accuracy_score(y_test_20, y_pred_20))
print("Accuracy - 30%:", metrics.accuracy_score(y_test_30, y_pred_30))

# Confusion Matrix
cm_clean = confusion_matrix(y_test_clean, y_pred_clean)
display_clean = ConfusionMatrixDisplay(confusion_matrix=cm_clean, display_labels=clf_clean.classes_)
display_clean.plot()
plt.show()

cm_10 = confusion_matrix(y_test_10, y_pred_10)
display_10 = ConfusionMatrixDisplay(confusion_matrix=cm_10, display_labels=clf_10.classes_)
display_10.plot()
plt.show()

cm_20 = confusion_matrix(y_test_20, y_pred_20)
display_20 = ConfusionMatrixDisplay(confusion_matrix=cm_20, display_labels=clf_20.classes_)
display_20.plot()
plt.show()

cm_30 = confusion_matrix(y_test_30, y_pred_30)
display_30 = ConfusionMatrixDisplay(confusion_matrix=cm_30, display_labels=clf_30.classes_)
display_30.plot()
plt.show()


#Visualize Trees
plt.figure(figsize=(12,12))
tree.plot_tree(clf_clean, feature_names=features_cols, class_names=clf_clean.classes_.tolist(), filled=True, max_depth=3, fontsize=2)
plt.savefig(r"C:\Users\sengj\Pictures\DT_clean", dpi=350)

plt.figure(figsize=(12,12))
tree.plot_tree(clf_10, feature_names=features_cols, class_names=clf_10.classes_.tolist(), filled=True, max_depth=3, fontsize=2)
plt.savefig(r"C:\Users\sengj\Pictures\DT_10", dpi=350)

plt.figure(figsize=(12,12))
tree.plot_tree(clf_20, feature_names=features_cols, class_names=clf_20.classes_.tolist(), filled=True, max_depth=3, fontsize=2)
plt.savefig(r"C:\Users\sengj\Pictures\DT_20", dpi=350)

plt.figure(figsize=(12,12))
tree.plot_tree(clf_30, feature_names=features_cols, class_names=clf_30.classes_.tolist(), filled=True, max_depth=3, fontsize=2)
plt.savefig(r"C:\Users\sengj\Pictures\DT_30", dpi=350)