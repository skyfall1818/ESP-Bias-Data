import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree
from matplotlib import pyplot as plt
import random
from sklearn.impute import SimpleImputer


def remove_datapoints(df, percent_entries):
    ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    for row, col in random.sample(ix, int(round(percent_entries * len(ix)))):
        df.iat[row, col] = pd.NA
    return df

X_Validation = None
Y_Validation = None

def get_accuracy(df, image_name, baseline=False):
    feature_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
    X = weather_data[feature_cols]
    Y = weather_data['weather']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plt.figure(figsize=(12,12))
    _ = tree.plot_tree(clf,
                       feature_names=feature_cols,
                       class_names=clf.classes_.tolist(),
                       filled=True)
    plt.savefig(image_name, dpi=600)

    return metrics.accuracy_score(y_test, y_pred)

feature_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
weather_data = pd.read_csv("seattle-weather.csv")
# indecies = pd.factorize(weather_data['weather'])[1].values
# weather_data['weather'] = pd.factorize(weather_data['weather'])[0]

weather_data_with_holes = weather_data.copy(deep=True)
weather_data_with_holes = remove_datapoints(weather_data_with_holes, .8)

print("Accuracy for untoched dataset:", get_accuracy(weather_data, "untouched.png"))

print("Accuracy for dataset with holes:", get_accuracy(weather_data_with_holes, "holes.png"))

weather_data_fixed_holes = weather_data_with_holes.copy(deep=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

weather_data_fixed_holes[feature_cols] = pd.DataFrame(imputer.fit_transform(weather_data_fixed_holes[feature_cols]), columns=feature_cols)

print("Accuracy for dataset with holes fixed:", get_accuracy(weather_data_fixed_holes, "fixed_holes.png"))