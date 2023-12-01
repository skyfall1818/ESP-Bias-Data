
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
def remove_datapoints(df, num_entries):
    df.iat[2,2 ] = np.nan


def get_accuracy(df):
    X = weather_data['precipitation', 'temp_max', 'temp_min', 'wind']
    Y = weather_data['weather']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return metrics.accuracy_score(y_test, y_pred)


weather_data = pd.read_csv("seattle-weather.csv.csv", header=None)

weather_data_with_holes = remove_datapoints(weather_data)

print("Accuracy for untoched dataset:", get_accuracy(weather_data))

print("Accuracy for dataset with holes:", get_accuracy(weather_data_with_holes))