import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree
from matplotlib import pyplot as plt
import random
from sklearn.impute import SimpleImputer

X_Validation = None
Y_Validation = None
feature_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
indecies = []

def remove_datapoints(df, percent_entries):
    ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    for row, col in random.sample(ix, int(round(percent_entries * len(ix)))):
        df.iat[row, col] = pd.NA
    return df


def get_accuracy(df, image_name, baseline=False):
    global X_Validation
    global Y_Validation
    global indecies
    X = df[feature_cols]
    Y = df['weather'].map(lambda x: indecies[int(x)])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
    # if baseline:
    #     X_Base, X_Validation, Y_Base, Y_Validation = train_test_split(X, Y, test_size=0.25, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plt.figure(figsize=(12, 12))
    _ = tree.plot_tree(clf,
                       feature_names=feature_cols,
                       class_names=clf.classes_.tolist(),
                       filled=True)
    plt.savefig(image_name, dpi=600)

    # base_y_pred = clf.predict(X_Validation)
    # print("Accuracy Overall: ", metrics.accuracy_score(Y_Validation, base_y_pred))
    return metrics.accuracy_score(y_test, y_pred)


weather_data = pd.read_csv("seattle-weather.csv")

indecies = pd.factorize(weather_data['weather'])[1].values

weather_data['weather'] = pd.factorize(weather_data['weather'])[0]

print("Accuracy for untoched dataset:", get_accuracy(weather_data, "untouched.png", baseline=True))
for i in range(1,10):
    weather_data_with_holes = weather_data.copy(deep=True)
    weather_data_with_holes = remove_datapoints(weather_data_with_holes, i / 10)

    # print("Accuracy for dataset with holes:", get_accuracy(weather_data_with_holes, "holes.png"))

    weather_data_fixed_holes = weather_data_with_holes.copy(deep=True)

    relevant_cols = ['precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    weather_data_fixed_holes[relevant_cols] = pd.DataFrame(imputer.fit_transform(weather_data_fixed_holes[relevant_cols]),
                                                          columns=relevant_cols)
    weather_data_fixed_holes = weather_data_fixed_holes.round({'weather': 0})
    print("Accuracy for dataset with", str(i*10), "% holes fixed:", get_accuracy(weather_data_fixed_holes, "".join(["fixed_holes_", str(i * 10), "_percent.png"]) ) )
