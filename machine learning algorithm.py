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

weather_file = pd.read_csv("seattle-weather.csv")
indecies = pd.factorize(weather_file['weather'])[1].values
weather_file['weather'] = pd.factorize(weather_file['weather'])[0]

test_file = pd.read_csv("Test_List.csv")
test_file['weather'] = pd.factorize(test_file['weather'])[0] 

def get_accuracy(df, image_name, baseline=False):
    global X_Validation
    global Y_Validation
    global indecies
    global weather_file
    X_train = df[feature_cols]
    y_train = df['weather'].map(lambda x: indecies[int(x)])
    X_t = test_file[feature_cols]
    Y_t = test_file['weather'].map(lambda x: indecies[int(x)])
    _, X_test, _, y_test = train_test_split(X_t, Y_t, test_size=0.25, random_state=1)
    # if baseline:
    #     X_Base, X_Validation, Y_Base, Y_Validation = train_test_split(X, Y, test_size=0.25, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plt.figure(figsize=(16, 9))
    _ = tree.plot_tree(clf,
                       feature_names=feature_cols,
                       class_names=clf.classes_.tolist(),
                       filled=True)
    plt.savefig(image_name, dpi=600)

    # base_y_pred = clf.predict(X_Validation)
    # print("Accuracy Overall: ", metrics.accuracy_score(Y_Validation, base_y_pred))
    return metrics.accuracy_score(y_test, y_pred)


weather_data = pd.read_csv("control_file.csv")
weather_data['weather'] = pd.factorize(weather_data['weather'])[0]
print("Accuracy for control dataset:", get_accuracy(weather_data, "control.png", baseline=True))

percentage_data = pd.read_csv("percentage.csv",header=None)
for i in range(1,5):
    percent = float(percentage_data.values[i-1][0])
    file_reader = pd.read_csv("bias_file" + str(i) + ".csv")
    file_reader['weather'] = pd.factorize(file_reader['weather'])[0]
    print("Accuracy for bias_file" , i , ".csv with" , percent, "% bias: ",
          get_accuracy(file_reader, "bias_" + str(i) ) )
