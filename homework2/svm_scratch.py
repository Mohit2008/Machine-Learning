import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from math import exp
np.random.seed(24)
names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]


def do_cleaning(inc_data):
    data_clean = inc_data.dropna()
    print("Cleaned data set has entries equal to ", np.shape(data_clean))
    data_clean_X_all = data_clean.iloc[:, 0:14]
    data_clean_X = data_clean_X_all.loc[:, data_clean_X_all.dtypes == "int64"]
    data_clean_Y = data_clean.iloc[:, -1]
    print("Cleaned data set has features equal to ", np.shape(data_clean_X))
    print("Cleaned data set has labels equal to ", np.shape(data_clean_Y))
    return data_clean_X, data_clean_Y


def create_train_test_validate(data_clean_X, data_clean_Y):
    X_train, data_test_X, Y_train, data_test_Y = train_test_split(data_clean_X, data_clean_Y, test_size=0.2,
                                                                  random_state=42)
    X_test, X_validate, Y_test, Y_validate = train_test_split(data_test_X, data_test_Y, test_size=0.5, random_state=42)
    return X_train, Y_train, X_test, Y_test, X_validate, Y_validate


def encode_labels(data_clean_Y):
    encoded_Y = list()
    for i, row in enumerate(list(data_clean_Y)):
        row = str(row).strip().replace('.', "")

        if row == "<=50K":
            encoded_Y.append(-1)
        else:
            encoded_Y.append(1)
    return pd.Series(encoded_Y)


def sigmoid_func(instance, param):
    val = param[0]
    for i in range(len(instance)):
        val += param[i + 1] * instance[i]
    return val


def find_params(train_x, train_y, reg_param, epoch):
    params = [0.0 for w in range(np.shape(train_x)[1]+1)]
    for epoch in range(epoch):
        sum_error = 0
        index= np.random.randint(len(train_y),size=50)
        acc_sample_x = train_x[index]
        acc_sample_y =train_y[index]
        left_sample_y= train_y[-index]
        left_sample_x =train_x[-index]
        for i in range(300):
            row_index = np.random.randint(len(left_sample_y),size=1)
            row = left_sample_y[row_index]
            y= left_sample_y[row_index]
            predicted = sigmoid_func(row, params)
            error =  max(0, 1-predicted*y)
            sum_error += error ** 2
            params[0] = params[0] + reg_param * error * predicted * (1.0 - predicted)
            for i in range(len(row)):
                params[i + 1] = params[i + 1] + reg_param * error * predicted * (1.0 - predicted) * row[i]
        print('>epoch=%d, lrate=%.5f, error=%.5f' % (epoch, reg_param, sum_error))
    return params




def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def logistic_regression(train_x, train_y, test_x, test_y, l_rate, n_epoch):
    predictions = list()
    coef = find_params(np.array(train_x), np.array(train_y), l_rate, n_epoch)
    for i, row in enumerate(test_x):
        yhat = sigmoid_func(row, coef)
        if yhat* test_y[i]>=1:
            predictions.append(+1)
        else:
            predictions.append(-1)
    return (predictions)




inc_data_train = pd.read_csv('income_data.csv', names=names, na_values=" ?")
inc_data_test = pd.read_csv('income_data_test.csv', names=names, na_values=" ?")
inc_data = pd.concat([inc_data_train, inc_data_test])

data_clean_X, data_clean_Y = do_cleaning(inc_data)
X_scaled = preprocessing.scale(data_clean_X)
encoded_Y = encode_labels(data_clean_Y)


X_train, Y_train, X_test, Y_test, X_validate, Y_validate = create_train_test_validate(X_scaled, encoded_Y)

pred_list = []
for i in [0.001,.01,1]:

    predictions = logistic_regression(np.array(X_train), list(Y_train), np.array(X_validate), list(Y_validate), i, 5)
    acc = accuracy_metric(list(Y_validate), predictions)
    pred_list.append(acc)

print(pred_list)
