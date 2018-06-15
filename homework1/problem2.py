import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from PIL import Image

np.random.seed(3217)

dataset = datasets.fetch_mldata("MNIST Original")

features = pd.DataFrame(np.array(dataset.data, 'int16'))
labels = pd.DataFrame(np.array(dataset.target, 'int'))

data = pd.concat([labels, features], axis=1)
train, test = train_test_split(data, test_size=0.2)

train_x = np.array(train.iloc[:, 1:])
train_y = np.array(train.iloc[:, 0])
test_y = np.array(test.iloc[:, 0])
test_x = np.array(test.iloc[:, 1:])

target_train_y = train_y.astype(np.uint8)
target_test_y = test_y.astype(np.uint8)


def rescale_strech_image(image):
    x = np.array(image).reshape((-1, 1, 28, 28)).astype(np.uint8)
    img1 = Image.fromarray(x[0][0])
    bw_img = img1.point(lambda x: 0 if x < 128 else 255, '1')
    img = np.reshape(np.array(bw_img), (28, 28))
    row= np.unique(np.nonzero(img)[0])
    col = np.unique(np.nonzero(img)[1])
    image_data_new = img[min(row):max(row), min(col):max(col)]
    image_data_new = imresize(image_data_new, (20, 20))
    return (np.array(image_data_new).astype(np.uint8))


train_modified = np.apply_along_axis(rescale_strech_image, axis=1, arr=train_x)
test_modified = np.apply_along_axis(rescale_strech_image, axis=1, arr=test_x)

train_final = np.reshape(train_modified, (train_modified.shape[0], 400))
test_final = np.reshape(test_modified, (test_modified.shape[0], 400))


def do_naive_bayes(train_x, train_y, test_x, test_y, model, case):
    bernoli_model = model
    bernoli_model.fit(np.array(train_x), train_y)
    predic = bernoli_model.predict(np.array(test_x))

    acc = accuracy_score(test_y, predic)
    print("Accuracy achieved for " + case + "is " + str(acc * 100) + "%")


do_naive_bayes(train_final, target_train_y, test_final, target_test_y, GaussianNB(),
                                 "bounded_stretched_gausian")
do_naive_bayes(train_final, target_train_y, test_final, target_test_y, BernoulliNB(),
                                 "bounded_stretched_bernoli")

do_naive_bayes(train_x, train_y, test_x, test_y, GaussianNB(), "untouched_gausian")
do_naive_bayes(train_x, train_y, test_x, test_y, BernoulliNB(), "untouched_bernoli")


def do_random_forest(d, n, train_x1, train_y1, test_x1, test_y1, case):
    random_forest_model = RandomForestClassifier(max_depth=d, random_state=0, n_estimators=n)
    random_forest_model.fit(train_x1, train_y1)
    preiction = random_forest_model.predict(test_x1)
    acc = accuracy_score(test_y1, preiction)
    print("Accuracy achieved for " + case + " is " + str(acc * 100) + "%")


do_random_forest(4, 10, train_x, train_y, test_x, test_y, "4/10 untouched")
do_random_forest(4, 20, train_x, train_y, test_x, test_y, "4/20 untouched")
do_random_forest(4, 30, train_x, train_y, test_x, test_y, "4/30 untouched")

do_random_forest(8, 10, train_x, train_y, test_x, test_y, "8/10 untouched")
do_random_forest(8, 20, train_x, train_y, test_x, test_y, "8/20 untouched")
do_random_forest(8, 30, train_x, train_y, test_x, test_y, "8/30 untouched")

do_random_forest(16, 10, train_x, train_y, test_x, test_y, "16/10 untouched")
do_random_forest(16, 20, train_x, train_y, test_x, test_y, "16/20 untouched")
do_random_forest(16, 30, train_x, train_y, test_x, test_y, "16/30 untouched")

do_random_forest(4, 10, train_final, target_train_y, test_final, target_test_y, "4/10 bounded and stretched")
do_random_forest(4, 20, train_final, target_train_y, test_final, target_test_y, "4/20 bounded and stretched")
do_random_forest(4, 30, train_final, target_train_y, test_final, target_test_y, "4/30 bounded and stretched")

do_random_forest(8, 10, train_final, target_train_y, test_final, target_test_y, "8/10 bounded and stretched")
do_random_forest(8, 20, train_final, target_train_y, test_final, target_test_y, "8/20 bounded and stretched")
do_random_forest(8, 30, train_final, target_train_y, test_final, target_test_y, "8/30 bounded and stretched")

do_random_forest(16, 10, train_final, target_train_y, test_final, target_test_y, "16/10 bounded and stretched")
do_random_forest(16, 20, train_final, target_train_y, test_final, target_test_y, "16/20 bounded and stretched")
do_random_forest(16, 30, train_final, target_train_y, test_final, target_test_y, "16/30 bounded and stretched")
