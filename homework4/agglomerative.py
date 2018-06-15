import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from os import path
from scipy.cluster.vq import vq
import os

np.random.seed(3217)

data = pd.read_csv("employement.txt", header=0, sep="\t", index_col="Country")  # read in the data
print("Data is of the shape " + str(np.shape(data)))  # The shape of data
here = path.abspath(path.dirname(__file__))

DATA_PATH = os.path.join(here, "HMP_Dataset")

###############################################################################################
# Problem 1 part 1
###############################################################################################

model = linkage(data, 'single')  # Create a linkage matrix with single link
dendrogram(model, labels=data.index)  # plot the dendogram with countries as labels
plt.title('European employment Dendogram with Single link')  # Set the title of plot
plt.xticks(rotation='vertical')  # Select the orientation of plot
plt.gcf().subplots_adjust(bottom=0.30)  # set some boundary space
plt.xlabel("Countries")  # label x axis
plt.ylabel("Distance")  # label y axis
plt.savefig('single_link.png')  # save the plot
plt.close()  # close the plot

model = linkage(data, 'complete')  # Create a linkage matrix with complete link
dendrogram(model, labels=data.index)  # plot the dendogram with countries as labels
plt.title('European employment Dendogram with Complete link')  # Set the title of plot
plt.xticks(rotation='vertical')  # Select the orientation of plot
plt.gcf().subplots_adjust(bottom=0.30)  # set some boundary space
plt.xlabel("Countries")  # label x axis
plt.ylabel("Distance")  # label y axis
plt.savefig('complete_link.png')  # save the plot
plt.close()  # close the plot

model = linkage(data, 'average')  # Create a linkage matrix with average link
dendrogram(model, labels=data.index)  # plot the dendogram with countries as labels
plt.title('European employment Dendogram with Group average')  # Set the title of plot
plt.xticks(rotation='vertical')  # Select the orientation of plot
plt.gcf().subplots_adjust(bottom=0.30)  # set some boundary space
plt.xlabel("Countries")  # label x axis
plt.ylabel("Distance")  # label y axis
plt.savefig('group_average.png')  # save the plot
plt.close()  # close the plot

###############################################################################################
# Problem 1 part 2
###############################################################################################


distorsions = []
K = range(2, 14)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=6).fit(data)  # fit the data on different values of k
    distorsions.append(k_means.inertia_)  # store the inertia in a list for each k

plt.plot(K, distorsions, marker="o")  # plot the inertia against k
plt.xlabel('k')  # label x axis
plt.ylabel('Distortion')  # label y axis
plt.title('Elbow plot showing the value of k')  # set the plot title
plt.grid(True)  # set grid on
plt.savefig('elbow_plot.png')  # save the plot
plt.close()  # close the plot
#

print("Problem 1 completed")

###############################################################################################
# Problem 2
###############################################################################################
n_cluster = 480  # set the no of cluster
time_unit = 32  # set the segment length
print("Problem 2 started with no of cluster = " + str(480) + " and segment length = " + str(32))
index_list = [os.path.join(DATA_PATH, 'Brush_teeth'), os.path.join(DATA_PATH, 'Climb_stairs'),
              os.path.join(DATA_PATH, 'Comb_hair')
    , os.path.join(DATA_PATH, 'Descend_stairs'), os.path.join(DATA_PATH, 'Drink_glass'),
              os.path.join(DATA_PATH, 'Eat_meat'),
              os.path.join(DATA_PATH, 'Eat_soup'), os.path.join(DATA_PATH, 'Getup_bed'),
              os.path.join(DATA_PATH, 'Liedown_bed'), os.path.join(DATA_PATH, 'Pour_water'),
              os.path.join(DATA_PATH, 'Sitdown_chair'), os.path.join(DATA_PATH, 'Standup_chair'),
              os.path.join(DATA_PATH, 'Use_telephone'),
              os.path.join(DATA_PATH, 'Walk')]  # create an index of adl activity


def generate_segments(data, n_cluster, time_unit):
    no_of_rows = np.shape(data)[0]  # get no of rows from the passed data
    mod = no_of_rows % time_unit  # get the extra segments from the data
    if mod != 0:
        data_to_segment = np.array(data)[:-mod, :]  # remove the extra segments
    else:
        data_to_segment = np.array(data)
    vector_segment = data_to_segment.reshape(int(no_of_rows / time_unit),
                                             time_unit * 3)  # reshape to have a segment represented by a single vector
    return pd.DataFrame(vector_segment)


def read_attribute_from_all_file(dir, n_cluster, time_unit):
    files = os.listdir(dir)  # get all the files in the given dir
    train_per = int(0.8 * len(files))  # get 80% of the files
    test_per = int(0.2 * len(files))  # get 20% of the files for the testing data
    full_data_train = pd.DataFrame()  # initialize a empty train data frame
    full_data_test = pd.DataFrame()  # initialize a empty test data frame
    for file in files[:train_per]:  # Go through all the files in the train set
        file_path = os.path.join(dir, file)  # generate the file actual path
        data = pd.read_csv(file_path, sep=" ", index_col=None, names=['x', 'y', 'z'],
                           skip_blank_lines=True).dropna()  # import the data from that file
        segmented_data_train = generate_segments(data, n_cluster,
                                                 time_unit)  # Generate segment of vectors of specified length
        full_data_train = full_data_train.append(segmented_data_train,
                                                 ignore_index=True)  # create a data frame of all such segments form all the files

    for file in files[-test_per:]:  # get all the test files in the given dir
        file_path = os.path.join(dir, file)  # generate the file actual path
        data = pd.read_csv(file_path, sep=" ", index_col=None, names=['x', 'y', 'z'], skip_blank_lines=True).dropna()
        segmented_data_test = generate_segments(data, n_cluster,
                                                time_unit)  # Generate segment of vectors of specified length
        full_data_test = full_data_test.append(segmented_data_test,
                                               ignore_index=True)  # create a data frame of all such segments f
    return full_data_train, full_data_test


def create_feature_for_classifier(model, dir, n_cluster, time_unit):
    files = os.listdir(dir)  # list all the files in dir
    train_per = int(0.8 * len(files))  # get 80% of the files
    test_per = int(0.2 * len(files))  # get 20% of the files for the testing data
    feature_train = pd.DataFrame()  # initialize a empty train data frame
    feature_test = pd.DataFrame()  # initialize a empty test data frame
    for file in files[:train_per]:  # Go through all the files in the train set
        file_path = os.path.join(dir, file)  # generate the file actual path
        data = pd.read_csv(file_path, sep=" ", index_col=None, names=['x', 'y', 'z'], skip_blank_lines=True).dropna()
        segmented_data_train = generate_segments(data, n_cluster,
                                                 time_unit)  # Generate segment of vectors of specified length

        assignment = vq(segmented_data_train,
                        model.cluster_centers_)  # Generate the assignment vector which shall assign each segment to a given cluster
        assignment_array = np.array(assignment[0])  # take the assignment array
        feature = [0 for s in
                   range(0, n_cluster + 1)]  # initialize a feature vector of a set length for random forest classifier
        for i in assignment_array:  # iterate through assignment
            feature[i] += 1  # fill up the bucket such that you have a histogram of points
        feature[n_cluster] = index_list.index(dir) + 1  # set the label for the classifier
        feature_df = pd.DataFrame(np.array(feature).reshape(1, n_cluster + 1))  # get everything in one row
        feature_df.columns = range(1, n_cluster + 2)  # set the column labels
        feature_train = feature_train.append(feature_df)  # create a full set of all these training

    for file in files[-test_per:]:  # get all the test files in the given dir
        file_path = os.path.join(dir, file)  # generate the file actual path
        data = pd.read_csv(file_path, sep=" ", index_col=None, names=['x', 'y', 'z'], skip_blank_lines=True).dropna()
        segmented_data_test = generate_segments(data, n_cluster,
                                                time_unit)  # Generate segment of vectors of specified length
        assignment = vq(segmented_data_test,
                        model.cluster_centers_)  # Generate the assignment vector which shall assign each segment to a given cluster
        assignment_array = np.array(assignment[0])  # take the assignment array
        feature = [0 for s in
                   range(0, n_cluster + 1)]  # initialize a feature vector of a set length for random forest classifier
        for i in assignment_array:  # iterate through assignment
            feature[i] += 1  # fill up the bucket such that you have a histogram of points
        feature[n_cluster] = index_list.index(dir) + 1  # set the label for the classifier
        feature_df = pd.DataFrame(np.array(feature).reshape(1, n_cluster + 1))  # get everything in one row
        feature_df.columns = range(1, n_cluster + 2)  # set the column labels
        feature_test = feature_test.append(feature_df)  # create a full set of all these training
    return feature_train, feature_test


def generate_vectors(n_cluster, time_unit):
    # Generate the vector segments for each adl and split them into train and test
    brush_vector_train, brush_vector_test = read_attribute_from_all_file(os.path.join(DATA_PATH, "Brush_teeth"),
                                                                         n_cluster, time_unit)
    climb_stairs_vector_train, climb_stairs_vector_test = read_attribute_from_all_file(
        os.path.join(DATA_PATH, "Climb_stairs"), n_cluster, time_unit)
    comb_hair_vector_train, comb_hair_vector_test = read_attribute_from_all_file(os.path.join(DATA_PATH, "Comb_hair"),
                                                                                 n_cluster, time_unit)
    descend_stairs_vector_train, descend_stairs_vector_test = read_attribute_from_all_file(
        os.path.join(DATA_PATH, "Descend_stairs"), n_cluster, time_unit)
    drink_glass_vector_train, drink_glass_vector_test = read_attribute_from_all_file(
        os.path.join(DATA_PATH, "Drink_glass"), n_cluster, time_unit)
    eat_meat_vector_train, eat_meat_vector_test = read_attribute_from_all_file(os.path.join(DATA_PATH, "Eat_meat"),
                                                                               n_cluster, time_unit)
    eat_soup_vector_train, eat_soup_vector_test = read_attribute_from_all_file(os.path.join(DATA_PATH, "Eat_soup"),
                                                                               n_cluster, time_unit)
    get_up_bed_vector_train, get_up_bed_vector_test = read_attribute_from_all_file(os.path.join(DATA_PATH, "Getup_bed"),
                                                                                   n_cluster, time_unit)
    liedown_bed_vector_train, liedown_bed_vector_test = read_attribute_from_all_file(
        os.path.join(DATA_PATH, "Liedown_bed"), n_cluster, time_unit)
    pour_water_vector_train, pour_water_vector_test = read_attribute_from_all_file(
        os.path.join(DATA_PATH, "Pour_water"), n_cluster, time_unit)
    sitdown_chair_vector_train, sitdown_chair_vector_test = read_attribute_from_all_file(
        os.path.join(DATA_PATH, "Sitdown_chair"), n_cluster, time_unit)
    stand_up_chair_vector_train, stand_up_chair_vector_test = read_attribute_from_all_file(
        os.path.join(DATA_PATH, "Standup_chair"), n_cluster, time_unit)
    use_telephone_vector_train, use_telephone_vector_test = read_attribute_from_all_file(
        os.path.join(DATA_PATH, "Use_telephone"), n_cluster, time_unit)
    walk_vector_train, walk_vector_test = read_attribute_from_all_file(os.path.join(DATA_PATH, "Walk"), n_cluster,
                                                                       time_unit)
    print("Vector segmentation complete")

    # # merge all the vector segments for each adl activity into one
    feature_vector = np.concatenate(
        [brush_vector_train, climb_stairs_vector_train, comb_hair_vector_train, descend_stairs_vector_train
            , drink_glass_vector_train, eat_meat_vector_train, eat_soup_vector_train, get_up_bed_vector_test,
         liedown_bed_vector_train,
         pour_water_vector_train, sitdown_chair_vector_train, stand_up_chair_vector_train,
         use_telephone_vector_train, walk_vector_train])
    return feature_vector


def generate_classifier_feature(feature_vector, n_cluster, time_unit):
    k_means = KMeans(n_clusters=n_cluster, random_state=0).fit(feature_vector)  # initialize a k-mean model
    print("K-means clustering complete")
    # Generate training and test features for all the adl activity
    train_classifier = pd.DataFrame()
    test_classifier = pd.DataFrame()
    for dir in index_list:
        train, test = create_feature_for_classifier(k_means, dir, n_cluster, time_unit) # create test and train for the classifier
        train_classifier = train_classifier.append(train)
        test_classifier = test_classifier.append(test)
    print("Finished creating features for classifier")
    return train_classifier, test_classifier


n_clust = 480
segment_l = 32
feature_vector = generate_vectors(n_clust, segment_l)
train_classifier, test_classifier = generate_classifier_feature(feature_vector, n_clust, segment_l)
# set up a random forest classifier
random_forest_model = RandomForestClassifier(max_depth=32, random_state=8, n_estimators=90)
# fit the model on training set
random_forest_model.fit(train_classifier.iloc[:, :n_clust], train_classifier.iloc[:, n_clust])
# test the model on test set
prediction = random_forest_model.predict(test_classifier.iloc[:, :n_clust])
# Calculate the accuracy
acc = accuracy_score(test_classifier.iloc[:, n_clust], prediction)
print("Accuracy achieved  is " + str(acc * 100) + "%")
print("Error rate for the classifier is "+ str((1-acc)*100)+"%")
print(confusion_matrix(test_classifier.iloc[:, n_clust], prediction))

####################################################################################################################
# Problem 2 part 2
####################################################################################################################

n_cluster_list = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 500]
time_unit_list = [4,8, 12,16, 20, 24, 28, 32, 36, 40]
accuracy_cluster_list = []
accuracy_segment_list = []
dist_list=[]

# get the optimum no of clusters
for k in n_cluster_list:
    k_means = KMeans(n_clusters=k, random_state=6).fit(feature_vector)  # fit the data on different values of k
    dist_list.append(k_means.inertia_)  # store the inertia in a list for each k

plt.plot(n_cluster_list, dist_list, marker="o")  # plot the inertia against k
plt.xlabel('k')  # label x axis
plt.ylabel('Distortion')  # label y axis
plt.title('Elbow plot showing the value of k')  # set the plot title
plt.grid(True)  # set grid on
plt.savefig('elbow_plot_part2.png')  # save the plot
plt.close()  # close the plot


# calculate the accuracy for different value of clusters
for n_cluster in n_cluster_list:
    print("Calculating accuracy for cluster = " + str(n_cluster))
    feature_vector = generate_vectors(n_cluster, segment_l)
    train_classifier, test_classifier = generate_classifier_feature(feature_vector, n_cluster, segment_l)
    random_forest_model = RandomForestClassifier(max_depth=32, random_state=8, n_estimators=90)
    random_forest_model.fit(train_classifier.iloc[:, :n_cluster], train_classifier.iloc[:, n_cluster])
    prediction = random_forest_model.predict(test_classifier.iloc[:, :n_cluster])
    acc = accuracy_score(test_classifier.iloc[:, n_cluster], prediction)
    accuracy_cluster_list.append(acc)
# calculate the accuracy for different value of segment length
for seg_length in time_unit_list:
    print("Calculating accuracy for segment length = " + str(seg_length))
    feature_vector = generate_vectors(n_clust, seg_length)
    train_classifier, test_classifier = generate_classifier_feature(feature_vector, n_clust, seg_length)
    random_forest_model = RandomForestClassifier(max_depth=32, random_state=8, n_estimators=90)
    random_forest_model.fit(train_classifier.iloc[:, :n_clust], train_classifier.iloc[:, n_clust])
    prediction = random_forest_model.predict(test_classifier.iloc[:, :n_clust])
    acc = accuracy_score(test_classifier.iloc[:, n_clust], prediction)
    accuracy_segment_list.append(acc)

# plot the accuracy as function of no of clusters
plt.plot(n_cluster_list, accuracy_cluster_list)
plt.xlabel("No of clusters")
plt.ylabel("Accuracy of Random Forest Classifier")
plt.title("Effect of changing no of clusters on the accuracy of classifier ")
plt.grid(True)
plt.savefig('cluster_accuracy.png')
plt.close()

# plot the accuracy as function of segment length
plt.plot(time_unit_list, accuracy_segment_list)
plt.xlabel("Vector segment length")
plt.ylabel("Accuracy of Random Forest Classifier")
plt.title("Effect of changing segment length on the accuracy of classifier ")
plt.grid(True)
plt.savefig('segment_length_accuracy.png')
plt.close()



# generate a plot that will have affect of both segment and no of cluster
n_cluster_list = [25, 50, 200, 400]
time_unit_list = [4, 8, 32]

result_list = []
for seg_length in time_unit_list:
    interim = []
    for n_cluster in n_cluster_list:
        feature_vector = generate_vectors(n_cluster, seg_length)
        train_classifier, test_classifier = generate_classifier_feature(feature_vector, n_cluster, seg_length)
        random_forest_model = RandomForestClassifier(max_depth=32, random_state=8, n_estimators=90)
        random_forest_model.fit(train_classifier.iloc[:, :n_cluster], train_classifier.iloc[:, n_cluster])
        prediction = random_forest_model.predict(test_classifier.iloc[:, :n_cluster])
        acc = accuracy_score(test_classifier.iloc[:, n_cluster], prediction)
        interim.append(acc*100)
    result_list.append(interim)


for i, y in enumerate(result_list):
    plt.plot(n_cluster_list, y, label=time_unit_list[i])
plt.xlabel("No of clusters")
plt.ylabel("Accuracy")
plt.title("Variation in Accuracy with segment size and no of cluster")
plt.ylim([0, 100])
plt.yticks(np.arange(0, 100, 5))
plt.legend()
plt.grid(True)
plt.savefig('problme2_part2.png')
plt.close()


## Best guess

feature_vector = generate_vectors(50, 4)
train_classifier, test_classifier = generate_classifier_feature(feature_vector, 50, 4)
# set up a random forest classifier
random_forest_model = RandomForestClassifier(max_depth=32, random_state=8, n_estimators=90)
# fit the model on training set
random_forest_model.fit(train_classifier.iloc[:, :50], train_classifier.iloc[:, 50])
# test the model on test set
prediction = random_forest_model.predict(test_classifier.iloc[:, :50])
# Calculate the accuracy
acc = accuracy_score(test_classifier.iloc[:, 50], prediction)
print("Accuracy achieved  is " + str(acc * 100) + "%")
print("Error rate for the classifier is "+ str((1-acc)*100)+"%")
print(confusion_matrix(test_classifier.iloc[:, 50], prediction))