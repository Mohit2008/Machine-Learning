import pandas as pd
import numpy as np
from sklearn import decomposition
from scipy.misc import imresize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from PIL import Image

np.random.seed(5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# Helper Function to read the data by using pickle
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data_features = np.array(dict[(b'data')]).reshape(10000, 3072)
    data_labels = np.array(dict[(b'labels')])
    return data_features, data_labels


# Get category labels
def get_labels(file):
    with open(file) as f:
        lines = f.read().splitlines()
    return lines


labels = get_labels("cifar-10-batches-py/batches.meta.txt")


# Get the error between original and approximated image using sum of squared distance
def calculate_difference(original_image, aprrox_image):
    return np.mean((np.power((np.subtract(original_image, aprrox_image)), 2).sum(axis=1)))


# Load the data in batches, also combine the train and test data
def load_train_data():
    batch1_X, batch1_Y = unpickle("cifar-10-batches-py/data_batch_1")
    batch2_X, batch2_Y = unpickle("cifar-10-batches-py/data_batch_2")
    batch3_X, batch3_Y = unpickle("cifar-10-batches-py/data_batch_3")
    batch4_X, batch4_Y = unpickle("cifar-10-batches-py/data_batch_4")
    batch5_X, batch5_Y = unpickle("cifar-10-batches-py/data_batch_5")
    test_x, test_y = unpickle("cifar-10-batches-py/test_batch")
    train_x = np.concatenate((batch1_X, batch2_X, batch3_X, batch4_X, batch5_X, test_x),
                             axis=0)  # Merge all the feature set
    train_y = np.concatenate((batch1_Y, batch2_Y, batch3_Y, batch4_Y, batch5_Y, test_y),
                             axis=0)  # Merge all the label set
    return train_x, train_y


# Helper method to view images
def show_image(image):
    x = np.array(image).reshape((-1, 1, 32, 32, 3)).astype(np.uint8) # reshape it to have rgb channels
    img1 = Image.fromarray(x[0][0]) #get the image matrix
    img1.show() # showthe image


train_x, train_y = load_train_data()  # Get all the data in one chunk
feature = pd.DataFrame(train_x,
                       index=train_y)  # Create a data frame of all 60000 feature instance and index them with their labels
pca = decomposition.PCA(n_components=20)  # Initialise the PCA for 20 components

# Get a list of error for each label set
error_list = []
# Get a list of mean for each label set
mean_list = []
# Get a list of feature for each label set
uncentered_list = []
original_image_list = []
eigen_vector_list = []
for i in range(0, 10):
    feat = feature[feature.index == i]  # Identify the labels
    uncentered_list.append(feat)
    feature_mean = np.mean(feat)  # Get the mean image
    mean_list.append(feature_mean)
    n_feature = feat - feature_mean  # Center the data
    original_image_list.append(n_feature)
    pca.fit(n_feature)  # Fit the centered data
    feature_pca = pca.transform(n_feature)  # Transform the data
    eigen_vector_list.append(np.array(pca.components_).transpose())  # Get the eigen vectors
    error = calculate_difference(n_feature, pca.inverse_transform(
        feature_pca))  # Get the error as the sum of unexplained variance in the model
    error_list.append(error)

########################################################################################################
# Problem 1
########################################################################################################

plt.bar(labels, error_list)  # Plot the error against each category

plt.ylabel("Error representing the image using first 20 PC")
plt.axis('tight')
plt.xticks(range(0, len(error_list)), labels, rotation='vertical')  # Setting the x labels
plt.margins(0.2)  # SE the plot margins
plt.title("Error Plot")
plt.savefig('error_part1.png', bbox_inches='tight')  # Save the plot to a file
plt.close()  # Close the plot instance to clear the canvas
print("Problem 1 ended")
########################################################################################################
# Problem 2
########################################################################################################

mean_diff_array = []
# Go through the mean list and calculate mean diff for each category against every other category
for i in range(0, len(mean_list)):
    for j in range(0, len(mean_list)):
        distance = np.linalg.norm(mean_list[i] - mean_list[j]) # get the distance between 2 images
        mean_diff_array.append(distance)  # Array of 100 elements

# Reshape to 10 rows and 10 columns such that each row represent error of category 1 with every other category
euclid_distance = np.array(mean_diff_array).reshape(10, 10)
np.savetxt("distance_part2.txt", euclid_distance, fmt='%1.2f', delimiter='\t') # save the distance matrix
# Get the multidimensional scaling done with 2 components
mds = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=1,
          dissimilarity="precomputed", n_jobs=1)
embed2d = mds.fit(euclid_distance).embedding_  # Fitting the model to generate 2D data

plt.scatter(embed2d[:, 0], embed2d[:, 1], s=20)  # Plotting the 2 dimensions
plt.title('2D Multidim. Scaling for the distance between 2 classes')
for i, txt in enumerate(labels):
    plt.annotate(txt, (embed2d[i, 0], embed2d[i, 1]))  # Annotate the data in the plot for better understanding
plt.axis('tight')
plt.savefig('mds_diff_part2.png')  # Save the plot to a file
plt.close()  # Close the plot instance to clear the canvas
print("Problem 2 ended")
########################################################################################################
# Problem 3
########################################################################################################


error_diff_array = []
# Get the error for each category against every other category while represeting a image in a category with PC of other category
for i in range(0, 10):
    for j in range(0, 10):
        per_pixel_rotation = np.dot(np.array(eigen_vector_list[j]).transpose(), np.array(
            original_image_list[i]).transpose())  # Multiply the original image with eigen vector of other category
        rotated_matrix = np.dot(np.array(eigen_vector_list[j]),
                                per_pixel_rotation).transpose()  # Get the rotated matrix
        reconstructed_image = np.array(rotated_matrix) + np.array(
            mean_list[i])  # Image of category i constructed from PC of category j
        diff = calculate_difference(uncentered_list[i], reconstructed_image)  # Get the error in doing so
        error_diff_array.append(diff)

error_matrix = (np.array(error_diff_array).reshape(10, 10))  # reshape it to have 10 rows and 10 columns

sim_array = []
# Now calculate a symmetric matrix that has similarity of E(1|2) with E(2|1) for every combination of category
for i in range(0, 10):
    for j in range(0, 10):
        sim_array.append((1 / 2) * (error_matrix[i, j] + error_matrix[j, i]))  # Similarity measure between category i,j

sim_matrix = np.array(sim_array).reshape(10, 10)  # reshape it to have 10 rows and 10 columns
np.savetxt("distance_part3.txt", sim_matrix, fmt='%1.2f', delimiter='\t') # save the similarity matrix
mds = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=1,
          dissimilarity="precomputed", n_jobs=1)
embed2d = mds.fit(sim_matrix).embedding_  # Fitting the model to generate 2D data

plt.scatter(embed2d[:, 0], embed2d[:, 1], s=20)  # Plotting the 2 dimensions
plt.title('2D Multidim. Scaling for the similarity between 2 classes')
for i, txt in enumerate(labels):
    plt.annotate(txt, (embed2d[i, 0], embed2d[i, 1]))  # Annotate the data in the plot for better understanding
plt.axis('tight')
plt.savefig('mds_sim_part3.png')  # Save the plot to a file
plt.close()  # Close the plot instance to clear the canvas
print("Problem 3 ended")
