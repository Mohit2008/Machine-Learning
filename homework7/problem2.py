import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
from scipy.spatial import distance

threshold = 0.0001


# Perform E-Step
def expectation(pixel, num_segments, mu, pij):
    bottom = np.zeros((pixel.shape[0]))
    weight = np.zeros((pixel.shape[0], num_segments))  # initialize the weights
    distances = distance.cdist(pixel, mu, 'euclidean')  # calculate distances
    dmin = np.square(np.amin(distances, axis=1))  # get dmin squared
    for k in range(num_segments):
        temp = pij[k] * np.exp(-1 * ((((pixel - mu[k]) ** 2).sum(1)) - dmin) / 2.0)
        bottom += temp
    for k in range(num_segments):
        temp = pij[k] * np.exp(-1 * ((((pixel - mu[k]) ** 2).sum(1)) - dmin) / 2.0)
        weight[:, k] = temp / bottom
    return weight


# Perform M-Step
def maximisation(pixel, w, n_segments):
    mu = np.zeros((n_segments, 3))
    pjk = np.zeros((n_segments))
    weight = np.zeros((w.shape[0], w.shape[1], 3))
    weight[:, :, 0] = w
    weight[:, :, 1] = w
    weight[:, :, 2] = w
    for j in range(n_segments):
        mu[j] = (np.sum(pixel * weight[:, j, :], axis=0) / np.sum(w[:, j]))
        pjk[j] = (np.sum(w[:, j]) / (pixel.shape[0]))
    return mu, pjk


# calculate the value of q
def compute_q(pixel, n_segments, mu, pij, w):
    sigma = 0
    for j in range(n_segments):
        exped = pij[j] * np.exp(-1 * ((((pixel - mu[j]) ** 2).sum(1))) / 2.0)
        dot = -1 * (exped) / 2.0
        sigma += (dot + pij[j]) * w[:, j]
    return np.sum(sigma)


########################################################################################################################
# Problem 2 part 1
########################################################################################################################

def do_soft_clustering(image_list,segment_list, run=0):
    for image in image_list:
        for num_segments in segment_list:
            image_path = "input/" + image + ".jpg"
            iter = 0
            diff = 10000
            prev_q = 0
            new_q = 0
            result = {}
            result['clusters'] = {}
            result['params'] = {}
            print(image, num_segments)
            input_pixel = misc.face()
            input_pixel = misc.imread(image_path)
            all_pixel = []
            image_arr = np.empty([len(input_pixel) * len(input_pixel[0]), 3])
            for row in input_pixel:
                for pixel in row:
                    all_pixel.append(pixel)
            for index, item in enumerate(all_pixel):
                image_arr[index] = item

            image_arr_copy = image_arr.copy()
            model = KMeans(n_clusters=num_segments, random_state=50).fit(image_arr_copy)  # do kmeans
            mu = model.cluster_centers_
            pij = np.full((num_segments), 1.0 / num_segments)
            while iter < 100 and diff > threshold:  # stopping criteria
                weig = expectation(image_arr_copy, num_segments, mu, pij)
                mu, pij = maximisation(image_arr_copy, weig, num_segments)
                prev_q = new_q
                new_q = compute_q(image_arr_copy, num_segments, mu, pij, weig)  # get the new value of q
                print(new_q)
                diff = abs(new_q - prev_q) / abs(new_q)  # calculate diff in q
                iter += 1

            # do segmentation
            for i in range(num_segments):
                result['params'][i] = {}
                result['params'][i]['pi'] = pij[i]
                result['params'][i]['mu'] = mu[i]
            for index, pixel in enumerate(image_arr_copy):
                p_m = np.zeros((num_segments, 3))
                for i in range(num_segments):
                    p_m[i] = pixel
                cluster = np.argmin(distance.cdist(p_m, mu, 'euclidean')) # get the neighbour
                if cluster not in result['clusters']:
                    result['clusters'][cluster] = []
                result['clusters'][cluster].append(index)

            # create output by replacing every elemnent with the group
            write_array = np.empty([len(input_pixel), len(input_pixel[0]), 3])
            for i in range(num_segments):
                for item in result['clusters'][i]:
                    row = item // len(input_pixel[0]);
                    col = item - row * len(input_pixel[0]);
                    write_array[row][col] = result['params'][i]['mu']
            # save the results
            if len(segment_list)==1:
                output_image = "output_images/" + image + "_seed_" + str(run) + '.jpg'
            else:
                output_image = "output_images/" + image + "_" + str(num_segments) + '.jpg'
            misc.imsave(output_image, write_array)

do_soft_clustering(['smallstrelitzia', 'RobertMixed03', 'smallsunset'], [10, 20, 50])

########################################################################################################################
# Problem 2 part 2
########################################################################################################################


for run in range(5):
    do_soft_clustering(['smallsunset'], [20], run)