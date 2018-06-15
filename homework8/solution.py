import mnist
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt, warnings
import math


warnings.filterwarnings("ignore")
sns.set()
np.random.seed(123)


# Read the files containing the order of update ina dataframe with each cell having tuple (row, column)
def read_update_order_cordinates():
    image = [i for i in range(0, 20)]
    cols = [i for i in range(0, 784)]
    update_order_df = pd.DataFrame(index=image, columns=cols)
    update_order= pd.read_csv("SupplementaryAndSampleData/UpdateOrderCoordinates.csv", header=0)
    update_order = update_order.drop(['Row Description'], axis=1)
    for i in range(0, 40, 2):
        for j in range(784):
            index = int(i / 2)
            row = update_order.iloc[i, j]
            col = update_order.iloc[i + 1, j]
            update_order_df.iloc[index,j]=(row, col)
    return  update_order_df

#read initializing parameters Q
def read_initial_parameters():
    initial_params = pd.read_csv("SupplementaryAndSampleData/InitialParametersModel.csv", header=None)
    return initial_params.copy()

# function to set the image in 28*28
def create_image(img_list):
    output = []
    for i in range(0,img_list.shape[0]):
        output.append(np.reshape(img_list[i],(28,28)))
    return output

# plot all the 10 original, noisy, denoised image
def show_image(binary_pixel,noise_pixel,denoised_images, index):
    binary_pixel = create_image(binary_pixel)
    noise_pixel = create_image(noise_pixel)
    denoised_images = create_image(denoised_images)
    plt.subplot(1,3,1)
    plt.title('Original image')
    plt.imshow(binary_pixel[index], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noise_pixel[index], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('De Noised Image')
    plt.imshow(denoised_images[index], cmap='gray')
    image_name= "image_at_index_"+str(index)+".png"
    plt.savefig(image_name)  # save the plot
    plt.close()


train_images = mnist.train_images()
train_images=train_images[:20,:,:]
noise_data=pd.read_csv("SupplementaryAndSampleData/NoiseCoordinates.csv", header=0) # read in the noisy data from csv


noise_data=noise_data.drop(['Row Description'], axis=1) # drop the unwanted column

binary_pixel = np.copy(train_images) # initilaize an array of pixels
binary_pixel=binary_pixel.astype(int)


#Binarze the entire image by bringing everything to -1 and +1
for k in range(20):
    for i in range(binary_pixel.shape[1]):
        for j in range(binary_pixel.shape[2]):
            if binary_pixel[k][i][j] <= 127:
                binary_pixel[k][i][j] = int(-1)
            else:
                binary_pixel[k][i][j] = 1

noise_pixel = np.copy(binary_pixel) # initialize a copy



#Create a noisy image by flipping some of the bits given in the csv file
for i in range(0, 40, 2):
    for j in range(0,15):
        index= int(i/2)
        row = noise_data.iloc[i, j]
        col = noise_data.iloc[i + 1, j]
        noise_pixel[index][row][col] = (noise_pixel[index][row][col]) * -1 # flip pixels



update_order_df=(read_update_order_cordinates()).copy() # read in the data about updating the order of pixels

# function to use the pij and update the pixel using the confidence supplied in pij
def update_images(pi):
    reconstructed_image = np.zeros((28,28))
    for i in range(0,28):
        for j in range(0,28):
            if (pi[i][j] > 0.5):
                reconstructed_image[i][j]=1
            else:
                reconstructed_image[i][j]=-1
    return reconstructed_image


#Run the logic of updating the pij
def run_mean_field_inference(img, theta_ij_1, theta_ij_2, pixel_location, pij):
    # pij = np.copy(read_initial_parameters()) # initial value of pij
    for p in range(0,10): # do this for 10 times
        for loc in pixel_location: # get the location from the list
                x_cordinate=loc[0]
                y_cordinate=loc[1]
                temp = theta_ij_2 * img[x_cordinate][y_cordinate] # known pixel
                if (x_cordinate > 0): #leftmost row
                    temp = temp + theta_ij_1 * (2 * pij[x_cordinate - 1][y_cordinate] - 1) # hidden
                if (x_cordinate < 27): #rightmost row
                    temp = temp +  theta_ij_1 * (2 * pij[x_cordinate + 1][y_cordinate] - 1)# hidden
                if (y_cordinate > 0): # topmost column
                    temp = temp +  theta_ij_1 * (2 * pij[x_cordinate][y_cordinate - 1] - 1) #hidden
                if (y_cordinate < 27): # bottommost column
                    temp = temp +  theta_ij_1 * (2 * pij[x_cordinate][y_cordinate + 1] - 1) #hidden

                pij[x_cordinate][y_cordinate] = (np.exp(temp) / (np.exp(-temp) + np.exp(temp))) # calculate pij
    return pij


theta_ij_1 = 0.8
theta_ij_2 = 2.0

# do the mean field inference for all images
def do_boltzman(theta_ij_1,theta_ij_2,noise_pixel, update_order_df ):
    pij = np.zeros((20, 28, 28))
    print("Starting boltzman")
    for i in range(0, 20):
        print("Processing image "+ str(i))
        pij[i] = run_mean_field_inference(np.copy(noise_pixel[i]), theta_ij_1, theta_ij_2, update_order_df.iloc[i, :],np.copy(read_initial_parameters()))
    denoised_image = np.zeros((20, 28, 28)) # create empty array to hold denoised images
    print("Start updating")
    for i in range(0, denoised_image.shape[0]):
        denoised_image[i] = (update_images(pij[i])) # do the image update based on the value of pij
    return denoised_image


denoised_image = do_boltzman(theta_ij_1, theta_ij_2, noise_pixel, update_order_df.copy())

show_image(binary_pixel, noise_pixel, denoised_image, 0) # plot some images
show_image(binary_pixel, noise_pixel, denoised_image, 1) # plot some images
show_image(binary_pixel, noise_pixel, denoised_image, 2) # plot some images
show_image(binary_pixel, noise_pixel, denoised_image, 15) # plot some images
show_image(binary_pixel, noise_pixel, denoised_image, 19) # plot some images

denoised_image[denoised_image == -1.0] =int(0) # get the data ready for inputting in csv that acceptable by autograder
denoised_image[denoised_image == 1.0] =int(1)# get the data ready for inputting in csv that acceptable by autograder

stacked_denoised_images = denoised_image[10] # initialize the array with the first denoised image
for i in range(11,20):
    stacked_denoised_images = np.column_stack((stacked_denoised_images, denoised_image[i])) # start stacking the images column wise

print(np.shape(stacked_denoised_images))
stacked_images_df = pd.DataFrame(stacked_denoised_images).astype(int) # create a dataframe of binary stacked images
stacked_images_df.to_csv("denoised.csv", header=False, index=False) # output the results to csv



################################################################################################
#Part 4
################################################################################################

# function to calculate EQ
def calculate_EQ(theta_ij_1, theta_ij_2, noise_pixel, impi):
    epQ = np.sum(impi * np.log(impi + math.pow(10, -10)) + (1 - impi) * np.log(
        (1 - impi) + math.pow(10, -10)))  # calculate the first part
    eqP = 0.0
    for x_cordinate in range(0, 28):
        for y_cordinate in range(0, 28):
            eqP = eqP + theta_ij_2 * (2 * impi[x_cordinate, y_cordinate] - 1) * noise_pixel[x_cordinate, y_cordinate] #known pixel
            if (x_cordinate > 0):
                eqP = eqP + theta_ij_1 * (2 * impi[x_cordinate, y_cordinate] - 1) * (2 * impi[x_cordinate - 1, y_cordinate] - 1)
            if x_cordinate < 27:
                eqP = eqP + theta_ij_1 * (2 * impi[x_cordinate, y_cordinate] - 1) * (2 * impi[x_cordinate + 1, y_cordinate] - 1)
            if y_cordinate > 0:
                eqP = eqP + theta_ij_1 * (2 * impi[x_cordinate, y_cordinate] - 1) * (2 * impi[x_cordinate, y_cordinate - 1] - 1)
            if y_cordinate < 27:
                eqP = eqP + theta_ij_1 * (2 * impi[x_cordinate, y_cordinate] - 1) * (2 * impi[x_cordinate, y_cordinate + 1] - 1)
    energy = (epQ - eqP)  # calculate energy
    return energy

#Run the logic of updating the pij
def update_pi(img, theta_ij_1, theta_ij_2, pixel_location, pij):
    for loc in pixel_location:  # get the location from the list
        x_cordinate = loc[0]
        y_cordinate = loc[1]
        temp = theta_ij_2 * img[x_cordinate][y_cordinate]  # known pixel
        if (x_cordinate > 0):  # leftmost row
            temp = temp + theta_ij_1 * (2 * pij[x_cordinate - 1][y_cordinate] - 1)  # hidden
        if (x_cordinate < 27):  # rightmost row
            temp = temp + theta_ij_1 * (2 * pij[x_cordinate + 1][y_cordinate] - 1)  # hidden
        if (y_cordinate > 0):  # topmost column
            temp = temp + theta_ij_1 * (2 * pij[x_cordinate][y_cordinate - 1] - 1)  # hidden
        if (y_cordinate < 27):  # bottommost column
            temp = temp + theta_ij_1 * (2 * pij[x_cordinate][y_cordinate + 1] - 1)  # hidden

        pij[x_cordinate][y_cordinate] = (np.exp(temp) / (np.exp(-temp) + np.exp(temp)))  # calculate pij
    return pij

energy=np.zeros((20,11))

print("Starting Energy calculation")
for i in range(0, 20):
    impi=np.copy(read_initial_parameters())
    print("Processing image "+ str(i))
    energy[i,0] = calculate_EQ(theta_ij_1, theta_ij_2,np.copy(noise_pixel[i]), impi)
    for p in range(1,11):
        impi=update_pi(np.copy(noise_pixel[i]), theta_ij_1, theta_ij_2, update_order_df.iloc[i, :],impi)
        energy[i,p]=calculate_EQ(theta_ij_1, theta_ij_2,np.copy(noise_pixel[i]), impi)

energy_df = pd.DataFrame(energy[10:12,:2])
energy_df.to_csv("energy.csv", header=False, index=False,float_format='%.9f') # output the results to csv

################################################################################################
#Part 5
################################################################################################

def generate_cumulative_plot(binary_pixel,noise_pixel,denoised_images):
    plt.figure(figsize=(1500 / 96, 1500 / 96), dpi=96)
    binary_pixel = create_image(binary_pixel)
    noise_pixel = create_image(noise_pixel)
    denoised_images = create_image(denoised_images)
    for i in range(0,10):
        plt.subplot(3,10,i+1)
        plt.imshow(binary_pixel[i+10], cmap='gray')
    for i in range(10,20):
        plt.subplot(3,10,i+1)
        plt.imshow(noise_pixel[i], cmap='gray')
    for i in range(10,20):
        plt.subplot(3,10,i+11)
        plt.imshow(denoised_images[i], cmap='gray')
    image_name= "combined_digits.png"
    plt.tight_layout()
    plt.savefig(image_name)  # save the plot
    plt.close()

generate_cumulative_plot(binary_pixel, noise_pixel, denoised_image)

