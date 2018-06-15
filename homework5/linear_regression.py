import numpy as np
import glmnet_python
import pandas as pd
import matplotlib.pyplot as plt, warnings
import math
import statsmodels
import scipy
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from cvglmnet import cvglmnet
from cvglmnetPlot import cvglmnetPlot
from glmnet import glmnet
from glmnetPlot import glmnetPlot
from cvglmnetPredict import cvglmnetPredict

sns.set(color_codes=True)  # set the seaborn setting for plot


# function to read in the file into dataframe
def read_data(file, isHeader, sep="\t"):
    data = pd.read_csv(file, header=isHeader, sep=sep)
    return data


lm = linear_model.LinearRegression()  # build a model to do linear regression
##################################################################################################################
## Problem 1
##################################################################################################################

transformer = FunctionTransformer(np.log)  # use a transformer to change the coordinates to log
blood_data = read_data("problem1_data.txt", 0)  # read the data
transformed_blood_data = pd.DataFrame(transformer.transform(blood_data),
                                      columns=blood_data.columns)  # change coordinates to log

lm.fit(pd.DataFrame(transformed_blood_data.ix[:, 0]),
       pd.DataFrame(transformed_blood_data.ix[:, 1]))  # fir the regression model on the log cordinates
translated_x = np.exp(transformed_blood_data.ix[:, 0])  # get the translated independent variable
translated_y = np.exp(transformed_blood_data.ix[:, 1])  # get the translated dependent variable
translated_prediction = np.array(np.exp(
    lm.predict(pd.DataFrame(transformed_blood_data.ix[:, 0])))).flatten()  # get the translated predicted variable
translated_residuals = np.subtract(translated_y, translated_prediction)  # get the translated predicted value



# Part 1
sns.regplot(x='Hours', y='Sulfate',
            data=transformed_blood_data)  # create a regression plot suing seaborn in log coordinates
plt.title('Regression plot in log-log coordinates')
plt.tight_layout()
plt.savefig('problem7.9a.png')  # save the plot
plt.close()  # close the plot
r2Score = r2_score(transformed_blood_data.ix[:, 1], lm.predict(pd.DataFrame(transformed_blood_data.ix[:, 0])))
print("For problem 7.9a with coordinates in log-log format the r2 score is " + str(r2Score))



# Part2
plt.scatter(translated_x, translated_y)  # data points in the transformed original coordinates
plt.plot(translated_x, translated_prediction)  # regression curve in the transformed original coordinates
plt.title('Regression plot in original coordinates')
plt.xlabel("Hours")
plt.ylabel("Sulphate")
plt.tight_layout()
plt.savefig('problem7.9b.png')  # save the plot
plt.close()  # close the plot
r2Score = r2_score(translated_y, translated_prediction)
print("For problem 7.9b with coordinates in original format the r2 score is " + str(r2Score))



# Part3a
plt.scatter(lm.predict(pd.DataFrame(transformed_blood_data.ix[:, 0])),
            pd.DataFrame(transformed_blood_data.ix[:, 1]) - lm.predict(pd.DataFrame(transformed_blood_data.ix[:, 0])),
            c='g', s=50)  # residual plot in log-log coordinates
plt.title('Residual plot in log-log coordinates')
plt.xlabel("Fitted Value")
plt.ylabel("Residuals")
plt.hlines(y=0, xmin=1.5, xmax=2.6, colors='orange')
plt.tight_layout()
plt.savefig('problem7.9c_part1.png')  # save the plot
plt.close()  # close the plot


# Part3b
plt.scatter(translated_prediction, (translated_residuals), c='g',
            s=50)  # residual plot in transformed original coordinates
plt.title('Residual plot in original coordinates')
plt.xlabel("Fitted Value")
plt.ylabel("Residuals")
plt.hlines(y=0, xmin=5, xmax=13.7, colors='orange')
plt.tight_layout()
plt.savefig('problem7.9c_part2.png')  # save the plot
plt.close()  # close the plot




##################################################################################################################
## Problem 2
##################################################################################################################


#part 1
body_data = read_data("problem2_data.txt", 0)  # read in the data from body mass

lm.fit(body_data.ix[:, 1:], body_data.ix[:, 0])  # fit a regression predicting body mass using diameters

plt.scatter(lm.predict(body_data.ix[:, 1:]), body_data.ix[:, 0] - lm.predict(body_data.ix[:, 1:]), c='b',
            s=40)  # residual plot
plt.ylabel("Residual")
plt.xlabel("Fitted Body Mass")
plt.title('Residual plot in original coordinates')
plt.hlines(y=0, xmin=53, xmax=98, colors='orange')
plt.savefig('problem7.10a.png')  # save the plot
plt.close()  # close the plot

r2Score = r2_score(body_data.ix[:, 0], lm.predict(body_data.ix[:, 1:]))
print("For problem 7.10a with coordinates in original format the r2 score is " + str(r2Score))



# part 2a
do_cube_root = lambda x: np.sign(x) * np.power(abs(x), 1. / 3)  # function to do cube root
lm.fit(body_data.ix[:, 1:], do_cube_root(body_data.ix[:, 0]))  # cube root the mass

plt.scatter(lm.predict(body_data.ix[:, 1:]), do_cube_root(body_data.ix[:, 0]) - lm.predict(body_data.ix[:, 1:]), c='b',
            s=50)  # Residual plot in cubic root coordinates
plt.ylabel("Residual")
plt.xlabel("Fitted Body Mass in cubic root")
plt.title('Residual plot in cubic root coordinates')
plt.hlines(y=0, xmin=3.8, xmax=4.6, colors='orange')
plt.tight_layout()
plt.savefig('problem7.10b_part1.png')  # save the plot
plt.close()  # close the plot

r2Score = r2_score(do_cube_root(body_data.ix[:, 0]), lm.predict(body_data.ix[:, 1:]))
print(
    "For problem 7.10b part1 with coordinates in original format and predicted in cube root format the r2 score is " + str(
        r2Score))



# part 2b
plt.scatter(np.power(lm.predict(body_data.ix[:, 1:]), 3),
            (body_data.ix[:, 0]) - np.power(lm.predict(body_data.ix[:, 1:]), 3), c='b',
            s=50)  # Residual plot in original coordinates
plt.ylabel("Residual")
plt.xlabel("Fitted Body Mass in original coordinates")
plt.title('Residual plot in original coordinates')
plt.hlines(y=0, xmin=53, xmax=98, colors='orange')
plt.tight_layout()
plt.savefig('problem7.10b_part2.png')  # save the plot
plt.close()  # close the plot

r2Score = r2_score((body_data.ix[:, 0]), np.power(lm.predict(body_data.ix[:, 1:]), 3))
print(
    "For problem 7.10b part2 with coordinates in original format and predicted in original format the r2 score is " + str(
        r2Score))



##################################################################################################################
## Problem 3
##################################################################################################################
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                'Shell_weight', 'Rings'] # initialize column names
measurement_data = pd.read_csv("problem3_data.txt", header=None, sep=',')  # read in the data
measurement_data.columns = column_names  # set the column name in dataframe



#part 1
measurement_data_without_gender = measurement_data.ix[:, 1:] # exclude the gender column
measurement_data_without_gender_copy= measurement_data_without_gender.copy()

lm.fit(measurement_data_without_gender.ix[:, 0:7],
       measurement_data_without_gender.ix[:, -1])  # fit the model predicting age ignoring gender

plt.scatter(lm.predict(measurement_data_without_gender.ix[:, 0:7]),
            (measurement_data_without_gender.ix[:, -1]) - lm.predict(measurement_data_without_gender.ix[:, 0:7]), c='b',
            s=50)  # Residual plot without gender
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
plt.title('Residual plot without gender')
plt.hlines(y=0, xmin=-3, xmax=22, colors='orange')
plt.tight_layout()
plt.savefig('problem7.11a.png')  # save the plot
plt.close()  # close the plot

r2Score = r2_score(measurement_data_without_gender.ix[:, -1], lm.predict(measurement_data_without_gender.ix[:, 0:7]))
print("For problem 7.11a with gender ignored the r2 score is " + str(r2Score))



# Part 2
le = preprocessing.LabelEncoder()  # get the label encoder
le.fit(measurement_data.ix[:, 0])  # encode the gender category
encoded_gender = pd.DataFrame(le.transform(measurement_data.ix[:, 0]))  # get the encoded gender

measurement_data_with_gender_encoded = measurement_data_without_gender.copy()  # create a copy data without gender column
measurement_data_with_gender_encoded['Sex'] = encoded_gender  # add the gender column in encoded form

measurement_data_with_gender_encoded = measurement_data_with_gender_encoded[column_names]  # rearrange the column
measurement_data_with_gender_encoded_copy = measurement_data_with_gender_encoded.copy()  # create a copy of this data

lm.fit(measurement_data_with_gender_encoded.ix[:, 0:8],
       measurement_data_with_gender_encoded.ix[:, -1])  # fit the model predicting age including encoded gender
plt.scatter(lm.predict(measurement_data_with_gender_encoded.ix[:, 0:8]),
            (measurement_data_with_gender_encoded.ix[:, -1]) - lm.predict(
                measurement_data_with_gender_encoded.ix[:, 0:8]), c='b', s=40)
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
plt.title('Residual plot with encoded gender')
plt.hlines(y=0, xmin=-3, xmax=23, colors='orange')
plt.tight_layout()
plt.savefig('problem7.11b.png')  # save the plot
plt.close()  # close the plot

r2Score = r2_score(measurement_data_with_gender_encoded.ix[:, -1], lm.predict(measurement_data_with_gender_encoded.ix[:, 0:8]))
print("For problem 7.11b with gender the r2 score is " + str(r2Score))


# Part 3
measurement_data_without_gender['Rings'] = np.log(measurement_data_without_gender.Rings)  # take the log of age
lm.fit(measurement_data_without_gender.ix[:, 0:7], measurement_data_without_gender.ix[:, -1])  # fit the model
plt.scatter(lm.predict(measurement_data_without_gender.ix[:, 0:7]),
            (measurement_data_without_gender.ix[:, -1]) - lm.predict(measurement_data_without_gender.ix[:, 0:7]), c='b',
            s=40)
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
plt.title('Residual plot without gender and log of dependent variable')
plt.hlines(y=0, xmin=1.0, xmax=3.3, colors='orange')
plt.tight_layout()
plt.savefig('problem7.11c.png')  # save the plot
plt.close()  # close the plot

r2Score = r2_score(measurement_data_without_gender.ix[:, -1], lm.predict(measurement_data_without_gender.ix[:, 0:7]))
print("For problem 7.11c without gender and log of dependent variable the r2 score is " + str(r2Score))


# Part 4
measurement_data_with_gender_encoded['Rings'] = np.log(
    measurement_data_with_gender_encoded.Rings)  # take the log of age
lm.fit(measurement_data_with_gender_encoded.ix[:, 0:8], measurement_data_with_gender_encoded.ix[:, -1])  # fit the model
plt.scatter(lm.predict(measurement_data_with_gender_encoded.ix[:, 0:8]),
            (measurement_data_with_gender_encoded.ix[:, -1]) - lm.predict(
                measurement_data_with_gender_encoded.ix[:, 0:8]), c='b', s=40)
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
plt.title('Residual plot with encoded gender and log of dependent variable')
plt.hlines(y=0, xmin=1.0, xmax=3.3, colors='orange')
plt.tight_layout()
plt.savefig('problem7.11d.png')  # save the plot
plt.close()  # close the plot

r2Score = r2_score(measurement_data_with_gender_encoded.ix[:, -1], lm.predict(measurement_data_with_gender_encoded.ix[:, 0:8]))
print("For problem 7.11d with gender and log of dependent variable the r2 score is " + str(r2Score))

# Part 6

x = np.array(measurement_data_without_gender).astype(scipy.float64) # change the data type to float64
cvfit_lasso = cvglmnet(x=x[:, :7].copy(), y=x[:, -1].copy(), ptype='mse', nfolds=10, alpha=1) # create a lasso regression
cvglmnetPlot(cvfit_lasso) # plot the lasso regression
plt.title("Lasso regularisation",y=1.14)
plt.tight_layout()
plt.savefig('problem7.11f_lasso.png')  # save the plot
plt.close()  # close the plot
print("The value of lambda giving minimum mse for lasso is "+str(cvfit_lasso['lambda_min']))


pred_lasso=cvglmnetPredict(cvfit_lasso, newx = measurement_data_without_gender.ix[:, 0:7].copy(), s='lambda_min').flatten() # get the predicted values
r2Score = r2_score(measurement_data_without_gender.ix[:, -1], pred_lasso)
print("For problem 7.11f with gender and lasso regression the r2 score is " + str(r2Score))

###########################################################################################################

cvfit_ridge = cvglmnet(x=x[:, :7].copy(), y=x[:, -1].copy(), ptype='mse', nfolds=10, alpha=0) # create a ridge regression
cvglmnetPlot(cvfit_ridge) # plot the ridge regression
plt.title("Ridge regularisation",y=1.14)
plt.tight_layout()
plt.savefig('problem7.11f_ridge.png')  # save the plot
plt.close()  # close the plot

print("The value of lambda giving minimum mse for ridge is "+str(cvfit_ridge['lambda_min'])) # set the min lambda

pred_ridge=cvglmnetPredict(cvfit_ridge, newx = measurement_data_without_gender.ix[:, 0:7].copy(), s='lambda_min').flatten() # get the predicted values
r2Score = r2_score(measurement_data_without_gender.ix[:, -1].copy(), pred_ridge)
print("For problem 7.11f with gender and ridge regression the r2 score is " + str(r2Score))


################################################################################################################
# Generating glmnet plots by applying regularisation on each model


# model a
a = np.array(measurement_data_without_gender_copy).astype(scipy.float64)
cvfit = cvglmnet(x=a[:, 0:7].copy(), y=a[:, -1].copy(), ptype='mse', nfolds=10, alpha=0)
cvglmnetPlot(cvfit)  # plot the ridge regression
plt.title("Ridge regularisation for model 7.11a",y=1.14)
plt.tight_layout()
plt.savefig('problem7.11f_parta.png')  # save the plot
plt.close()  # close the plot

pred_ridge=cvglmnetPredict(cvfit, newx = measurement_data_without_gender_copy.ix[:, 0:7].copy(), s='lambda_min').flatten() # get the predicted values
r2Score = r2_score(measurement_data_without_gender_copy.ix[:, -1].copy(), pred_ridge)
print("For problem 7.11f model a the r2 score is " + str(r2Score))

# model b
x = np.array(measurement_data_with_gender_encoded_copy).astype(scipy.float64)
cvfit_ridge = cvglmnet(x=x[:, 0:8].copy(), y=x[:, -1].copy(), ptype='mse', nfolds=10, alpha=0)
cvglmnetPlot(cvfit_ridge)  # plot the ridge regression
plt.title("Ridge regularisation for model 7.11b",y=1.14)
plt.tight_layout()
plt.savefig('problem7.11f_partb.png')  # save the plot
plt.close()  # close the plot

pred_ridge=cvglmnetPredict(cvfit_ridge, newx = measurement_data_with_gender_encoded_copy.ix[:, 0:8].copy(), s='lambda_min').flatten() # get the predicted values
r2Score = r2_score(measurement_data_with_gender_encoded_copy.ix[:, -1].copy(), pred_ridge)
print("For problem 7.11f model b the r2 score is " + str(r2Score))

# model c
x = np.array(measurement_data_without_gender).astype(scipy.float64)
cvfit_ridge = cvglmnet(x=x[:, 0:7].copy(), y=x[:, -1].copy(), ptype='mse', nfolds=10, alpha=0)
cvglmnetPlot(cvfit_ridge)  # plot the ridge regression
plt.title("Ridge regularisation for model 7.11c",y=1.14)
plt.tight_layout()
plt.savefig('problem7.11f_partc.png')  # save the plot
plt.close()  # close the plot

pred_ridge=cvglmnetPredict(cvfit_ridge, newx = measurement_data_without_gender.ix[:, 0:7].copy(), s='lambda_min').flatten() # get the predicted values
r2Score = r2_score(measurement_data_without_gender.ix[:, -1].copy(), pred_ridge)
print("For problem 7.11f model c the r2 score is " + str(r2Score))

# model d
x = np.array(measurement_data_with_gender_encoded).astype(scipy.float64)
cvfit_ridge = cvglmnet(x=x[:, 0:8].copy(), y=x[:, -1].copy(), ptype='mse', nfolds=10, alpha=0)
cvglmnetPlot(cvfit_ridge)  # plot the ridge regression
plt.title("Ridge regularisation for model 7.11d",y=1.14)
plt.tight_layout()
plt.savefig('problem7.11f_partd.png')  # save the plot
plt.close()  # close the plot

pred_ridge=cvglmnetPredict(cvfit_ridge, newx = measurement_data_with_gender_encoded.ix[:, 0:8].copy(), s='lambda_min').flatten() # get the predicted values
r2Score = r2_score(measurement_data_with_gender_encoded.ix[:, -1].copy(), pred_ridge)
print("For problem 7.11f model d the r2 score is " + str(r2Score))