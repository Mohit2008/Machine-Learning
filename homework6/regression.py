import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
import seaborn as sns
import matplotlib.pyplot as plt, warnings
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split
import scipy
import glmnet_python
from cvglmnet import cvglmnet
from cvglmnetPlot import cvglmnetPlot
from glmnet import glmnet
from glmnetPlot import glmnetPlot
from cvglmnetPredict import cvglmnetPredict
from cvglmnetCoef import cvglmnetCoef
import xlrd
import warnings

warnings.filterwarnings("ignore")
sns.set()
np.random.seed(123)
####################################################################################################################
# Problem1
####################################################################################################################

music_data = pd.read_csv(
    "/home/khanna/codebase/ml-practice/homework6/Geographical Original of Music/Geographical Original of Music/default_plus_chromatic_features_1059_tracks.txt",
    header=None, sep=',')  # read in the music data

ncol = np.shape(music_data)[1]  # get the no of column of data frame
features = music_data.ix[:, :ncol - 3]  # get all the features
latitude = music_data.ix[:, ncol - 2] + 90  # get the latitude column and add 90 to make it positive
longitude = music_data.ix[:, ncol - 1] + 180  # get the longitude column and add 180 to make it positive


# the function can be used to generate a model for doing liner regression
def generate_model(features, labels, type):
    lm = linear_model.LinearRegression()
    lm.fit(features, labels)  # fit the model
    prediction = lm.predict(features)  # get predictions
    return prediction


# the function can be used to generate residual plots
def generate_residual_plot(label, prediction, type):
    plt.scatter(prediction, np.subtract(label, prediction))  # scatter plot
    title = 'Residual plot for predicting ' + type
    plt.title(title)  # set title
    plt.xlabel("Fitted Value")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.hlines(y=0, xmin=min(prediction), xmax=max(prediction), colors='orange', linewidth=3)  # plot ref line


# function that can be used to generate a scatter plot of actual vs prediction values
def generate_actual_vs_predicted_plot(label, prediction, type):
    plt.scatter(prediction, label, s=30, c='r', marker='+', zorder=10)  # scatter plot
    title = 'Actual vs Predicted values for ' + type
    plt.title(title)  # set title
    plt.xlabel("Predicted Values from model")  # set the xlabel
    plt.ylabel("Actual Values")  # set the ylabel
    plt.tight_layout()

def generate_box_cox_plot(input, min_lambda):
    lmbdas = np.linspace(-2, 10)
    llf = np.zeros(lmbdas.shape, dtype=float)
    for ii, lmbda in enumerate(lmbdas):
        llf[ii] = stats.boxcox_llf(lmbda, input)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lmbdas, llf, 'b.-')
    ax.axhline(stats.boxcox_llf(min_lambda, input), color='r')
    ax.set_xlabel('lambda parameter')
    ax.set_ylabel('Box-Cox log-likelihood')
    plt.tight_layout()

def get_aic_bic(output, predictions, no_of_features, no_of_observation):
    resid = output - predictions
    sse = sum(resid ** 2)
    aic = 2*(no_of_features)-2*np.log(sse)
    bic = no_of_observation * np.log(sse / no_of_observation) + no_of_features * np.log(no_of_observation)
    return aic,bic



###########################################################################################################################
# Problem1 Part 1
###########################################################################################################################

################## Do for latitude
print(
    "----------------------------------------------------------------------------------------------------------------")
print("Problem 1 part 1 started")
print(
    "----------------------------------------------------------------------------------------------------------------")
print("--------------Latitude--------------")
plt.figure(figsize=(1200 / 96, 1200 / 96), dpi=96)  # initialize a canvas
latitude_prediction = generate_model(features, latitude, "latitude")  # get the linear regression model for latitude
print("Mean square error for regression model to predict latitude is " + str(
    mean_squared_error(latitude, latitude_prediction)))  # print out mse
print("R squared value for regression model to predict latitude is " + str(
    r2_score(latitude, latitude_prediction)))  # print out r2
plt.subplot(2, 1, 1)
generate_residual_plot(latitude, latitude_prediction,
                       "latitude")  # generate the residual plot using the helper function
plt.subplot(2, 1, 2)
generate_actual_vs_predicted_plot(latitude, latitude_prediction,
                                  "latitude")  # generate the actual vs predicted plot using the helper function

plt.savefig("Latitude_regression.png")  # save the image
plt.close()  # close the canvas

aic,bic=get_aic_bic(latitude, latitude_prediction, np.shape(features)[1],np.shape(features)[0])
print("For latitude for regression the aic is = "+str(aic)+" and bic is = "+str(bic))

#####################Do for longitude

print("---------------------Longitude--------------------")
plt.figure(figsize=(1200 / 96, 1200 / 96), dpi=96)  # initialize a canvas
longitude_prediction = generate_model(features, longitude, "longitude")  # get the linear regression model for longitude
print("Mean square error for regression model to predict longitude is " + str(
    mean_squared_error(longitude, longitude_prediction)))  # print out mse
print("R squared value for regression model to predict longitude is " + str(
    r2_score(longitude, longitude_prediction)))  # print out r2
plt.subplot(2, 1, 1)
generate_residual_plot(longitude, longitude_prediction,
                       "longitude")  # generate the residual plot using the helper function
plt.subplot(2, 1, 2)
generate_actual_vs_predicted_plot(longitude, longitude_prediction,
                                  "longitude")  # generate the actual vs predicted plot using the helper function
plt.savefig("Longitude_regression.png")  # save the image
plt.close()  # close the canvas

aic,bic=get_aic_bic(longitude, longitude_prediction, np.shape(features)[1],np.shape(features)[0])
print("For longitude for regression the aic is = "+str(aic)+" and bic is = "+str(bic))

##########################################################################################################################
# Problem1 Part 2
##########################################################################################################################
print(
    "----------------------------------------------------------------------------------------------------------------")
print("Problem 1 part 2 started")
print(
    "----------------------------------------------------------------------------------------------------------------")

transformed_latitude, latitude_lambda = stats.boxcox(latitude)  # do a box cox transformation on latitude
transformed_longitude, longitude_lambda = stats.boxcox(longitude)  # do a box cox transformation on longitude

print("Lambda values that maximises the log likelihood of latitude is " + str(latitude_lambda))  # get the best lambda
print("Lambda values that maximises the log likelihood of longitude is " + str(longitude_lambda))  # get the best lambda

generate_box_cox_plot(latitude, latitude_lambda) # generate log likelihood vs best lambda for latitude
plt.savefig("Latitude_box_cox.png")  # save the image
plt.close()  # close the canvas

generate_box_cox_plot(longitude, longitude_lambda)# generate log likelihood vs best lambda for longitude
plt.savefig("Longitude_box_cox.png")  # save the image
plt.close()  # close the canvas

################## Do for latitude

print("---------------Latitude-------------------")
plt.figure(figsize=(1200 / 96, 1200 / 96), dpi=96)  # initialize a canvas
transformed_latitude_prediction = inv_boxcox(generate_model(features, transformed_latitude, "transformed_latitude"),
                                             latitude_lambda)  # get the latitude predictions in original cordinate

print("Mean square error for regression model to predict transformed_latitude is " + str(
    mean_squared_error(latitude, transformed_latitude_prediction)))  # get mse
print("R squared value for regression model to predict transformed_latitude is " + str(
    r2_score(latitude, transformed_latitude_prediction)))  # get r2
plt.subplot(2, 1, 1)
generate_residual_plot(latitude, transformed_latitude_prediction,
                       "transformed_latitude")  # generate the residual plot using the helper function
plt.subplot(2, 1, 2)
generate_actual_vs_predicted_plot(latitude, transformed_latitude_prediction,
                                  "transformed_latitude")  # generate the actual vs predicted plot using the helper function
plt.savefig("Transformed_Latitude_regression.png")  # save the image
plt.close()  # close the canvas

aic,bic=get_aic_bic(latitude, transformed_latitude_prediction, np.shape(features)[1],np.shape(features)[0])
print("For transformed latitude for regression the aic is = "+str(aic)+" and bic is = "+str(bic))

################## Do for longitude

print("-------------------Longitude---------------------")
plt.figure(figsize=(1200 / 96, 1200 / 96), dpi=96)  # initialize a canvas
transformed_longitude_prediction = inv_boxcox(generate_model(features, transformed_longitude, "transformed_longitude"),
                                              longitude_lambda)  # get the longitude predictions in original cordinate

print("Mean square error for regression model to predict transformed_longitude is " + str(
    mean_squared_error(longitude, transformed_longitude_prediction)))  # get mse
print("R squared value for regression model to predict transformed_longitude is " + str(
    r2_score(longitude, transformed_longitude_prediction)))  # get r2
plt.subplot(2, 1, 1)
generate_residual_plot(longitude, transformed_longitude_prediction,
                       "transformed_longitude")  # generate the residual plot using the helper function
plt.subplot(2, 1, 2)
generate_actual_vs_predicted_plot(longitude, transformed_longitude_prediction,
                                  "transformed_longitude")  # generate the actual vs predicted plot using the helper function
plt.savefig("Transformed_Longitude_regression.png")  # save the image
plt.close()  # close the canvas

aic,bic=get_aic_bic(longitude, transformed_longitude_prediction, np.shape(features)[1],np.shape(features)[0])
print("For transformed longitude for regression the aic is = "+str(aic)+" and bic is = "+str(bic))

##########################################################################################################################
# Problem1 Part 3 Ridge Regularisation
##########################################################################################################################
print(
    "----------------------------------------------------------------------------------------------------------------")
print("Problem 1 part 3a started")
print(
    "----------------------------------------------------------------------------------------------------------------")
music_data = np.array(music_data).astype(scipy.float64)  # change the data type to float64
ncol = np.shape(music_data)[1]  # get no od columns
features = music_data[:, :ncol - 3]  # get the features
latitude = music_data[:, ncol - 2] + 90  # get the latitude column and add 80 to make it positive
longitude = music_data[:, ncol - 1] + 180  # get the longitude column and add 180 to make it positive

# Part 3a

################## Do for latitude

print("-----------Latitude-------------------")
plt.figure(figsize=(1200 / 96, 1200 / 96), dpi=96)  # initialize a canvas
plt.subplot(2, 2, 1)
cvfit_ridge = cvglmnet(x=features.copy(), y=latitude.copy(), ptype='mse', nfolds=20,
                       alpha=0)  # create a ridge regression

coff_count = sum(1 for x in cvglmnetCoef(cvfit_ridge) if x != 0)  # get the no of non zero weights
cvglmnetPlot(cvfit_ridge)  # plot the ridge regression
plt.title("Ridge on latitude", y=1.10)
print("The value of lambda giving minimum mse for ridge for latitude is " + str(
    (cvfit_ridge['lambda_min'])))  # min lambda
print("The value of lambda giving 1se mse for ridge for latitude is " + str(
    (cvfit_ridge['lambda_1se'])))  # 1se lambda
print("The cross validated error for ridge for latitude is " + str(
    np.mean(cvfit_ridge['cvm'])))  # cv error


pred_ridge = cvglmnetPredict(cvfit_ridge, newx=features.copy(), s='lambda_min').flatten()  # get the predicted values
r2Score = r2_score(latitude.copy(), pred_ridge)  # get r2
print("r2 score for ridge regularisation for latitude is " + str(r2Score))
print("Mean square error for regression model to predict latitude with ridge regularisation is " + str(
    mean_squared_error(latitude, pred_ridge)))  # get mse

print("The no of explanatory variables used for ridge regularisation for latitude is " + str(coff_count))

plt.subplot(2, 2, 3)
plt.scatter(pred_ridge, np.subtract(latitude, pred_ridge))  # residual plot
plt.tight_layout()
plt.hlines(y=0, xmin=min(pred_ridge), xmax=max(pred_ridge), colors='orange', linewidth=3)
plt.title("Residual plot")
plt.xlabel("Fitted Value")
plt.ylabel("Residuals")

################## Do for longitude

print("----------------Longitude---------------------")
plt.subplot(2, 2, 2)
cvfit_ridge = cvglmnet(x=features.copy(), y=longitude.copy(), ptype='mse', nfolds=20,
                       alpha=0)  # create a ridge regression
cvglmnetPlot(cvfit_ridge)  # plot the ridge regression
plt.title("Ridge on longitude", y=1.10)
print(
    "The value of lambda giving minimum mse for ridge for longitude is " + str(cvfit_ridge['lambda_min']))  # min lambda

print("The value of lambda giving 1se mse for ridge for longitude is " + str(
    (cvfit_ridge['lambda_1se'])))  # 1se lambda

print("The cross validated error for ridge for longitude is " + str(
    np.mean(cvfit_ridge['cvm'])))  # cv error

pred_ridge = cvglmnetPredict(cvfit_ridge, newx=features.copy(), s='lambda_min').flatten()  # get the predicted values
r2Score = r2_score(longitude.copy(), pred_ridge)  # r2 score
print("r2 score for ridge regularisation for longitude is " + str(r2Score))

print("Mean square error for regression model to predict longitude with ridge regularisation is " + str(
    mean_squared_error(longitude, pred_ridge)))  # get mse
coff_count = sum(1 for x in cvglmnetCoef(cvfit_ridge) if x != 0)  # get the no of non zero weights
print("The no of explanatory variables used for ridge regularisation for longitude is " + str(coff_count))

plt.subplot(2, 2, 4)
plt.scatter(pred_ridge, np.subtract(longitude, pred_ridge))  # residual plot
plt.tight_layout()
plt.hlines(y=0, xmin=min(pred_ridge), xmax=max(pred_ridge), colors='orange', linewidth=3)
plt.title("Residual plot")
plt.xlabel("Fitted Value")
plt.ylabel("Residuals")
plt.savefig("Ridge_regression_latitude_longitude.png")  # save the plot
plt.close()  # close the plot

# ######################################################################################################################
# # Problem1 Part 3b Lasso Regularisation
########################################################################################################################

################## Do for latitude
print(
    "----------------------------------------------------------------------------------------------------------------")
print("Problem 1 part 3b started")
print(
    "----------------------------------------------------------------------------------------------------------------")
print("---------------Latitude-----------------")
plt.figure(figsize=(1200 / 96, 1200 / 96), dpi=96)  # initialize a canvas
plt.subplot(2, 2, 1)
cvfit_lasso = cvglmnet(x=features.copy(), y=latitude.copy(), ptype='mse', nfolds=20,
                       alpha=1)  # create a lasso regression
cvglmnetPlot(cvfit_lasso)  # plot the ridge regression
plt.title("Lasso on latitude", y=1.10)
print(
    "The value of lambda giving minimum mse for lasso for latitude is " + str(cvfit_lasso['lambda_min']))  # min lambda
print("The value of lambda giving 1se mse for lasso for latitude is " + str(
    (cvfit_lasso['lambda_1se'])))  # 1se lambda

print("The cross validated error for lasso for latitude is " + str(
    np.mean(cvfit_lasso['cvm'])))  # cv error
pred_lasso = cvglmnetPredict(cvfit_lasso, newx=features.copy(), s='lambda_min').flatten()  # get the predicted values
r2Score = r2_score(latitude.copy(), pred_lasso)  # get r2
print("r2 score for lasso regularisation for latitude is " + str(r2Score))

print("Mean square error for regression model to predict latitude with lasso regularisation is " + str(
    mean_squared_error(latitude, pred_lasso)))  # get mse

coff_count = sum(1 for x in cvglmnetCoef(cvfit_lasso) if x != 0)  # get the no of non zero weights
print("The no of explanatory variables used for lasso regularisation for latitude is " + str(coff_count))
plt.subplot(2, 2, 3)
plt.scatter(pred_lasso, np.subtract(latitude, pred_lasso))  # residual plot
plt.tight_layout()
plt.hlines(y=0, xmin=min(pred_lasso), xmax=max(pred_lasso), colors='orange', linewidth=3)
plt.title("Residual plot")
plt.xlabel("Fitted Value")
plt.ylabel("Residuals")

################## Do for longitude


print("---------------------Longitude-------------------")
plt.subplot(2, 2, 2)
cvfit_lasso = cvglmnet(x=features.copy(), y=longitude.copy(), ptype='mse', nfolds=20,
                       alpha=1)  # create a lasso regression
cvglmnetPlot(cvfit_lasso)  # plot the ridge regression
plt.title("Lasso on longitude", y=1.10)
print(
    "The value of lambda giving minimum mse for lasso for longitude is " + str(cvfit_lasso['lambda_min']))  # min lambda

print("The value of lambda giving 1se mse for lasso for longitude is " + str(
    (cvfit_lasso['lambda_1se'])))  # 1se lambda
print("The cross validated error for lasso for longitude is " + str(
    np.mean(cvfit_lasso['cvm'])))  # cv error
pred_lasso = cvglmnetPredict(cvfit_lasso, newx=features.copy(), s='lambda_min').flatten()  # get the predicted values
r2Score = r2_score(longitude.copy(), pred_lasso)  # get r2
print("r2 score for lasso regularisation for longitude is " + str(r2Score))

print("Mean square error for regression model to predict longitude with lasso regularisation is " + str(
    mean_squared_error(longitude, pred_lasso)))  # get mse

coff_count = sum(1 for x in cvglmnetCoef(cvfit_lasso) if x != 0)  # get the no of non zero weights
print("The no of explanatory variables used for lasso regularisation for longitude is " + str(coff_count))
plt.subplot(2, 2, 4)
plt.scatter(pred_lasso, np.subtract(longitude, pred_lasso))  # residual plot
plt.tight_layout()
plt.hlines(y=0, xmin=min(pred_lasso), xmax=max(pred_lasso), colors='orange', linewidth=3)
plt.title("Residual plot")
plt.xlabel("Fitted Value")
plt.ylabel("Residuals")

plt.savefig("Lasso_regression_latitude_longitude.png")  # save the plot
plt.close()  # close the plot

# ####################################################################################################################
# # Problem1 Part 3c
######################################################################################################################

################## Do for latitude
print(
    "----------------------------------------------------------------------------------------------------------------")
print("Problem 1 part 3c started")
print(
    "----------------------------------------------------------------------------------------------------------------")
print("-------------------Latitude--------------------")
alpha_list = [0.25, 0.5, 0.75]  # set 3 aplha values
plt.figure(figsize=(1600 / 96, 1600 / 96), dpi=96)  # initialize a canvas

for row, i in enumerate(alpha_list):  # iterate through alpha values
    plt.subplot(2, 3, row + 1)
    cvfit_elastic = cvglmnet(x=features.copy(), y=latitude.copy(), ptype='mse', nfolds=20,
                             alpha=i)  # create a elastic net regression
    cvglmnetPlot(cvfit_elastic)  # plot the elastic net regression
    title = "Elasticnet with " + str(i)
    plt.title(title, y=1.10)
    print("The value of lambda giving minimum mse for elastic net for latitude at aplha= " + str(i) + " is " + str(
        cvfit_elastic['lambda_min']))  # min lambda
    print("The value of lambda giving 1se mse for elastic net for latitude at aplha= " + str(i) + " is " + str(
        (cvfit_elastic['lambda_1se'])))  # 1se lambda
    print("The cross validated error for elastic net for latitude at aplha= " + str(i) + " is " + str(
        np.mean(cvfit_elastic['cvm'])))  # cv error
    pred_elastic = cvglmnetPredict(cvfit_elastic, newx=features.copy(),
                                   s='lambda_min').flatten()  # get the predicted values
    r2Score = r2_score(latitude.copy(), pred_elastic)  # get r2
    print("r2 score for elastic net regularisation for latitude at alpha= " + str(i) + " is " + str(r2Score))
    print("mse for elastic net regularisation for latitude at alpha= " + str(i) + " is " + str(
        mean_squared_error(latitude, pred_elastic)))  # get mse
    coff_count = sum(1 for x in cvglmnetCoef(cvfit_elastic) if x != 0)  # get the no of non zero weights
    print("The no of explanatory variables used for elastic-net regularisation for latitude at alpha= " + str(
        i) + " is " + str(coff_count))
    plt.subplot(2, 3, row + 4)
    plt.scatter(pred_elastic, np.subtract(latitude, pred_elastic))  # residual plot
    plt.tight_layout()
    plt.hlines(y=0, xmin=min(pred_elastic), xmax=max(pred_elastic), colors='orange', linewidth=3)
    plt.title("Residual plot")
    plt.xlabel("Fitted Value")
    plt.ylabel("Residuals")
    print("---------------")

plt.savefig("Elastic_net_regression_latitude.png")  # save the plot
plt.close()  # close the plot

################## Do for longitude

print("-------------------Longitude-------------------")
plt.figure(figsize=(1600 / 96, 1600 / 96), dpi=96)  # initialize a canvas
for row, i in enumerate(alpha_list):  # iterate through alpha values
    plt.subplot(2, 3, row + 1)
    cvfit_elastic = cvglmnet(x=features.copy(), y=longitude.copy(), ptype='mse', nfolds=20,
                             alpha=i)  # create a elastic net regression
    cvglmnetPlot(cvfit_elastic)  # plot the elastic net regression
    title = "Elasticnet with " + str(i)
    plt.title(title, y=1.10)
    print("The value of lambda giving minimum mse for elastic net for longitude at aplha= " + str(i) + " is " + str(
        cvfit_elastic['lambda_min']))  # min lambda

    print("The value of lambda giving 1se mse for elastic net for longitude at aplha= " + str(i) + " is " + str(
        (cvfit_elastic['lambda_1se'])))  # 1se lambda
    print("The cross validated error for elastic net for longitude at aplha= " + str(i) + " is " + str(
        np.mean(cvfit_elastic['cvm'])))  # cv error
    pred_elastic = cvglmnetPredict(cvfit_elastic, newx=features.copy(),
                                   s='lambda_min').flatten()  # get the predicted values
    r2Score = r2_score(longitude.copy(), pred_elastic)  # get r2
    print("r2 score for elastic net regularisation for longitude at alpha= " + str(i) + " is " + str(r2Score))
    print("mse for elastic net regularisation for longitude at alpha= " + str(i) + " is " + str(
        mean_squared_error(longitude, pred_elastic)))  # get mse
    coff_count = sum(1 for x in cvglmnetCoef(cvfit_elastic) if x != 0.0)  # get the no of non zero weights
    print("The no of explanatory variables used for elastic-net regularisation for longitude at alpha= " + str(
        i) + " is " + str(coff_count))
    plt.subplot(2, 3, row + 4)
    plt.scatter(pred_elastic, np.subtract(longitude, pred_elastic))  # residual plot
    plt.tight_layout()
    plt.hlines(y=0, xmin=min(pred_elastic), xmax=max(pred_elastic), colors='orange', linewidth=3)
    plt.title("Residual plot")
    plt.xlabel("Fitted Value")
    plt.ylabel("Residuals")
    print("-------------------")

plt.savefig("Elastic_net_regression_longitude.png")  # save the plot
plt.close()  # close the plot

# ####################################################################################################################
#
# ####################################################################################################################
# # Problem2
# ####################################################################################################################

print("Problem 2 started")
print(
    "----------------------------------------------------------------------------------------------------------------")

credit_data = pd.read_excel("default of credit card clients.xls", header=1)  # read the data

credit_data = np.array(credit_data).astype(scipy.float64)  # change the data type to float64
features = credit_data[:, 1:24]  # get the features
label = credit_data[:, 24]  # get the labels
# X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.20, random_state=123) # split the data


logreg = LogisticRegression()  # initialize a logistic regression model
predicted = cross_validation.cross_val_predict(logreg, features, label, cv=10)
# logreg.fit(X_train, y_train)  # fit the model
# predic = logreg.predict(X_test)  # predict the model

print("Accuracy score  for logistic regression model is " + str(metrics.accuracy_score(label, predicted.round())))  # get accuracy
print("Cross validated error for logistic regression model is " + str(1-(metrics.accuracy_score(label, predicted.round()))))

alpha_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # list of alpha values to try from
plt.figure(figsize=(1500 / 96, 1500 / 96), dpi=96)  # set the canvas
for row, i in enumerate(alpha_list):
    plt.subplot(2, 3, row + 1)
    title = "Regularisation with " + str(i)
    plt.title(title, y=1.10)
    cvfit = cvglmnet(x=features.copy(), y=label.copy(), family='binomial', ptype='class', alpha=i)  # do elastic net
    cvglmnetPlot(cvfit)
    plt.tight_layout()
    pred = cvglmnetPredict(cvfit, newx=features.copy(),
                           s='lambda_min', ptype='class').flatten()  # get the predicted values
    print("Accuracy score  for logistic regression model with regularisation at alpha= " + str(i) + " is " + str(
        accuracy_score(label, pred.round())))  # get accuracy
    print("The value of lambda giving minimum mse for elastic net at aplha= " + str(i) + " is " + str(
        cvfit['lambda_min']))  # min lambda
    print("The value of lambda giving 1se mse for elastic net at aplha= " + str(i) + " is " + str(i) + " is " + str(
        (cvfit['lambda_1se'])))  # 1se lambda
    coff_count = sum(1 for x in cvglmnetCoef(cvfit) if x != 0.0)  # get the no of non zero weights
    print("The no of explanatory variables used for elastic-net regularisation at aplha= " + str(i) + " is " + str(
        i) + " is " + str(coff_count))
    print("The cross validated error for elastic net regularisation at aplha= " + str(i) + " is " + str(i) + " is " + str(
        np.mean(cvfit['cvm'])))  # cv error
    print("-----------------")

plt.savefig("Logistic_regression_with_different_regularisation")  # save the plot
plt.close()  # close the plot
