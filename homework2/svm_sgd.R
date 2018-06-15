library(klaR)
library(caret)
library(e1071)
set.seed(43213)

#Read in the data
income_data_train <- read.csv('/home/khanna/codebase/ml-practice/homework2/income_data.csv', header=FALSE, na.strings = " ?")
income_data_test <- read.csv('/home/khanna/codebase/ml-practice/homework2/income_data_test.csv', header=FALSE, na.strings = " ?")
#Combine both the train and test set to have total of 48842 observations
full_dataset <- rbind(income_data_train, income_data_test)
# Remove rows having na making total observation to 45222 observations
clean_dataset<- na.omit(full_dataset)
# Take only continous features
x_vector <- clean_dataset[,c(1,3,5,11,12,13)]
# form the label vector
y_labels <- clean_dataset[,c(15)]

#Function that will scale in the values by bringing mean =0 and sd=1
normalise_feature<-function(feature){
  for (i in 1:6){
    feature[i] <- scale(as.numeric(as.matrix(feature[,i])))
  }
  return(feature)
}
# Encoding the labels from the original dataset
encode_y <- function(y){
  if(y == " <=50K" | y == " <=50K."){
    return (-1)
  }
  else{
    return (1)
  }
}
# Get the encoded label vector
encoded_y = sapply(y_labels, encode_y)
# Get the scaled feature matrix
scaled_feature= normalise_feature(x_vector)

#split data into train, test, and validation set
split_1 <- createDataPartition(y=encoded_y, p=.8, list=FALSE)
train_X <- scaled_feature[split_1,]
train_Y <- encoded_y[split_1]
interim_X <- scaled_feature[-split_1,]
interim_Y <- encoded_y[-split_1]
split_2 <- createDataPartition(y=interim_Y, p=.5, list=FALSE)
test_X <- interim_X[split_2,]
test_Y <- interim_Y[split_2]
validate_X <- interim_X[-split_2,]
validate_Y <- interim_Y[-split_2]

# Function to get the predicted value based upon model parameters 
predict <- function(x, model_coef, intercept){
  new_x <- as.numeric(as.matrix(x))
  return (t(model_coef) %*% new_x + intercept) 
}
# Calculate accuracy of the current model
accuracy <- function(x,y,model_coef,intercept){
  predict<-c()
  for (i in 1:length(y)){
    val <- predict(x[i,], model_coef, intercept)
    if(val >= 0){
      predict[i]=1
    }
    else{
      predict[i]=-1
    }
  }
  gotright<-predict == y
  return(sum(gotright)/(sum(gotright)+sum(!gotright)))
}
# Get the magnitude of the cofficents vector
norm_vec <- function(x) sqrt(sum(x^2))

validation_accuracies = c()
test_accuracies = c()
step_a <- .01
step_b <- 50
lambda_vector<-c(.001, .01, .1, 1)
# Iterate through each lambda value
for (lambda in lambda_vector){
  model_coef <- c(0.0,0.0,0.0,0.0,0.0,0.0)
  intercept <- 0
  season_accuracy <- c()
  magnitude_vector <-c()
  # Go through each epoch 
  for (epoch in 1:50){
    # Randomnly sample 50 exaples and held that out 
    sam <- sample(1:dim(train_X)[1], 50)
    accuracy_data <- train_X[sam, ]
    accuracy_labels <- train_Y[sam]
    # Train on rest of the data
    train_data <- train_X[-sam,]
    train_labels <- train_Y[-sam]
    # Iterate through each epoch
    for (step in 1:300){
      # Sample randomly one training example
      k <- sample(1:length(train_labels), 1)
      # Form feature matrix
      feature_x <- as.numeric(as.matrix( train_data[k,] ))
      # Form label matrix
      label_y <-  as.numeric(as.matrix(train_labels[k] ))
      # Do prediction with current model parameters
      pred <- predict(feature_x, model_coef, intercept)
      # Calculate steplength
      steplength = 1 / ((step_a * epoch) + step_b)
      # Update the model parameters based on condition
      if(label_y * pred >= 1){
        model_coef <- model_coef - (steplength * lambda * model_coef)
        intercept <- intercept - (steplength * 0)
      } else {
        model_coef <- model_coef - (steplength * ((lambda * model_coef) - (label_y * feature_x)))
        intercept <- intercept - (steplength * -label_y)
      }
      # For every season(30 steps) get the season accuracy used for plotting
      if(step %% 30 == 0){
        calc <- accuracy(accuracy_data, accuracy_labels, model_coef, intercept)
        season_accuracy <- c(season_accuracy, calc)
        magnitude=norm_vec(model_coef)
        magnitude_vector= c(magnitude_vector, magnitude)
      }
    }
  }
  # Get the accuracy on validation set for the given lambda
  valeval <- accuracy (validate_X, validate_Y, model_coef, intercept)
  validation_accuracies <- c(validation_accuracies, valeval)
  testeval <- accuracy(test_X, test_Y, model_coef, intercept)
  test_accuracies <- c(test_accuracies, testeval)
  title <- paste("Lambda = ", toString(lambda))
  jpeg(file=paste('/home/khanna/codebase/ml-practice/homework2/lambda_', toString(lambda), "_error.jpg"))
  plot(1:500 , season_accuracy, col="blue", type="l",xlab ="Epoch", ylab ="Held Out Error",xlim = c(0,500), ylim = c(0,1), main = title)
  dev.off()
  jpeg(file=paste('/home/khanna/codebase/ml-practice/homework2/lambda_', toString(lambda), "_magnitude.jpg"))
  plot(1:500 , magnitude_vector, col="green", type="l",xlab ="Epoch", ylab ="Size of W", xlim = c(0,500), ylim = c(0,2),main = title)
  dev.off()
}
# Find the index having maximum accuracy
index= which(validation_accuracies==max(validation_accuracies))
# Show the lambda for which accuracy was max
lambda_vector[index]
# Show the accuracy for the lambda that gave max accuracy in validation set
test_accuracies[index]