data <- read.csv('/Users/mohitkhanna/Desktop/ml-practice/pima-indians-diabetes.data', header = FALSE)

library(klaR)
library(caret)
library(e1071)

feature_matrix <- data[, -c(9)] # matrix of features
label_matrix <- data[, c(9)]     # labels; class value 1 means "tested positive for diabetes"


do_bayes_classification <- function(feature_matrix, label_matrix)
{
  set.seed(43210)
  trainingscore <- array(dim = 10) # create an array of 1 row and 10 columns and we will store training score here
  testscore <- array(dim = 10) # create an array of 1 row and 10 columns and we will store test score here
  for (wi in 1:10) {
    split <- createDataPartition(y = label_matrix, p = 0.8, list = FALSE) # 80% of the data into training 
    feature_split <- feature_matrix                                 # matrix of features
    training_feature_matrix <- feature_split[split, ]                         # training features
    training_label_matrix <- label_matrix[split]                          # training labels
  
    positive_labels <- training_label_matrix > 0                      # training labels for diabetes positive
    positive_training_features <- training_feature_matrix[positive_labels, ]                # training rows features with diabetes positive
    negative_training_features <- training_feature_matrix[!positive_labels, ]               # training rows features with diabetes negative
  
 
    test_feature_matrix <- feature_split[-split, ]                        # test rows - features
    test_label_matrix <- label_matrix[-split]                         # test rows - labels
  
    positive_labels_test <- test_label_matrix>0
    positive_test_features <- test_feature_matrix[positive_labels_test,]  # test row with with positive class
    negative_test_features <- test_feature_matrix[!positive_labels_test,] # test row with with negative class
    
    positive_test_prior <- nrow(positive_training_features)/(nrow(positive_training_features) +nrow(negative_training_features))  # positive prior
    negative_test_prior <- nrow(negative_training_features)/(nrow(positive_training_features) +nrow(negative_training_features))  # negative prior
    
    
    positive_training_mean <- sapply(positive_training_features, mean, na.rm = T)  # vector of means for training, diabetes positive
    negative_training_mean <- sapply(negative_training_features, mean, na.rm = T)  # vector of means for training, diabetes negative
    positive_training_sd   <- sapply(positive_training_features, sd, na.rm = T)    # vector of sd for training, diabetes positive
    negative_training_sd   <- sapply(negative_training_features, sd, na.rm = T)    # vector of sd for training, diabetes negative
  
    positive_training_offset <- t(t(training_feature_matrix) - positive_training_mean)         # normalisaing by, subtract mean
    positive_training_scales  <- t(t(positive_training_offset) / positive_training_sd)      # normalisaing by , divide by sd
  
    # Posterier prob for training positive class
    positive_training_logs <- -(1/2) * rowSums(apply(positive_training_scales, c(1,2),function(x) x^2), na.rm = T) - sum(log(positive_training_sd))
  
  
    negative_training_offset <- t(t(training_feature_matrix) - negative_training_mean)
    negative_training_scales  <- t(t(negative_training_offset) / negative_training_sd)
    # Posterier prob for training negative class
    negative_training_logs    <- -(1/2) * rowSums(apply(negative_training_scales, c(1,2) , function(x) x^2), na.rm = T) - sum(log(negative_training_sd))
  
  
    predicted_label_matrix_training  <- positive_training_logs > negative_training_logs              # Rows classified as diabetes positive by classifier 
    gotrighttr <- predicted_label_matrix_training == training_label_matrix                 # compare with true labels
    trainingscore[wi]<- sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # Accuracy with training set
  
    positive_test_offset <- t(t(test_feature_matrix)-positive_training_mean)            # Normalize test dataset with parameters from training
    positive_test_scales  <- t(t(positive_test_offset)/positive_training_sd)
    positive_test_logs    <- -(1/2)*rowSums(apply(positive_test_scales,c(1, 2)
                                                , function(x)x^2), na.rm=TRUE) -sum(log(positive_training_sd)) + log2(positive_test_prior)
  
    negative_test_offset <- t(t(test_feature_matrix)-negative_training_mean)            # Normalize test for diabetes negative class
    negative_test_scales  <- t(t(negative_test_offset)/negative_training_sd)
    negative_test_logs    <- -(1/2)*rowSums(apply(negative_test_scales,c(1, 2)
                                                , function(x)x^2), na.rm=TRUE)-sum(log(negative_training_sd)) + log2(negative_test_prior)
  
    predicted_label_matrix_test<-positive_test_logs>negative_test_logs
    gotright<-predicted_label_matrix_test==test_label_matrix
    testscore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))  # Accuracy on the test set
  }
  return (testscore)
}
#Part A

testscore_without_na=do_bayes_classification(feature_matrix, label_matrix)


##############################################################################################################################################
#Part B
copy_feature_matrix <- feature_matrix

for (i in c(3, 4, 6, 8))
{check_zero_feature<-feature_matrix[, i]==0
copy_feature_matrix[check_zero_feature, i]=NA
}
testscore_with_na=do_bayes_classification(copy_feature_matrix, label_matrix)
(testscore_without_na)
(testscore_with_na)



###############################################################################################################################################
#Part C

check_for_na <- apply(data, 1, function(x){any(is.na(x))})  # to ensure if there is any na
sum(check_for_na)==0



partision<-createDataPartition(y=label_matrix, p=.8, list=FALSE)
train_x<-feature_matrix[partision,]
train_y<-label_matrix[partision]
model<-train(train_x, as.factor(train_y), 'nb', trControl=trainControl(method='cv', number=10))
predicted_val<-predict(model,newdata=feature_matrix[-partision,])
confusionMatrix(data=predicted_val, label_matrix[-partision])

###############################################################################################################################################
#Part D

svm<-svmlight(feature_matrix[partision,], label_matrix[partision], pathsvm="/Users/mohitkhanna/Desktop/ml-practice/svm_light_osx.8.4_i7/")
labels<-predict(svm, feature_matrix[-partision,])
predicted_svm<-labels$class
sum(predicted_svm==label_matrix[-partision])/(sum(predicted_svm==label_matrix[-partision])+sum(!(predicted_svm==label_matrix[-partision])))
