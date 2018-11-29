#Machine learning

rm(list = ls()); # clear workspace variables
cat("\014") # it means ctrl+L. clear window
graphics.off() # close all plots

#install.packages("caret")
#install.packages("caret", dependencies=c("Depends", "Suggests"))
#install.packages("lattice")
#install.packages("dplyr")
#install.packages("readr")

library(ggplot2)
library(lattice)
library(dplyr)
library(caret)  #Main machine learning package
library(readr)
creditc <- read_csv("C:/Users/nadir/Downloads/creditc_default.csv")

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(creditc$default, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- creditc[-validation_index,]
# use the remaining 80% of data to training and testing the models
creditc <- creditc[validation_index,]

# Looking at dimensions of dataset
dim(creditc)

# list types for each attribute
sapply(creditc, class)

# summarize the class distribution
percentage <- prop.table(table(creditc$default)) * 100
cbind(freq=table(creditc$default), percentage=percentage)
# We can see that defaults happen 22.11% of the time

# summarize attribute distributions
summary(creditc)


#5. Evaluate Some Algorithms
# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
#LDA
set.seed(7)
fit.lda <- train(default~., data=creditc, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# kNN
set.seed(7)
fit.knn <- train(default~., data=creditc, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(default~., data=creditc, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(default~., data=creditc, method="rf", metric=metric, trControl=control)


# summarize accuracy of models
results <- resamples(list(lda=fit.lda, knn=fit.knn, svm= fit.svm, rf=fit.rf))

# compare accuracy of models
summary(results)
dotplot(results)


# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$default)

# estimate skill of KNN on the validation dataset
predictions <- predict(fit.knn, validation)
confusionMatrix(predictions, validation$default)

# estimate skill of SVM on the validation dataset
predictions <- predict(fit.svm, validation)
confusionMatrix(predictions, validation$default)

# estimate skill of RF on the validation dataset
predictions <- predict(fit.rf, validation)
confusionMatrix(predictions, validation$default)

