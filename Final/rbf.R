library(data.table)
library(dplyr)
library(ggplot2)
library(e1071)

# target function
target <- function(data){
  return(cbind(data, y = (sign(data[, x2] - data[, x1] + 0.25*sin(pi*data[, x1])))))
}

# training data
train <- function(N){
  data <- data.table(x1=runif(N, -1, 1), x2=runif(N, -1, 1))
  return(target(data))
}

# Q13

Ein <- c()

for (i in 1:10000){
  trainingData <- train(100)
  model <- svm(y~x1+x2, data = as.matrix(trainingData),
               type = 'C-classification',
               scale = FALSE, shrinking = FALSE,
               kernel = "radial", gamma = 1.5, 
               cost = 10e6)
  
  
  Ein <- c(Ein, sum(trainingData[, y]!=predict(model, as.matrix(trainingData[, .(x1, x2)]))))
}

sum(Ein==0)

# Q14

addCenter <- function(data, K){
  clust <- kmeans(data[,.(x1,x2)], centers = k,iter.max = 100,nstart = 5,algorithm = "Lloyd")
  return(cbind(data, k=clust$cluster))
}






