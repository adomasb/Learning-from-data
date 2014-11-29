library(data.table)
library(dplyr)
library(glmnet)
library(e1071)


train <- read.table("/home/adomas/Learning/Learning-from-data/Final/train.txt") %>%
  data.table() %>%
  setnames(c("digit", "x1", "x2"))
test <- read.table("/home/adomas/Learning/Learning-from-data/Final/test.txt") %>%
  data.table() %>%
  setnames(c("digit", "x1", "x2"))


# Q7
Ein <- c()
for (i in 5:9){
  training <- train
  
  training <- mutate(training, class = ifelse(digit == i, 1, -1)) %>%
    select(class, x1, x2) %>%
    as.matrix()
  
  model <- glmnet(x = training[, 2:3], y = training[, 1],family = "gaussian",
                  alpha = 1, lambda=1/nrow(training),
                  standardize = FALSE)
  
  Ein <- c(Ein, 1-mean(training[, 1]==sign(predict(model, training[, 2:3]))))
}

# Q8

trainTrans <- mutate(train, x1x2 = x1*x2, x12 = x1^2, x22 = x2^2)
testTrans <- mutate(test, x1x2 = x1*x2, x12 = x1^2, x22 = x2^2)

Eout <- c()
for (i in 0:4){
  training <- trainTrans
  testing <- testTrans
  
  training <- mutate(training, class = ifelse(digit == i, 1, -1)) %>%
    select(class, x1, x2, x1x2, x12, x22) %>%
    as.matrix()
  
  testing <- mutate(testing, class = ifelse(digit == i, 1, -1)) %>%
    select(class, x1, x2, x1x2, x12, x22) %>%
    as.matrix()
  
  model <- glmnet(x = training[, 2:6], y = training[, 1] ,family = "gaussian",
                  alpha = 1, lambda = 1/nrow(training), intercept=TRUE,
                  standardize = FALSE)
  
  Eout <- c(Eout, 1-(mean(testing[, 1]==sign(predict(model, testing[, 2:6])))))
}

# Q9

Ein <- c()
Eout <- c()
Ein_t <- c()
Eout_t <- c()

trainTrans <- mutate(train, x1x2 = x1*x2, x12 = x1^2, x22 = x2^2)
testTrans <- mutate(test, x1x2 = x1*x2, x12 = x1^2, x22 = x2^2)

for (i in 0:9){
  training <- train
  testing <- test
  
  training <- mutate(training, class = ifelse(digit == i, 1, -1)) %>%
    select(class, x1, x2) %>%
    as.matrix()
  
  testing <- mutate(testing, class = ifelse(digit == i, 1, -1)) %>%
    select(class, x1, x2) %>%
    as.matrix()
  
  trainingTrans <- trainTrans
  testingTrans <- testTrans
  
  trainingTrans <- mutate(trainingTrans, class = ifelse(digit == i, 1, -1)) %>%
    select(class, x1, x2, x1x2, x12, x22) %>%
    as.matrix()
  
  testingTrans <- mutate(testingTrans, class = ifelse(digit == i, 1, -1)) %>%
    select(class, x1, x2, x1x2, x12, x22) %>%
    as.matrix()
  

  model <- glmnet(x = training[, 2:3], y = training[, 1],family = "gaussian",
                  alpha = 1, lambda=1/nrow(training),
                  standardize = FALSE)
  
  modelTrans <- glmnet(x = trainingTrans[, 2:6], y = trainingTrans[, 1], 
                       family = "gaussian",
                       alpha = 1, lambda = 1/nrow(trainingTrans), intercept=TRUE,
                       standardize = FALSE)
  
  Ein <- c(Ein, 1-mean(training[, 1] == sign(predict(model, training[, 2:3]))))
  Eout <- c(Eout, 1-mean(testing[, 1] == sign(predict(model, testing[, 2:3]))))
  
  Ein_t <- c(Ein_t, 1-mean(trainingTrans[, 1] == sign(predict(modelTrans, trainingTrans[, 2:6]))))
  Eout_t <- c(Eout_t, 1-mean(testingTrans[, 1] == sign(predict(modelTrans, testingTrans[, 2:6]))))
}

# Q10
train15 <- mutate(train, x1x2 = x1*x2, x12 = x1^2, x22 = x2^2) %>%
  filter(digit %in% c(1,5)) %>%
  mutate(class = ifelse(digit == 1, 1, -1)) %>%
  select(class, x1, x2, x1x2, x12, x22) %>%
  as.matrix()

test15 <- mutate(test, x1x2 = x1*x2, x12 = x1^2, x22 = x2^2) %>%
  filter(digit %in% c(1,5)) %>%
  mutate(class = ifelse(digit == 1, 1, -1)) %>%
  select(class, x1, x2, x1x2, x12, x22) %>%
  as.matrix()

model1 <- glmnet(x = train15[, 2:6], y = train15[, 1], 
                 family = "gaussian", alpha = 1, lambda = 1, 
                 intercept=TRUE, standardize = FALSE)

model2 <- glmnet(x = train15[, 2:6], y = train15[, 1], 
                 family = "gaussian", alpha = 1, lambda = 0.01, 
                 intercept=TRUE, standardize = FALSE)

Ein1 <- 1 - mean(train15[, 1] == sign(predict(model1, train15[, 2:6])))
Eout1 <- 1 - mean(test15[, 1] == sign(predict(model1, test15[, 2:6])))
Ein2 <- 1 - mean(train15[, 1] == sign(predict(model2, train15[, 2:6])))
Eout2 <- 1 - mean(test15[, 1] == sign(predict(model2, test15[, 2:6])))

####################################

# Q11

data <- data.table(x1=c(1, 0, 0, -1, 0, 0, -2),
                   x2=c(0, 1, -1, 0, 2, -2, 0),
                   y=c(-1, -1, -1, 1, 1, 1, 1))


dataT <- mutate(data, z1 = x2^2-2*x1-1, z2 = x1^2-2*x2+1) %>%
  select(y, z1, z2)

ggplot(dataT, aes(z1, z2, color=factor(y)))+geom_point(size=3)

# Q12

model <- svm(x = as.matrix(data[, .(x1, x2)]), y = as.matrix(data[, .(y)]),
             type = 'C-classification',
             scale = FALSE,shrinking = FALSE,
             kernel = 'polynomial', degree = 2, gamma = 1, coef0 = 1, cost = 10e10)

sum(model$nSV)
