library(e1071)
library(data.table)
library(dplyr)
library(Hmisc)

# data
train <- data.table(read.table("/home/adomas/Learning/Learning-from-data/HW8/train.txt"))
test <- data.table(read.table("/home/adomas/Learning/Learning-from-data/HW8/test.txt"))
train <- data.table(read.table("/home/adomas/Learning-from-data/HW8/train.txt"))
test <- data.table(read.table("/home/adomas/Learning-from-data/HW8/test.txt"))
setnames(train, c("digit", "x1", "x2"))
setnames(test, c("digit", "x1", "x2"))


# parameters
C <- 0.01
Q <- 2
Ein <- c()
sv1 <- c()
sv2 <- c()
sv <- c()

for (y in 0:9){
  trainData <- mutate(train, y = ifelse(digit == y, 1, 0)) %>% 
    select(-digit) %>%
    as.matrix()
  
  model <- svm(y~x1+x2, data = trainData, type = 'C-classification',
               scale = FALSE,shrinking = FALSE,
               kernel = 'polynomial', degree = Q, gamma = 1, coef0 = 1, cost = C)
  
  Ein <- c(Ein, sum(predict(model, trainData) != trainData[, 3]))
  
  sv <- c(sv, sum(model$nSV))
}


# Q2
which.max(Ein[seq(1, 10, 2)])

# Q3
which.min(Ein[seq(2, 10, 2)])

# Q4
sv[1]-sv[2]

# Q5 

train15 <- filter(train, digit %in% c(1, 5)) %>%
  as.matrix()

test15 <- filter(test, digit %in% c(1, 5)) %>%
  as.matrix()

Q <- 2
C <- c(0.001, 0.01, 0.1, 1)
Eout <- c()
Ein <- c()
sv <- c()


for (c in C){
  model <- svm(digit~., data = train15, type = 'C-classification',
               scale = FALSE,shrinking = FALSE,
               kernel = 'polynomial', degree = Q,
               gamma = 1, coef0 = 1, cost = c)
  
  Ein <- c(Ein, sum(predict(model, train15) != train15[, 1]))
  Eout <- c(Eout, sum(predict(model, test15) != test15[, 1]))
  sv <- c(sv, sum(model$nSV))
}

# Q6

# a FALSE
model <- svm(digit~., data = train15, type = 'C-classification',
             scale = FALSE,shrinking = FALSE,
             kernel = 'polynomial', degree = 5,
             gamma = 1, coef0 = 1, cost = 0.0001)

sum(predict(model, train15) != train15[, 1])

# b TRUE
model <- svm(digit~., data = train15, type = 'C-classification',
             scale = FALSE,shrinking = FALSE,
             kernel = 'polynomial', degree = 5,
             gamma = 1, coef0 = 1, cost = 0.001)

model$nSV

# c False
model <- svm(digit~., data = train15, type = 'C-classification',
             scale = FALSE,shrinking = FALSE,
             kernel = 'polynomial', degree = 2,
             gamma = 1, coef0 = 1, cost = 0.01)

sum(predict(model, train15) != train15[, 1])

# d FALSE
model <- svm(digit~., data = train15, type = 'C-classification',
             scale = FALSE,shrinking = FALSE,
             kernel = 'polynomial', degree = 5,
             gamma = 1, coef0 = 1, cost = 1)

sum(predict(model, test15) != test15[, 1])

#######################################

train15 <- filter(train, digit %in% c(1, 5)) %>%
  as.matrix()

test15 <- filter(test, digit %in% c(1, 5)) %>%
  as.matrix()

Q <- 2
C <- c(0.0001, 0.001, 0.01, 0.1, 1)

splits <- data.table(from=c(1, seq(157, 1405, 156)), to=c(seq(156, 1404, 156), nrow(train15)))

for (times in 1:100){
  data <- filter(train, digit %in% c(1, 5)) %>%
    sample_frac(size = 1, replace = FALSE) %>% 
    as.matrix()
  
  for (i in 1:10){
    Eout <- c()
    for (c in C){
      test <- data[splits[i, from]:splits[i, to], ]
      train <- data[setdiff(1:1561, splits[i, from]:splits[i, to]), ]
      
      model <- svm(digit~., data = train, type = 'C-classification',
                   scale = FALSE,shrinking = FALSE,
                   kernel = 'polynomial', degree = 5,
                   gamma = 1, coef0 = 1, cost = c)
      
      Eout <- c(Eout, sum(predict(model, test) != test[,1]))
    }
    selected <- which.min(Eout)
  }
  
  
}



trainData <- filter(train, digit %in% c(1, 5)) %>%
  sample_frac(size = 1, replace = FALSE) %>% 
  mutate(fold = cut2(1:1561, cuts=splits)) %>%
  as.matrix()



