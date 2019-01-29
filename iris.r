rm(list=ls())
require(rpart)
require(rpart.plot)
require(neuralnet)
require(e1071)

#Loading dataset
data("iris")
df = iris

#Preprocessing - scaling data
maxs = apply(df[1:4], MARGIN = 2, max)
mins = apply(df[1:4], MARGIN = 2, min)
scaled = as.data.frame(scale(df[1:4], center = mins, scale =	maxs - mins))
data = scaled
data['Species'] = df['Species']

for(i in 1:5){
  
  #Split data for tree, svm, naive bayes
  trainIndex <- sample(1:nrow(data), 0.8 * nrow(data))
  train <- data[trainIndex,]
  test <- data[-trainIndex,]

  #Split data for perceptron and neural net

  nn_train <- train
  nn_train <- cbind(nn_train, train$Species == 'setosa')
  nn_train <- cbind(nn_train, train$Species == 'versicolor')
  nn_train <- cbind(nn_train, train$Species == 'virginica')

  names(nn_train)[6] <- 'setosa'
  names(nn_train)[7] <- 'versicolor'
  names(nn_train)[8] <- 'virginica'

  #tree
  tree <- rpart(Species ~ ., data = train, method = "class", parms = list(split = "information"))
  t_pred = predict(tree, test, type="class")
  cm_tree = table(t_pred, test$Species)
  t_accuracy = (sum(diag(cm_tree))/sum(cm_tree))*100

  #perceptron and neural net
  perceptron <- neuralnet(setosa+versicolor+virginica ~ Sepal.Length+Sepal.Width +Petal.Length +Petal.Width,data=nn_train, hidden=0, threshold = 0.02, rep = 50, learningrate = 0.1)
  perceptron_predict <- compute(perceptron, test[1:4])$net.result

  nn <- neuralnet(setosa+versicolor+virginica ~ Sepal.Length+Sepal.Width +Petal.Length +Petal.Width,data=nn_train, hidden=c(3,2), threshold = 0.1, rep = 100, learningrate = 0.1)
  nn_predict <- compute(nn, test[1:4])$net.result

  getMax <- function(mylist) {
    return(which(mylist == max(mylist)))
  }

  percep_id <- apply(perceptron_predict, c(1), getMax)
  perceptron_pred <- c('setosa', 'versicolor', 'virginica')[percep_id]

  nn_id <- apply(nn_predict, c(1), getMax)
  nn_pred <- c('setosa', 'versicolor', 'virginica')[nn_id]

  cm_perceptron = table(perceptron_pred, test$Species)

  cm_nn = table(nn_pred, test$Species)

  per_accuracy = (sum(diag(cm_perceptron))/sum(cm_perceptron))*100
  nn_accuracy = (sum(diag(cm_nn))/sum(cm_nn))*100

  #SVM
  svm_m = svm(train[,1:4], train[,5], cost = 100, gamma = 1, kernel = 'linear')
  svm_pred <- predict(svm_m,subset(test, select=-Species))
  y_t = test$Species
  svm_cm = table(svm_pred,y_t)
  svm_accuracy = (sum(diag(svm_cm))/sum(svm_cm))*100

  #Naive Bayes
  nbc<-naiveBayes(train[,1:4], train[,5], laplace = 3, threshold = 0.1)
  nb_pred = predict(nbc, test[,-5])
  nb_cm = table(nb_pred, test[,5])
  nb_accuracy = (sum(diag(nb_cm))/sum(nb_cm))*100

  cat('\n\nIteration ', i, '\n' ,'\nID3: ', t_accuracy, '\nPerceptron: ', per_accuracy, '\nNeural Net: ', nn_accuracy, '\nSVM: ', svm_accuracy, '\nNaive Bayes: ', nb_accuracy)
}
