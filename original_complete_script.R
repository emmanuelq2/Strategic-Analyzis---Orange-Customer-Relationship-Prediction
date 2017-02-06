
getwd()
# [1] "C:/Users/emmanuel/Documents"
setwd("./R/data")
churn <- read.csv('orange_small_train_churn.labels', header = FALSE)
appetency <- read.csv('orange_small_train_appetency.labels', header = FALSE)
upsell <- read.csv('orange_small_train_upselling.labels', header = FALSE)
str(churn)
# 'data.frame':   50000 obs. of  1 variable:
# V1: int  -1 1 -1 -1 -1 -1 -1 -1 -1 -1 ...
str(appetency)
# data.frame':   50000 obs. of  1 variable:
# V1: int  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ...
str(upsell)
# data.frame':   50000 obs. of  1 variable:
# $ V1: int  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ...

churn[churn$V1 < 0,] <- 0
appetency[appetency$V1 < 0,] <- 0
upsell[upsell$V1 < 0,] <- 0
colnames(churn) <- c('churn')
colnames(appetency) <- c('appetency')
colnames(upsell) <- c('upsell')

mean(churn$churn)
# 0.07344
mean(upsell$upsell)
# 0.07364
mean(appetency$appetency)
# 0.0178

# No need for the following operation - keep it simple! 
# collabels <- cbind(churn, appetency, upsell)
# str(collabels)
# 'data.frame':   50000 obs. of  3 variables:
# $ churn    : num  0 1 0 0 0 0 0 0 0 0 ...
# $ appetency: num  0 0 0 0 0 0 0 0 0 0 ...
# $ upsell   : num  0 0 0 0 0 0 0 0 0 0 ...

library(timeSeries)
library(timeDate)
library(fBasics)
library(xtable)
library(dplyr)


collabels <- dplyr::select(data.frame(t(basicStats(collabels))), 1:9)
colnames(collabels)[c(5,6,9)] <- c('Q1', 'Q2','Positive_Instances')
print(xtable(data.frame(collabels), caption = 'Response Variables'))

summary(collabels)

df <- read.table('orange_small_train.data', header = TRUE, sep = '\t', na.strings = '')
str(df)
# summary(df)

# some of the 230 variables are 100% missing, they are the only logical class vars
# so we can safely remove all logical class vars
            
for (i in names(df)){
  vclass <- class(df[,i])
  if(vclass == 'logical'){
    df[,i] <- NULL
  }else if(vclass %in% c('integer', 'numeric')){
# impute mising data with zeros and "missing"
# first check that there are missing variables
    if(sum(is.na(df[,i])) == 0) next
# create a missing variable column
      df[,paste(i,'_missing',sep='')] <- as.integer(is.na(df[,i]))
# fill missing variables with 0
      df[is.na(df[,i]),i] <- 0
    }else{
# gather infrequent levels into 'other'
      levels(df[,i])[xtabs(~df[,i])/dim(df)[1] < 0.015] <- 'other'
# replace NA with 'missing'
      levels(df[,i]) <- append(levels(df[,i]), 'missing')
      df[is.na(df[,i]), i] <- 'missing'
    }
  }

df$churn <- churn$churn
df$appetency <- appetency$appetency
df$upsell <- upsell$upsell

# str(df)
# summary(df)


table(churn$churn)
# 0     1 
# 46328  3672 

truedf <- df[,0:230]

truedf$appetency <- appetency$appetency
truedf$upsell <- upsell$upsell
truedf$churn <- churn$churn
# str(truedf)
# summary(truedf) -> 18 missing columns (8,15,31,32,39,42,48,55,141,167,169,175,185,209...)
#
data = (truedf[,c(1:212,231:233)])

# convert all factor type columns into numeric ones              
for (i in names(data)){
  vclass <- class(data[,i])
  if(vclass == 'factor'){
# transform all factor columns into numeric ones
    data[,i] <- as.numeric(data[,i])
    }
  }


# library(caTools)
# set.seed(123)
# splchurn = sample.split(data$churn, SplitRatio = 0.1)
# SampleTrainChurn = subset(data, splchurn==TRUE)
# logregchurn = glm(churn ~ . -appetency -upsell, family=binomial(logit), data=SampleTrainChurn)
# Warning messages:
# 1: glm.fit: algorithm did not converge 
# 2: glm.fit: fitted probabilities numerically 0 or 1 occurred


# Hierarchical clustering
set.seed(123)
splchurn = sample.split(data$churn, SplitRatio = 0.1)
SampleTrainCluster = subset(data, splchurn==TRUE)
testdata = scale(SampleTrainCluster)
d = dist(testdata, method = "euclidean")
hcward = hclust(d, method="ward.D")
SampleTrainCluster$groups=cutree(hcward,k=5)
# Aggregation by group and computation of the mean values
aggdata = aggregate(.~ groups, data=SampleTrainCluster, FUN=mean) 
# Computation of the number of observations by group
proptempapp = aggregate(appetency ~ groups, data=SampleTrainCluster, FUN=length) 
proptempchurn = aggregate(churn ~ groups, data=SampleTrainCluster, FUN=length)
proptempupsell = aggregate(upsell ~ groups, data=SampleTrainCluster, FUN=length)
# Computation of the proportion by group
aggdata$propappetency = (proptempapp$appetency)/sum(proptempapp$appetency) 
aggdata$propchurn = (proptempchurn$churn)/sum(proptempchurn$churn)
aggdata$propupsell = (proptempupsell$upsell)/sum(proptempupsell$upsell)


orange <- aggdata[,214:217]
orange$frequencies <- orange$propappetency
orange$propappetency <- NULL
write.csv(orange, file = "orange.csv", sep =",", row.names=FALSE)

# Même clustering selon les trois critères
write.csv(proptempupsell, file = "proptempupsell.csv", sep =",", row.names=FALSE)
# aggdata=aggdata[order(aggdata$proportion,decreasing=T),] # Ordering from the largest group to the smallest



# # ---- train_test_mat ---- back to original data set
# get the index for training/testing data - train: 75% - test: 25%
set.seed(123)
smp_size <- floor(0.75 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
# making a "tiny" data set so I cn quickly test r markdown and graphical paramters
# this will be removed in the submitted version
tiny_ind <- sample(seq_len(nrow(data)), size = floor(0.01 * nrow(data)))
# split the data
train <- data[train_ind, ]
test <- data[-train_ind, ]
tiny <- data[tiny_ind, ]



# Preparing data with regularization and scaling for glm and KNN
df_mat <- select(data, -churn, -appetency, -upsell)

for (i in names(df_mat)){
  if (class(df_mat[,i]) == 'factor'){
    for(level in unique(df_mat[,i])){
      df_mat[sprintf('%s_dummy_%s', i, level)] <- ifelse(df_mat[,i] == level, 1, 0)
    }
    df_mat[,i] <- NULL
  } else {
    # scale numeric variables
    # this is important for regularized logistic regression and KNN
    df_mat[,i] <- scale(df_mat[,i])
  }
}

df_mat <- data.matrix(df_mat)



## Logistic Regression with Elastic-Net Penalty

# ---- lreg_churn ----
library(glmnet)
# regularized logistic regression with cross validation
# this takes a while, try using nfolds < 10 to reduce time
churn_lreg.cv <- cv.glmnet(df_mat[train_ind,],
                           factor(train$churn), family = "binomial",
                           nfolds = 10, type.measure = 'auc')
# ----

# view the Area Under the Curve for different values of lambda.
plot(churn_lreg.cv)
title('Cross Validation Curve Logistic Regression',line =+2.8)


library(data.table)
library(knitr)
cv_coefs <- data.table(variable = row.names(coef(churn_lreg.cv))[
  abs(as.vector(coef(churn_lreg.cv))) > 1e-5],
  coeficient = coef(churn_lreg.cv)[abs(coef(churn_lreg.cv)) > 1e-5])


kable(cv_coefs[variable %like% '26'],
      caption = "Variables Selected by Elastic-Net")


# |variable | coeficient|
# |:--------|----------:|
# |Var126   |   0.156603|

# KNN - Fast Nearest Neighbour 
# library(FNN)
#
# auc_vec <- rep(0, 20)
#
library(ROCR)
library(e1071)

# for(i in 1:20){
#   print(sprintf('trying k = %d', i))
#   yhat <- knn(df_mat[train_ind,], df_mat[-train_ind,],
#               cl = factor(train$churn), k = i, prob = TRUE)
#   pred <- predict((as.numeric(yhat[1:dim(df_mat[-train_ind,])[1]]) - 1) * attr(yhat,'prob'),
#                     factor(test$churn))
#   # the following commented out code is for use with the tiny data set
#   yhat <- knn(df_mat[tiny_ind,], df_mat[tiny_ind,],
#             cl = factor(tiny$churn), k = i, prob = TRUE)
#   pred <- predict((as.numeric(yhat[1:dim(df_mat[tiny_ind,])[1]]) - 1) * attr(yhat,'prob'),
#                    factor(tiny$churn))
#  perf <- performance(pred, measure = "tpr", x.measure = "fpr")
#  print(sprintf('AUC: %f',
#                attributes(performance(pred, 'auc'))$y.values[[1]]))
#   auc_vec[i] <- attributes(performance(pred, 'auc'))$y.values[[1]]
# }

# Error in UseMethod("predict") : 
# no applicable method for 'predict' applied to an object of class "c('double', 'numeric')"

# p <- qplot(y = auc_vec, color = 'AUC') + geom_line() +   xlab('k = x') + ylab('AUC') + ggtitle('K-NN')
# p


## Decision Tree

# ---- dt_churn ----
library(rpart)
library(rpart.plot)

churn_tree <- rpart(factor(churn)~.,
                    data = select(train, -appetency, -upsell),
                    method = 'class',
                    control=rpart.control(minsplit=40, minbucket=10, cp=0.001))
# ----

# ---- plot_tree_churn ----
rpart.plot(churn_tree, main = 'Churn Decision Tree')
# ----


## Random Forest

# ---- rf_churn ----
library(randomForest)
set.seed(123)
churn_rf <- randomForest(factor(churn)~.,
                         data = select(train, -appetency, -upsell),
                         ntree = 10, nodesize = 10, importance = TRUE)
# ----
# ---- plot_rf_churn ----
varImpPlot(churn_rf, type = 2, main = 'Variable Importance Churn')
# ----

#
# ---- roc_churn ----
yhat <- predict(churn_rf, select(test, -appetency, -upsell), type = 'prob')

pred <- prediction(yhat[,2], factor(test$churn))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
library(ggplot2)
p <- ggplot(data.frame(TPR = unlist(perf@y.values),
                       FPR = unlist(perf@x.values)),
            aes(FPR, TPR)) + geom_line(color = '#3b5b92', size = 1) +
  xlab('False Positvie Rate') + ylab('True Positive Rate') +
  ggtitle('Random Forest ROC Curve') +
  theme(plot.title = element_text(lineheight=.8, face="bold")) +
  annotate("text", x = 0.75, y = 0.20, label = sprintf('AUC: %f',
  attributes(performance(pred, 'auc'))$y.values[[1]]))
p
# ----


# ## Principal Components
# pca <- princomp(df_mat)
#
#
# library(ggbiplot)
#
# p <- ggbiplot(pca, groups = factor(df$churn), ellipse = FALSE,
#               var.axes = FALSE) +
#   ggtitle('First 2 Principal Components') +
#   xlim(-3, 3) + ylim(-3, 3) +
#   scale_fill_discrete(name = 'Churn')
#
# p




# Appetency


## Logistic Regression with Elastic-Net Penalty
# ---- lreg_app ----
app_lreg.cv <- cv.glmnet(df_mat[train_ind,], factor(train$appetency), family = "binomial",
                         nfolds = 8, type.measure = 'auc')
# ----

# view the bionmial deviance (log loss) of differnt values of lambda
plot(app_lreg.cv)
title('Cross Validation Curve Logistic Regression', line =+2.8)


cv_coefs <- data.frame(
  coeficient = coef(app_lreg.cv, s = 'lambda.1se')[abs(coef(app_lreg.cv,
                                                            s = 'lambda.1se')) > 1e-3])

row.names(cv_coefs) <- row.names(coef(app_lreg.cv,
          s = 'lambda.1se'))[abs(as.vector(coef(app_lreg.cv, s = 'lambda.1se'))) > 1e-3]

kable(cv_coefs, caption = "Variables Selected by Elastic-Net")

#|            | coeficient|
#  |:-----------|----------:|
#  |(Intercept) | -4.4194465|
#  |Var81       |  0.0073112|
#  |Var126      | -0.7952068|
#  |Var140      |  0.0107185|
#  |Var217      | -0.0395200|
#  |Var218      |  0.5296083|


yhat <- predict(app_lreg.cv, df_mat[-train_ind,], type = 'response')

# Error in as.matrix(cbind2(1, newx) %*% nbeta) : 
#  error in evaluating the argument 'x' in selecting a method for function 'as.matrix': Error in cbind2(1, newx) %*% nbeta : 
#  Cholmod error 'out of memory' at file ../Core/cholmod_memory.c, line 147

pred <- prediction(yhat, factor(test$appetency))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")

p <- ggplot(data.frame(TPR = unlist(perf@y.values),
                       FPR = unlist(perf@x.values)),
            aes(FPR, TPR)) + geom_line(color = '#3b5b92', size = 1) +
  xlab('False Positvie Rate') + ylab('True Positive Rate') +
  ggtitle('Logistic Regression ROC Curve') +
  theme(plot.title = element_text(lineheight=.8, face="bold")) +
  annotate("text", x = 0.75, y = 0.20, label = sprintf('AUC: %f',
  attributes(performance(pred, 'auc'))$y.values[[1]]))
p



## Decision Tree

# ---- dt_app ----
app_tree <- rpart(factor(appetency)~.,
                  data = select(train, -churn, -upsell),
                  method = 'class',
                  control=rpart.control(minsplit=40, minbucket=10, cp=0.001))
# ----
rpart.plot(app_tree, main = 'Appetency Decision Tree')
app_tree



## Random Forest

# ---- rf_app ----
app_rf <- randomForest(factor(appetency)~.,
                       data = select(train, -churn, -upsell),
                       ntree = 10, nodesize = 10, importance = TRUE)
# ----
varImpPlot(app_rf, type = 2, main = 'Variable Importance Appetency')


yhat <- predict(app_rf, select(test, -churn, -upsell), type = 'prob')

pred <- prediction(yhat[,2], factor(test$appetency))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
p <- ggplot(data.frame(TPR = unlist(perf@y.values),
                       FPR = unlist(perf@x.values)),
            aes(FPR, TPR)) + geom_line(color = '#3b5b92', size = 1) +
  xlab('False Positvie Rate') + ylab('True Positive Rate') +
  ggtitle(' Random Forest ROC Curve') +
  theme(plot.title = element_text(lineheight=.8, face="bold")) +
  annotate("text", x = 0.75, y = 0.20, label = sprintf('AUC: %f',
  attributes(performance(pred, 'auc'))$y.values[[1]]))
p



# Up-Sell


## Logistic Regression with Elastic-Net Penalty
# ---- lreg_upsell ----
upsell_lreg.cv <- cv.glmnet(df_mat[train_ind,], factor(train$upsell), family = "binomial",
                            nfolds = 8, type.measure = 'auc')
# ----
# view the bionmial deviance (log loss) of differnt values of lambda
plot(upsell_lreg.cv)
title('Cross Validation Curve Logistic Regression',line =+2.8)




cv_coefs <- data.frame( coeficient = coef(upsell_lreg.cv, s = 'lambda.1se')[
  abs(coef(upsell_lreg.cv, s = 'lambda.1se')) > 1e-3])

row.names(cv_coefs) <- row.names(coef(upsell_lreg.cv, s = 'lambda.1se'))[
  abs(as.vector(coef(upsell_lreg.cv, s = 'lambda.1se'))) > 1e-3]
kable(cv_coefs, caption = "Variables Selected by Elastic-Net")

#| coeficient|
#  |:-----------|----------:|
#  |(Intercept) | -3.2269956|
#  |Var3        |  0.0216974|
#  |Var7        |  0.0582234|
#  |Var12       |  0.0020522|
#  |Var16       | -0.0553140|
#  |Var24       |  0.0185167|
#  |Var25       |  0.0483422|
#  |Var28       | -0.4540365|
#  |Var30       | -0.0270243|
#  |Var37       | -0.0073661|
#  |Var38       | -0.0799400|
#  |Var51       | -0.0085940|
#  |Var69       | -0.0116969|
#  |Var73       |  0.0704382|
#  |Var78       | -0.0318526|
#  |Var81       | -0.0415958|
#  |Var82       | -0.0138749|
#  |Var83       | -0.0209971|
#  |Var85       |  0.0223490|
#  |Var93       | -0.0827863|
#  |Var94       |  0.0218111|
#  |Var111      | -0.0133717|
#  |Var113      |  0.1669209|
#  |Var121      |  0.0023023|
#  |Var123      | -0.0392339|
#  |Var125      |  0.0180325|
#  |Var126      | -0.9621325|
#  |Var132      | -0.0182298|
#  |Var134      | -0.0331331|
#  |Var135      | -0.1318753|
#  |Var139      | -0.0051017|
#  |Var144      | -0.1194631|
#  |Var153      | -0.0642778|
#  |Var180      | -0.0025787|
#  |Var182      | -0.0275585|
#  |Var188      | -0.0790933|
#  |Var192      |  0.0327368|
#  |Var199      |  0.0058800|
#  |Var204      | -0.0415323|
# |Var208      | -0.0106547|
# |Var210      |  0.2872484|
#  |Var211      | -1.3567484|
#  |Var215      |  0.0153961|
#  |Var216      |  0.0624003|
#  |Var217      |  0.0965256|
#  |Var218      | -0.1443144|
#  |Var219      | -0.0217717|
#  |Var225      | -0.1544053|
#  |Var227      | -0.0187451|
#  |Var228      |  0.0017325|
# |Var229      | -0.0194458|



## Decision Tree
# ---- dt_upsell ----
upsell_tree <- rpart(factor(upsell)~.,
                     data = select(train, -appetency, -churn),
                     method = 'class',
                     control=rpart.control(minsplit=100, minbucket=10, cp=0.001))
# ----
rpart.plot(upsell_tree, main = 'Up-Sell Decision Tree')




summary(upsell_tree)


## Random Forest
# ---- rf_upsell ----
upsell_rf <- randomForest(factor(upsell)~.,
                          data = select(train, -appetency, -churn),
                          ntree = 10, nodesize = 10, importance = TRUE)
# ----
varImpPlot(upsell_rf, type = 2, main = 'Variable Importance Up-Sell')

save(list = c('churn_lreg.cv', 'churn_rf', 'churn_tree',
              'app_lreg.cv', 'app_rf', 'app_tree',
              'upsell_lreg.cv', 'upsell_rf', 'upsell_tree'),
     file = "C:/Users/emmanuel/Documents/R/data")
