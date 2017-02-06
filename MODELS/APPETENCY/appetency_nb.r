
################################################################################
###       COMMENT FOR NAIVE MODEL FOR APPETENCY
#################################################################################
#The Naive Bayes technique was applied in a computational EDA manner to obtain the highest
#AUC score for appetency.
#
#The variable selection process was based on the smallest deviance of each variable.
#This variable selection process resulted in 31 variables out of 230 with deviance of 186.294
#based on the Calibration data set.
#
#The Calibration data set is a 10% random selection of observations from the original data set.
#
#The resulting Naive Bayes model using the selected variables shows that the model is overfitting
#the data because the AUC Score with the Train data is 0.9619 but the AUC Score with the Test data
#is 0.7624, which is about a 20-point difference.  However, the AUC for the Test is significantly
#above 0.50 of a random guess, so we could consider the Naive Bayes model for appetency
#to be reasonably accurate.
#
#################################################################################
setwd(".R/data")

###   READ DATA FILES:
d <- read.table('orange_small_train.data',
                header=T,
                sep='\t',
                na.strings=c('NA',''))

churn <- read.table('orange_small_train_churn.labels',
                    header=F,sep='\t')
d$churn <- churn$V1

upselling <- read.table('orange_small_train_upselling.labels',
                        header=F,sep='\t')
d$upselling <- upselling$V1

appetency <- read.table('orange_small_train_appetency.labels',
                        header=F,sep='\t')
d$appetency <- appetency$V1


###   CREATING TRAIN, CALIBRATION, AND TEST DATA SETS
# get the index for training/testing data
# this portion of the code should be copied exactly
# in every data transformation script
# that way we will all be using the same training/testing data

set.seed(123)
smp_size <- floor(0.70 * nrow(d))
test_ind <- seq_len(nrow(d))
train_ind <- sample(test_ind, size = smp_size)
# remove train observations from test
test_ind <- test_ind[! test_ind %in% train_ind]

# create an ensemble test set
set.seed(123)
smp_size <- floor(0.15 * nrow(d))
ens_ind <- sample(test_ind, size = smp_size)
# remove ensemble observations from test
test_ind <- test_ind[! test_ind %in% ens_ind]
# partition the data
ensemble_test <- d[ens_ind, ]
train <- d[train_ind, ]
test <- d[test_ind, ]

tiny_ind <- sample(seq_len(nrow(d)), size = floor(0.10 * nrow(d)))
dCal <- d[tiny_ind, ]


###  SETTING OTHER VARIABLES:
outcomes=c('churn','appetency','upselling')

vars <- setdiff(colnames(train), c(outcomes,'rgroup'))

catVars <- vars[sapply(train[,vars],class) %in% c('factor','character')]

numericVars <- vars[sapply(train[,vars],class) %in% c('numeric','integer')]

rm(list=c('d','churn','appetency','upselling'))

outcome <- 'appetency'

pos <- '1'


# Title: Function to build single-variable models for categorical variables

mkPredC <- function(outCol,varCol,appCol) {
  pPos <- sum(outCol==pos)/length(outCol)
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[pos]
  vTab <- table(as.factor(outCol),varCol)
  pPosWv <- (vTab[pos,]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(appCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred
}

# Title: Applying single-categorical variable models to all of our datasets

for(v in catVars) {
  pi <- paste('pred',v,sep='')
  train[,pi] <- mkPredC(train[,outcome],train[,v],train[,v])
  ensemble_test[,pi] <- mkPredC(train[,outcome],train[,v],ensemble_test[,v])
  dCal[,pi] <- mkPredC(train[,outcome],train[,v],dCal[,v])
  test[,pi] <- mkPredC(train[,outcome],train[,v],test[,v])
}


# Title: Scoring categorical variables by AUC

library('ROCR')

calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

for(v in catVars) {
  pi <- paste('pred',v,sep='')
  aucTrain <- calcAUC(train[,pi],train[,outcome])
  if(aucTrain>=0.8) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    aucTest <- calcAUC(test[,pi],test[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f  testAUC: %4.3f",
                  pi,aucTrain,aucCal,aucTest))
  }
}

# [1] "predVar198, trainAUC: 0.879 calibrationAUC: 0.767  testAUC: 0.526"
# [1] "predVar199, trainAUC: 0.827 calibrationAUC: 0.766  testAUC: 0.549"
# [1] "predVar200, trainAUC: 0.847 calibrationAUC: 0.786  testAUC: 0.532"
# [1] "predVar202, trainAUC: 0.923 calibrationAUC: 0.837  testAUC: 0.571"
# [1] "predVar214, trainAUC: 0.847 calibrationAUC: 0.786  testAUC: 0.532"
# [1] "predVar217, trainAUC: 0.935 calibrationAUC: 0.858  testAUC: 0.576"
# [1] "predVar220, trainAUC: 0.879 calibrationAUC: 0.767  testAUC: 0.526"
# [1] "predVar222, trainAUC: 0.879 calibrationAUC: 0.767  testAUC: 0.526"



# Title: Scoring numeric variables by AUC
mkPredN <- function(outCol,varCol,appCol) {
  cuts <- unique(as.numeric(quantile(varCol,probs=seq(0, 1, 0.1),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}

# Error in cut(appCol, cuts) : argument "appCol" is missing, with no default

for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  train[,pi] <- mkPredN(train[,outcome],train[,v],train[,v])
  test[,pi] <- mkPredN(train[,outcome],train[,v],test[,v])
  ensemble_test[,pi] <- mkPredN(train[,outcome],dCal[,v])
  aucTrain <- calcAUC(train[,pi],train[,outcome])
  if(aucTrain>=0.55) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    aucTest <- calcAUC(test[,pi],test[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f TestAUC: %4.3f",
                  pi,aucTrain,aucCal,aucTest))
  }
}

# [1] "predVar6, trainAUC: 0.565 calibrationAUC: 0.571 TestAUC: 0.532"
# [1] "predVar21, trainAUC: 0.561 calibrationAUC: 0.548 TestAUC: 0.546"
# [1] "predVar22, trainAUC: 0.563 calibrationAUC: 0.547 TestAUC: 0.546"
# [1] "predVar24, trainAUC: 0.553 calibrationAUC: 0.535 TestAUC: 0.527"
# [1] "predVar25, trainAUC: 0.559 calibrationAUC: 0.556 TestAUC: 0.544"
# [1] "predVar28, trainAUC: 0.590 calibrationAUC: 0.577 TestAUC: 0.616"
# [1] "predVar38, trainAUC: 0.572 calibrationAUC: 0.568 TestAUC: 0.544"
# [1] "predVar73, trainAUC: 0.553 calibrationAUC: 0.562 TestAUC: 0.522"
# [1] "predVar76, trainAUC: 0.556 calibrationAUC: 0.558 TestAUC: 0.588"
# [1] "predVar81, trainAUC: 0.612 calibrationAUC: 0.599 TestAUC: 0.606"
# [1] "predVar109, trainAUC: 0.554 calibrationAUC: 0.526 TestAUC: 0.544"
# [1] "predVar112, trainAUC: 0.565 calibrationAUC: 0.545 TestAUC: 0.512"
# [1] "predVar119, trainAUC: 0.584 calibrationAUC: 0.573 TestAUC: 0.559"
# [1] "predVar123, trainAUC: 0.561 calibrationAUC: 0.519 TestAUC: 0.517"
# [1] "predVar126, trainAUC: 0.751 calibrationAUC: 0.717 TestAUC: 0.752"
# [1] "predVar133, trainAUC: 0.562 calibrationAUC: 0.551 TestAUC: 0.617"
# [1] "predVar134, trainAUC: 0.568 calibrationAUC: 0.576 TestAUC: 0.569"
# [1] "predVar149, trainAUC: 0.556 calibrationAUC: 0.535 TestAUC: 0.530"
# [1] "predVar153, trainAUC: 0.594 calibrationAUC: 0.605 TestAUC: 0.617"
# [1] "predVar160, trainAUC: 0.553 calibrationAUC: 0.518 TestAUC: 0.519"
# [1] "predVar163, trainAUC: 0.554 calibrationAUC: 0.514 TestAUC: 0.569"
# [1] "predVar189, trainAUC: 0.571 calibrationAUC: 0.621 TestAUC: 0.575"


# Title: Basic variable selection
logLikelyhood <- function(outCol,predCol) {
  sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
}

selVars <- c() # this variable stores the selected variable
minStep <- 5
baseRateCheck <- logLikelyhood(dCal[,outcome],
                               sum(dCal[,outcome]==pos)/length(dCal[,outcome]))

for(v in catVars) {
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) -
                   baseRateCheck))
  if(liCheck>minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}


# "predVar192, calibrationScore: 35.6552"
# [1] "predVar193, calibrationScore: 10.072"
# [1] "predVar199, calibrationScore: 37.2442"
# [1] "predVar200, calibrationScore: 180.268"
# [1] "predVar202, calibrationScore: 45.6126"
# [1] "predVar204, calibrationScore: 45.0995"
# [1] "predVar206, calibrationScore: 10.9268"
# [1] "predVar207, calibrationScore: 5.23037"
# [1] "predVar211, calibrationScore: 7.14388"
# [1] "predVar212, calibrationScore: 27.6835"
# [1] "predVar214, calibrationScore: 180.268"
# [1] "predVar216, calibrationScore: 32.0394"
# [1] "predVar217, calibrationScore: 125.581"
# [1] "predVar218, calibrationScore: 50.821"
# [1] "predVar221, calibrationScore: 6.50989"
# [1] "predVar225, calibrationScore: 23.221"
# [1] "predVar226, calibrationScore: 30.0101"
> 


for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) -
                   baseRateCheck))
  if(liCheck>=minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}

# [1] "predVar28, calibrationScore: 7.15878"
# [1] "predVar34, calibrationScore: 7.86954"
# [1] "predVar126, calibrationScore: 63.8675"
# [1] "predVar153, calibrationScore: 9.8185"
# [1] "predVar162, calibrationScore: 5.48982"
# [1] "predVar189, calibrationScore: 16.2407"


# Title: Plotting the receiver operating characteristic curve

library(ggplot2)
plotROC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'tpr','fpr')
  pf <- data.frame(
    FalsePositiveRate=perf@x.values[[1]],
    TruePositiveRate=perf@y.values[[1]])
  ggplot() +
    geom_line(data=pf,aes(x=FalsePositiveRate,y=TruePositiveRate)) +
    geom_line(aes(x=c(0,1),y=c(0,1)))
}



# Title: Using a Naive Bayes package

library('e1071')
# with only the selVars the variables:
ff <- paste('as.factor(',outcome,'>0) ~ ', paste(selVars,collapse=' + '),sep='')

nbmodel <- naiveBayes(as.formula(ff),data=train)

train$nbpred <- predict(nbmodel,newdata=train,type='raw')[,'TRUE']
# Error: cannot allocate vector of size 118.8 Mb
dCal$nbpred <- predict(nbmodel,newdata=dCal,type='raw')[,'TRUE']
test$nbpred <- predict(nbmodel,newdata=test,type='raw')[,'TRUE']

calcAUC(train$nbpred,train[,outcome])
## [1] 0.9618698  # with selVars
calcAUC(dCal$nbpred,dCal[,outcome])
## [1] 0.9069976
## [1] 0.9128786? with selVars
calcAUC(test$nbpred,test[,outcome])
## [1] 0.7998207
## [1] 0.7624391 with selVars

print(plotROC(train$nbpred,train[,outcome]))
print(plotROC(dCal$nbpred,dCal[,outcome]))
print(plotROC(test$nbpred,test[,outcome]))

#### Model Output .RData for Project:
appetency_nb_model <- naiveBayes(as.formula(ff),data=train)
appetency_nb_predictions <-predict(
  appetency_nb_model,newdata=test,type='raw')[,'TRUE']

app_ens_nb_pred <-predict(appetency_nb_model,
                                 newdata=ensemble_test, type='raw')[,'TRUE']


###   SET DIRECTORY PATH:
# save the output
save(list = c('appetency_nb_model', 'appetency_nb_predictions',
              'app_ens_sandra_pred'),
     file = 'appetency_nb.RData')
     # file = 'models/appetency/appetency_nb.RData')