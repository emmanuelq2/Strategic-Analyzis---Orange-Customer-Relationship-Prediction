
##  churn_nb

##################################################################################################
###       COMMENT FOR NAIVE MODEL FOR CHURN
##################################################################################################
#The Na???ve Bayes technique was applied in a computational EDA manner to obtain the highest AUC
#score for churn.
#
#The variable selection process was based on the smallest deviance of each variable.
#This variable selection process resulted in 47 variables out of 230 with deviance of 291.862
#based on the Calibration data set.
#
#The Calibration data set is a 10% random selection of observations from the original data set.

#The resulting Naive Bayes model using the selected variables shows that the model is
#overfitting the data because the AUC Score with the Train data is 0.9315 but the AUC Score
#with the Test data is 0.6622, which is about a 27-point difference.  However, the AUC for
#the Test is significantly above 0.50 of a random guess, so we could consider the Naive Bayes
#model for churn to be reasonably accurate.
#
##################################################################################################
##################################################################################################

###   SET DIRECTORY PATH:


for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}

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
outcome <- 'churn'
pos <- '1'


# Title: Function to build single-variable models for categorical variables
# example 6.4
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
# example 6.5
for(v in catVars) {
  pi <- paste('pred',v,sep='')
  train[,pi] <- mkPredC(train[,outcome],train[,v],train[,v])
  ensemble_test[,pi] <- mkPredC(train[,outcome],train[,v],ensemble_test[,v])
  dCal[,pi] <- mkPredC(train[,outcome],train[,v],dCal[,v])
  test[,pi] <- mkPredC(train[,outcome],train[,v],test[,v])
}


# Title: Scoring categorical variables by AUC
# example 6.6
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

# [1] "predVar200, trainAUC: 0.831 calibrationAUC: 0.754  testAUC: 0.537"
# [1] "predVar202, trainAUC: 0.842 calibrationAUC: 0.761  testAUC: 0.532"
# [1] "predVar214, trainAUC: 0.831 calibrationAUC: 0.754  testAUC: 0.537"
# [1] "predVar217, trainAUC: 0.906 calibrationAUC: 0.821  testAUC: 0.546"

# Title: Scoring numeric variables by AUC
mkPredN <- function(outCol,varCol,appCol) {
  cuts <- unique(as.numeric(quantile(varCol,probs=seq(0, 1, 0.1),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}

for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  train[,pi] <- mkPredN(train[,outcome],train[,v],train[,v])
  test[,pi] <- mkPredN(train[,outcome],train[,v],test[,v])
  ensemble_test[,pi] <- mkPredN(train[,outcome],train[,v],ensemble_test[,v])
  dCal[,pi] <- mkPredN(train[,outcome],train[,v],dCal[,v])
  aucTrain <- calcAUC(train[,pi],train[,outcome])
  if(aucTrain>=0.55) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    aucTest <- calcAUC(test[,pi],test[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f TestAUC: %4.3f",
                  pi,aucTrain,aucCal,aucTest))
  }
}


# [1] "predVar6, trainAUC: 0.555 calibrationAUC: 0.561 TestAUC: 0.561"
# [1] "predVar7, trainAUC: 0.559 calibrationAUC: 0.565 TestAUC: 0.559"
# [1] "predVar13, trainAUC: 0.568 calibrationAUC: 0.563 TestAUC: 0.556"
# [1] "predVar73, trainAUC: 0.604 calibrationAUC: 0.602 TestAUC: 0.615"
# [1] "predVar74, trainAUC: 0.575 calibrationAUC: 0.568 TestAUC: 0.569"
# [1] "predVar81, trainAUC: 0.553 calibrationAUC: 0.540 TestAUC: 0.558"
# [1] "predVar113, trainAUC: 0.563 calibrationAUC: 0.574 TestAUC: 0.544"
# [1] "predVar125, trainAUC: 0.551 calibrationAUC: 0.543 TestAUC: 0.546"
# [1] "predVar126, trainAUC: 0.632 calibrationAUC: 0.634 TestAUC: 0.643"
# [1] "predVar140, trainAUC: 0.562 calibrationAUC: 0.559 TestAUC: 0.561"
# [1] "predVar189, trainAUC: 0.574 calibrationAUC: 0.591 TestAUC: 0.581"

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

# [1] "predVar199, calibrationScore: 89.4399"
# [1] "predVar200, calibrationScore: 307.844"
# [1] "predVar204, calibrationScore: 16.0919"
# [1] "predVar205, calibrationScore: 17.3774"
# [1] "predVar206, calibrationScore: 33.7276"
# [1] "predVar207, calibrationScore: 15.605"
# [1] "predVar210, calibrationScore: 13.4692"
# [1] "predVar212, calibrationScore: 49.4103"
# [1] "predVar214, calibrationScore: 307.844"
# [1] "predVar216, calibrationScore: 96.7538"
# [1] "predVar217, calibrationScore: 215.153"
# [1] "predVar218, calibrationScore: 45.5074"
# [1] "predVar221, calibrationScore: 17.0659"
# [1] "predVar225, calibrationScore: 23.6971"
# [1] "predVar226, calibrationScore: 14.8155"
# [1] "predVar227, calibrationScore: 18.161"
# [1] "predVar228, calibrationScore: 27.7042"
# [1] "predVar229, calibrationScore: 23.6519"


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
# [1] "predVar6, calibrationScore: 18.4885"
# [1] "predVar7, calibrationScore: 20.2423"
# [1] "predVar13, calibrationScore: 17.3254"
# [1] "predVar16, calibrationScore: 5.07644"
# [1] "predVar28, calibrationScore: 9.39595"
# [1] "predVar38, calibrationScore: 5.53493"
# [1] "predVar51, calibrationScore: 6.64316"
# [1] "predVar65, calibrationScore: 6.04579"
# [1] "predVar73, calibrationScore: 46.6431"
# [1] "predVar74, calibrationScore: 22.515"
# [1] "predVar81, calibrationScore: 6.15129"
# [1] "predVar85, calibrationScore: 7.61005"
# [1] "predVar113, calibrationScore: 20.0959"
# [1] "predVar125, calibrationScore: 10.9109"
# [1] "predVar126, calibrationScore: 98.5053"
# [1] "predVar133, calibrationScore: 5.45914"
# [1] "predVar140, calibrationScore: 19.475"
# [1] "predVar153, calibrationScore: 6.32451"
# [1] "predVar188, calibrationScore: 8.30214"
# [1] "predVar189, calibrationScore: 47.4864"


# Title: Plotting the receiver operating characteristic curve
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
## [1] 0.9315453  # with selVars
calcAUC(dCal$nbpred,dCal[,outcome])
## [1] 0.8739238  # with selVars
# [1] 0.8708916 (mine)
calcAUC(test$nbpred,test[,outcome])
## [1] 0.6622495  # with selVars
# [1] 0.6671416 (mine)
# install.packages("ggplot2")

require("ggplot2")
print(plotROC(train$nbpred,train[,outcome]))
print(plotROC(dCal$nbpred,dCal[,outcome]))
print(plotROC(test$nbpred,test[,outcome]))

#### Model Output .RData for Project:
churn_nb_model <- naiveBayes(as.formula(ff),data=train)
churn_nb_predictions <-predict(churn_nb_model,
                                      newdata=test,type='raw')[,'TRUE']

churn_ens_nb_predictions <-predict(churn_nb_model,
                                          ensemble_test,type='raw')[,'TRUE']

# save the output
save(list = c('churn_nb_model', 'churn_nb_predictions',
              'churn_ens_nb_predictions'),
     file = 'models/churn_nb.RData')