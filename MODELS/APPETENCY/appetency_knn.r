setwd(".R/data")

###   READ DATA FILES
df <- read.table('orange_small_train.data',
                header=T,
                sep='\t',
                na.strings=c('NA',''))

churn <- read.table('orange_small_train_churn.labels',
                    header=F,sep='\t')
df$churn <- churn$V1

upselling <- read.table('orange_small_train_upselling.labels',
                        header=F,sep='\t')
df$upselling <- upselling$V1

appetency <- read.table('orange_small_train_appetency.labels',
                        header=F,sep='\t')
df$appetency <- appetency$V1


###   CREATING TRAIN, CALIBRATION, AND TEST DATA SETS
# get the index for training/testing data
set.seed(123)
smp_size <- floor(0.85 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
# making a "tiny" data set so I can quickly
# test r markdown and graphical paramters
# this will be removed in the submitted version
tiny_ind <- sample(seq_len(nrow(df)), size = floor(0.01 * nrow(df)))
# split the data
train <- df[train_ind, ]
test <- df[-train_ind, ]
tiny <- df[tiny_ind, ]

# create a validation set
set.seed(123)
smp_size <- 7500
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

ensemble_test <- train[-train_ind, ]
train <- df[train_ind, ]
dCal <- df[tiny_ind, ]


###
set.seed(123)
smp_size <- floor(0.85 * nrow(df))
test_ind <- seq_len(nrow(df))
train_ind <- sample(test_ind, size = smp_size)
# remove train observations from test
test_ind <- test_ind[! test_ind %in% train_ind]

# create an ensemble test set
set.seed(123)
smp_size <- floor(0.15 * nrow(df))
ens_ind <- sample(test_ind, size = smp_size)
# remove ensemble observations from test
test_ind <- test_ind[! test_ind %in% ens_ind]
# partition the data
ensemble_test <- df[ens_ind, ]
train <- df[train_ind, ]
test <- df[test_ind, ]

tiny_ind <- sample(seq_len(nrow(df)), size = floor(0.01 * nrow(df)))
dCal <- df[tiny_ind, ]


###



###  SETTING OTHER VARIABLES:
outcomes=c('churn','appetency','upselling')
vars <- setdiff(colnames(train), c(outcomes,'rgroup'))
catVars <- vars[sapply(train[,vars],class) %in% c('factor','character')]
numericVars <- vars[sapply(train[,vars],class) %in% c('numeric','integer')]
rm(list=c('df','churn','appetency','upselling'))
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
  tiny[,pi] <- mkPredC(train[,outcome],train[,v],tiny[,v])
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
    aucCal <- calcAUC(tiny[,pi],tiny[,outcome])
    aucTest <- calcAUC(test[,pi],test[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f  testAUC: %4.3f",
                  pi,aucTrain,aucCal,aucTest))
  }
}

# [1] "predVar192, trainAUC: 0.850 calibrationAUC: 0.660  testAUC: 0.591"
# [1] "predVar198, trainAUC: 0.951 calibrationAUC: 0.482  testAUC: 0.549"
# [1] "predVar199, trainAUC: 0.943 calibrationAUC: 0.366  testAUC: 0.614"
# [1] "predVar200, trainAUC: 0.863 calibrationAUC: 0.540  testAUC: 0.613"
# [1] "predVar202, trainAUC: 0.978 calibrationAUC: 0.524  testAUC: 0.537"
# [1] "predVar214, trainAUC: 0.863 calibrationAUC: 0.540  testAUC: 0.613"
# [1] "predVar216, trainAUC: 0.817 calibrationAUC: 0.569  testAUC: 0.614"
# [1] "predVar217, trainAUC: 0.980 calibrationAUC: 0.480  testAUC: 0.533"
# [1] "predVar220, trainAUC: 0.951 calibrationAUC: 0.482  testAUC: 0.549"
# [1] "predVar222, trainAUC: 0.951 calibrationAUC: 0.482  testAUC: 0.549"


# Title: Scoring numeric variables by AUC
mkPredN <- function(outCol,varCol,appCol) {
  cuts <- unique(as.numeric(quantile(varCol,probs=seq(0, 1, 0.001),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}

for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  train[,pi] <- mkPredN(train[,outcome],train[,v],train[,v])
  test[,pi] <- mkPredN(train[,outcome],train[,v],test[,v])
  tiny[,pi] <- mkPredN(train[,outcome],train[,v],tiny[,v])
  aucTrain <- calcAUC(train[,pi],train[,outcome])
  if(aucTrain>=0.55) {
    aucCal <- calcAUC(tiny[,pi],tiny[,outcome])
    aucTest <- calcAUC(test[,pi],test[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f TestAUC: %4.3f",
                  pi,aucTrain,aucCal,aucTest))
  }
}


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



# Title: Running k-nearest neighbors

library('class')

nK <- 200
knnTrain <- dTrain[,selVars]
knnCl <- dTrain[,outcome]==pos

knnPred <- function(df) {
  knnDecision <- knn(knnTrain,df,knnCl,k=nK,prob=T)
  ifelse(knnDecision==TRUE,
         attributes(knnDecision)$prob,
         1-(attributes(knnDecision)$prob))
}

dTrain.AUC <- calcAUC(knnPred(dTrain[,selVars]),dTrain[,outcome])
dTrain.AUC

dCal.AUC <- calcAUC(knnPred(dCal[,selVars]),dCal[,outcome])
dCal.AUC

dTest.AUC <- calcAUC(knnPred(dTest[,selVars]),dTest[,outcome])
dTest.AUC


# Title: Plotting 200-nearest neighbor performance


###  TRAIN KNN PREDICTIONS:
dTrain$kpred <- knnPred(dTrain[,selVars])

#  Create a vector of the predictions to be exporeted to a file:
appetency.knnTrainPred <- dTrain$kpred
# save the output
save(list = c('appetency.knnTrainPred'),
     file = 'models/appetency/appetency.knnTrainPred.RData')

# plot the predictions
plotROC(dTrain$kpred,dTrain[,outcome])

install.packages("ggplot2")
require("ggplot2")
ggplot(data=dTrain) +
  geom_density(aes(x=kpred,
                   color=as.factor(appetency),linetype=as.factor(appetency)))


###  CALIBRATION KNN PREDICTIONS:
dCal$kpred <- knnPred(dCal[,selVars])

#  Create a vector of the predictions to be exporeted to a file:
appetency.knnCalPred <- dCal$kpred
# save the output
save(list = c('appetency.knnCalPred'),
     file = 'models/appetency/appetency.knnCalPred.RData')

# plot the predictions
plotROC(dCal$kpred,dCal[,outcome])

ggplot(data=dCal) +
  geom_density(aes(x=kpred,
                   color=as.factor(appetency),linetype=as.factor(appetency)))


###  TEST KNN PREDICTIONS:
dTest$kpred <- knnPred(dTest[,selVars])

#  Create a vector of the predictions to be exporeted to a file:
appetency.knnTestPred <- dTest$kpred
# save the output

save(list = c('appetency.knnTestPred'),
     file = 'models/appetency/appetency.knnTestPred.RData')

# plot the predictions
plotROC(dTest$kpred,dTest[,outcome])

ggplot(data=dTest) +
  geom_density(aes(x=kpred,
                   color=as.factor(appetency),linetype=as.factor(appetency)))

