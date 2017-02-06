##################################################################################################
###       COMMENT FOR NAIVE MODEL FOR UPSELL
##################################################################################################
#The Na�ve Bayes technique was applied in a computational EDA manner to obtain the highest
#AUC score for upsell.
#
#The variable selection process was based on the smallest deviance of each variable.
#This variable selection process resulted in 51 variables out of 230 with deviance of 504.483
#based on the Calibration data set.
#
#The Calibration data set is a 10% random selection of observations from the original data set.
#
#The resulting Naive Bayes model using the selected variables shows that the model is overfitting
#the data because the AUC Score with the Train data is 0.9177 but the AUC Score with the
#Test data is 0.7515, which is about a 16-point difference.  However, the AUC for the Test
#is significantly above 0.50 of a random guess, so we could consider the Naive Bayes model
#for upsell to be reasonably accurate.
#
##################################################################################################
##################################################################################################

###   SET DIRECTORY PATH:


for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}

getwd()
[1] "C:/Users/emmanuel/Documents/R/data"

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

outcome <- 'upselling'

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

# [1] "predVar200, trainAUC: 0.879 calibrationAUC: 0.777  testAUC: 0.520"
# [1] "predVar202, trainAUC: 0.872 calibrationAUC: 0.744  testAUC: 0.511"
# [1] "predVar214, trainAUC: 0.879 calibrationAUC: 0.777  testAUC: 0.520"
# [1] "predVar217, trainAUC: 0.941 calibrationAUC: 0.815  testAUC: 0.535"

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
# [1] "predVar6, trainAUC: 0.578 calibrationAUC: 0.581 TestAUC: 0.570"
# [1] "predVar7, trainAUC: 0.557 calibrationAUC: 0.562 TestAUC: 0.544"
# [1] "predVar13, trainAUC: 0.566 calibrationAUC: 0.564 TestAUC: 0.543"
# [1] "predVar21, trainAUC: 0.581 calibrationAUC: 0.556 TestAUC: 0.563"
# [1] "predVar22, trainAUC: 0.580 calibrationAUC: 0.552 TestAUC: 0.562"
# [1] "predVar24, trainAUC: 0.573 calibrationAUC: 0.562 TestAUC: 0.559"
# [1] "predVar25, trainAUC: 0.584 calibrationAUC: 0.583 TestAUC: 0.585"
# [1] "predVar28, trainAUC: 0.684 calibrationAUC: 0.685 TestAUC: 0.683"
# [1] "predVar38, trainAUC: 0.578 calibrationAUC: 0.572 TestAUC: 0.582"
# [1] "predVar73, trainAUC: 0.583 calibrationAUC: 0.581 TestAUC: 0.561"
# [1] "predVar74, trainAUC: 0.554 calibrationAUC: 0.563 TestAUC: 0.552"
# [1] "predVar76, trainAUC: 0.585 calibrationAUC: 0.565 TestAUC: 0.550"
# [1] "predVar81, trainAUC: 0.611 calibrationAUC: 0.599 TestAUC: 0.603"
# [1] "predVar85, trainAUC: 0.573 calibrationAUC: 0.572 TestAUC: 0.571"
# [1] "predVar109, trainAUC: 0.577 calibrationAUC: 0.577 TestAUC: 0.574"
# [1] "predVar112, trainAUC: 0.576 calibrationAUC: 0.562 TestAUC: 0.550"
# [1] "predVar113, trainAUC: 0.589 calibrationAUC: 0.573 TestAUC: 0.570"
# [1] "predVar119, trainAUC: 0.582 calibrationAUC: 0.573 TestAUC: 0.581"
# [1] "predVar125, trainAUC: 0.568 calibrationAUC: 0.555 TestAUC: 0.538"
# [1] "predVar126, trainAUC: 0.789 calibrationAUC: 0.808 TestAUC: 0.790"
# [1] "predVar133, trainAUC: 0.592 calibrationAUC: 0.570 TestAUC: 0.577"
# [1] "predVar134, trainAUC: 0.582 calibrationAUC: 0.563 TestAUC: 0.577"
# [1] "predVar140, trainAUC: 0.563 calibrationAUC: 0.569 TestAUC: 0.542"
# [1] "predVar149, trainAUC: 0.573 calibrationAUC: 0.561 TestAUC: 0.549"
# [1] "predVar153, trainAUC: 0.611 calibrationAUC: 0.594 TestAUC: 0.599"
# [1] "predVar160, trainAUC: 0.573 calibrationAUC: 0.542 TestAUC: 0.562"
# [1] "predVar163, trainAUC: 0.575 calibrationAUC: 0.541 TestAUC: 0.564"

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



# [1] "predVar191, calibrationScore: 5.98527"
# [1] "predVar192, calibrationScore: 39.1512"
# [1] "predVar195, calibrationScore: 5.30838"
# [1] "predVar197, calibrationScore: 52.1645"
# [1] "predVar200, calibrationScore: 294.406"
# [1] "predVar204, calibrationScore: 22.045"
# [1] "predVar206, calibrationScore: 23.9314"
# [1] "predVar210, calibrationScore: 22.6213"
# [1] "predVar211, calibrationScore: 150.627"
# [1] "predVar214, calibrationScore: 294.406"
# [1] "predVar216, calibrationScore: 80.5126"
# [1] "predVar218, calibrationScore: 32.5672"
# [1] "predVar219, calibrationScore: 16.4077"
# [1] "predVar223, calibrationScore: 8.48873"
# [1] "predVar224, calibrationScore: 9.5161"
# [1] "predVar225, calibrationScore: 32.1731"
# [1] "predVar226, calibrationScore: 34.5282"
# [1] "predVar229, calibrationScore: 6.8917"

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

# [1] "predVar6, calibrationScore: 29.3358"
# [1] "predVar7, calibrationScore: 18.7526"
# [1] "predVar13, calibrationScore: 20.9999"
# [1] "predVar21, calibrationScore: 14.5261"
# [1] "predVar22, calibrationScore: 12.7259"
# [1] "predVar24, calibrationScore: 18.4081"
# [1] "predVar25, calibrationScore: 31.178"
# [1] "predVar28, calibrationScore: 156.355"
# [1] "predVar36, calibrationScore: 5.95646"
# [1] "predVar37, calibrationScore: 6.23699"
# [1] "predVar38, calibrationScore: 23.3527"
# [1] "predVar46, calibrationScore: 5.59008"
# [1] "predVar59, calibrationScore: 6.66413"
# [1] "predVar65, calibrationScore: 5.0019"
# [1] "predVar73, calibrationScore: 24.5221"
# [1] "predVar74, calibrationScore: 19.8508"
# [1] "predVar75, calibrationScore: 6.30185"
# [1] "predVar76, calibrationScore: 19.2425"
# [1] "predVar81, calibrationScore: 37.1083"
# [1] "predVar85, calibrationScore: 30.2555"
# [1] "predVar88, calibrationScore: 7.3312"
# [1] "predVar96, calibrationScore: 5.14811"
# [1] "predVar99, calibrationScore: 6.03045"
# [1] "predVar102, calibrationScore: 5.10137"
# [1] "predVar104, calibrationScore: 6.61716"
# [1] "predVar105, calibrationScore: 6.61716"
# [1] "predVar109, calibrationScore: 27.1646"
# [1] "predVar111, calibrationScore: 5.92325"
# [1] "predVar112, calibrationScore: 14.1176"
# [1] "predVar113, calibrationScore: 27.0169"
# [1] "predVar114, calibrationScore: 5.85683"
# [1] "predVar117, calibrationScore: 6.25053"
# [1] "predVar119, calibrationScore: 25.8223"
# [1] "predVar123, calibrationScore: 8.93693"
# [1] "predVar125, calibrationScore: 20.1316"
# [1] "predVar126, calibrationScore: 590.791"
# [1] "predVar128, calibrationScore: 7.3312"
# [1] "predVar133, calibrationScore: 19.9732"
# [1] "predVar134, calibrationScore: 5.99056"
# [1] "predVar135, calibrationScore: 15.1135"
# [1] "predVar139, calibrationScore: 5.11013"
# [1] "predVar140, calibrationScore: 22.0921"
# [1] "predVar145, calibrationScore: 8.81273"
# [1] "predVar149, calibrationScore: 14.8172"
# [1] "predVar150, calibrationScore: 5.09063"
# [1] "predVar152, calibrationScore: 5.98057"
# [1] "predVar153, calibrationScore: 31.4714"
# [1] "predVar162, calibrationScore: 5.20764"
# [1] "predVar170, calibrationScore: 5.34556"
# [1] "predVar171, calibrationScore: 6.78511"
# [1] "predVar174, calibrationScore: 6.30197"
# [1] "predVar177, calibrationScore: 5.96564"
# [1] "predVar178, calibrationScore: 6.13557"
# [1] "predVar182, calibrationScore: 8.21629"
# [1] "predVar184, calibrationScore: 5.99662"
# [1] "predVar188, calibrationScore: 12.8762"


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
dCal$nbpred <- predict(nbmodel,newdata=dCal,type='raw')[,'TRUE']
test$nbpred <- predict(nbmodel,newdata=test,type='raw')[,'TRUE']

calcAUC(train$nbpred,train[,outcome])
## [1] 0.9177398 # with selVars
# [1] 0.8537491
calcAUC(dCal$nbpred,dCal[,outcome])
## [1] 0.8827544 # with selVars
# [1] 0.8271452
calcAUC(test$nbpred,test[,outcome])
## [1] 0.7515061 # with selVars
# [1] 0.749751

print(plotROC(train$nbpred,train[,outcome]))
print(plotROC(dCal$nbpred,dCal[,outcome]))
print(plotROC(test$nbpred,test[,outcome]))

#### Model Output .RData for Project:
upsell_nb_model <- naiveBayes(as.formula(ff),data=train)
upsell_nb_predictions <-predict(upsell_nb_sandra_model,
                                       newdata=test,type='raw')[,'TRUE']

upsell_ens_nb_predictions <-predict(upsell_nb_model,
                                           ensemble_test,type='raw')[,'TRUE']

# save the output
save(list = c('upsell_nb_model', 'upsell_nb_predictions',
              'upsell_ens_nb_predictions'),
     file = 'upsell_nb.RData')