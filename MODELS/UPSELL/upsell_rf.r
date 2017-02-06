library(randomForest)
library(dplyr)
library(ROCR)


for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}


# choose a script to load and transform the data
source('impute_0.r')

# the data needs to be in matrix form, so I'm using make_mat()
# from kdd_tools.r
source('kdd_tools.r')

# over sample the possitive instances of upsell
train_over <- rbind(train, train[train$upsell == 1,],
               train[train$upsell == 1,],
               train[train$upsell == 1,])
train_over <- select(train_over, -appetency, - churn)
train_over = (train_over[,c(1:212,384)])
train_over = (train_over[1:15000,])

ensemble_over <- (ensemble_test[,c(1:212,386)])
ensemble_over <- rbind(ensemble_over, ensemble_over[ensemble_over$upsell == 1,],
               ensemble_over[ensemble_over$upsell == 1,],
              ensemble_over[ensemble_over$upsell == 1,])
# ensemble <- select(ensemble, -appetency, - churn)



# set.seed(434)
upsell_rf_ens <- randomForest(factor(upsell) ~ ., data = ensemble_over,
                             nodesize = 4, ntree = 100,
                              strata = factor(ensemble_over$upsell),
                              sampsize = c(100, 100),
                              importance = TRUE)

set.seed(434)
upsell_rf_train <- randomForest(factor(upsell) ~ ., data = train_over,
                              nodesize = 4, ntree = 100,
                              strata = factor(train_over$upsell),
                              sampsize = c(100, 100),
                              importance = TRUE)


upsell.varImp <- importance(upsell_rf_jay)
upsell.selVars <- names(sort(upsell.varImp[,1],decreasing=T))[1:50]
set.seed(434)
upsell_rf <- randomForest(x=train[,upsell.selVars], y=factor(train$upsell),
                              nodesize = 1, ntree = 100,
                              strata = factor(train$upsell),
                              sampsize = c(100, 100),
                              importance = TRUE)


upsell_rf_predictions <- predict(upsell_rf, test,
                                     type = 'prob')[,2]

upsell_ens_rf_predictions <- predict(upsell_rf, ensemble_test,
                                         type = 'prob')[,2]

pred <- prediction(upsell_rf_jay_predictions, test$upsell)
perf <- performance(pred,'auc')
perf@y.values
# [1] 0.834602

library(e1071)
library(caret)
ensemble_test$Pred <- predict(upsell_rf_jay, type="class", newdata=ensemble_test)
upsell.pred.rp.conf.matrix.ensemble <- confusionMatrix(ensemble_test$Pred,ensemble_test$upsell)

# Confusion Matrix and Statistics

#          Reference
#  Prediction    0    1
#            0 4221   91
#            1 2731  457

# Accuracy : 0.6237          
# 95% CI : (0.6127, 0.6347)
# No Information Rate : 0.9269          
# P-Value [Acc > NIR] : 1               

# Kappa : 0.137           
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity/Recall: probability of detecting yes?
# Sensitivity : 0.6072  : actually, SPEFICITY: When it's actually no, how often does it predict no?
# sensitivity <- 4221 / (4221 + 2731) : When it's actually no, how often does it predict no?

# ACTUALLY, not speficity, but sensitivity / recall: 457/(457+91) = 0.8339
# Specificity : 0.8339      
# specificity <- 457/(457+91) probability of detecting yes?

# ADDED: Precision: When it predicts yes, how often is it correct?
# Precision: 457 / (457+2731)  = 0.1433501

# Pos Pred Value : 0.9789 - NPV <- 4221 / 4312 = 0.9788961   
# Neg Pred Value : 0.1434 - PPV <- 457 / (457+2731) = 0.1433501        
# Prevalence : 0.9269          (4231+2731)/(4231+2731+457+91)
# Detection Rate : 0.5628      (4231)/(4231+2731+457+91)    
# Detection Prevalence : 0.5749     (4231+91)/(4231+2731+457+91)     
# Balanced Accuracy : 0.7206       (0.8339+0.6072)/2 

# 'Positive' Class : 0 

upsell.pred.rp.conf.matrix.ensemble <- confusionMatrix(ensemble_test$Pred,ensemble_test$upsell, 
                                                       positive="1")
print(upsell.pred.rp.conf.matrix.ensemble)

# Confusion Matrix and Statistics

#        Reference
#  Prediction    0    1
#            0 4221   91
#           1 2731   457

# Accuracy : 0.6237          
# 95% CI : (0.6127, 0.6347)
# No Information Rate : 0.9269          
# P-Value [Acc > NIR] : 1               

# Kappa : 0.137           
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.83394         - 457/(457+91) = 0.8339
# Specificity : 0.60716         4221 / (4221 + 2731)
# Pos Pred Value : 0.14335  - PPV <- 457 / (457+2731) = 0.1433501       
# Neg Pred Value : 0.97890   - NPV <- 4221 / 4312 = 0.9788961
# Prevalence : 0.07307      -   (457+91)/(4231+2731+457+91)
# Detection Rate : 0.06093        (457)/(4231+2731+457+91) 
# Detection Prevalence : 0.42507      (457+2731)/(4231+2731+457+91)    
# Balanced Accuracy : 0.72055         (0.8339+0.6072)/2

# 'Positive' Class : 1        

# In relation to Bayesian statistics, the sensitivity and specificity are the conditional 
# probabilities, 
# the prevalence is the prior, and the positive/negative predicted values are the posterior 
# probabilities

#  If a randomly selected individual tests positive, what is the probability that he is a user?
# p <- 0.0734
# sensitivity = 0.83394
# specificity = 0.60716
# prob = (p*sensitivity) / ((p*sensitivity) + ((1-specificity) * (1-p)))
# [1] 0.1439529

save(list = c('upsell_rf_predictions', 'upsell_ens_rf_predictions'),
     file = 'upsell_rf.RData')