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

# over sample the possitive instances of churn
train <- select(train, -upsell, -appetency)

set.seed(2651)
churn_rf <- randomForest(factor(churn) ~ ., data = train,
                             nodesize = 4, ntree = 100,
                             strata = factor(train$churn),
                             sampsize = c(2500, 2500))

churn_rf_predictions <- predict(churn_rf_jay, test,
                                    type = 'prob')[,2]

churn_ens_rf_predictions <- predict(churn_rf, ensemble_test,
                                        type = 'prob')[,2]

pred <- prediction(churn_rf_predictions, test$churn)
perf <- performance(pred,'auc')
perf@y.values
# [1] 0.7106634

save(list = c('churn_rf_predictions', 'churn_rf_jay',
              'churn_ens_rf_predictions'),
     file = 'churn_rf.RData')