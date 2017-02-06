# choose a script to load and transform the data
source('data_transformations/impute_0.r')

# the data needs to be in matrix form, so I'm using make_mat()
# from kdd_tools.r
source('kdd_tools.r')
df_mat <- make_mat(df)

train <- select(train, -upsell, - churn)


library(randomForest)
library(dplyr)
library(ROCR)
set.seed(287)
app_rf <- randomForest(factor(appetency) ~ ., data = train,
                           nodesize = 4, ntree = 1000,
                           strata = factor(train$appetency),
                           sampsize = c(608, 608))


app_rf_predictions <- predict(app_rf_jay, test,
                                  type = 'prob')[,2]

app_ens_rf_pred <- predict(app_rf_jay, ensemble_test,
                               type = 'prob')[,2]


pred <- prediction(app_rf_jay_predictions, test$appetency)
perf <- performance(pred,'auc')
perf@y.values
# [1] 0.8373027


save(list =c('app_rf', 'app_rf_predictions', 'app_ens_rf_pred'),
     file = 'app_rf.RData')