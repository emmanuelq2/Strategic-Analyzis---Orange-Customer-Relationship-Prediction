library(ROCR)
library(ggplot2)
library(randomForest)
library(dplyr)
library(tidyr)



)

# for (d in dirs){
#  if(dir.exists(d)){
#    setwd(d)
#  }
#}


source('kdd_tools.r')
source('impute_0.r')
load("data_transformations.RData")

my_colors <- c('#1f78b4', '#33a02c', '#e31a1c', '#ff7f00',
               '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a',
               '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99')

# load the predictions 
load("appetency_nb.RData")
load("app_lreg.RData")

load("rf.RData")
load("app_lreg_fitLASSO.RData")



app_vote <- rowSums(data.frame(
  scale_vec(app_ens_rf_pred),
  scale_vec(app_ens_lreg_pred),
  scale_vec(app_ens_nb_pred),
  scale_vec(app_ens_lreg_predictions)
 ))/2

app_vote <- rowSums(data.frame(
  # scale_vec(app_ens_rf_pred),
  scale(app_ens_lreg_pred, scale= FALSE),
  # scale_vec(app_ens_nb_pred),
  scale(app_ens_lreg_predictions, scale=FALSE)
))/2


# dataframe to train neural network ensemble on

app_train <- data.frame(
  appetency = test$appetency,
  random_forest2 = app_rf_predictions,
  logistic_regression = app_lreg_pred,
  naive_bayes = appetency_nb_predictions,
  logistic_regression2 = app_lreg_predictions
)

app_train <- cbind(app_train, select(test))
# app_train <- cbind(app_train, select(test, -upsell, -churn))

set.seed(61)
rf_stack <- randomForest(factor(appetency) ~ ., data = app_train,
                         nodesize = 5, ntree = 100,
                         strata = factor(test$appetency),
                         sampsize = c(500, 100))
# Error in randomForest.default(m, y, ...) : 
#  cannot allocate memory block of size 7.6 Gb


# Error in randomForest.default(m, y, ...) : 
#  cannot allocate memory block of size 2.4 Gb

app_df <- data.frame(
  appetency = ensemble_test$appetency,
  random_forest2 = app_ens_rf_pred,
  logistic_regression = app_ens_lreg_pred,
  naive_bayes = app_ens_nb_pred,
  logistic_regression2 = app_ens_lreg_predictions,
  vote_ensemble = app_vote
)

rf_stack_pred <- predict(rf_stack,
                         cbind(ensemble_test, app_df),
                         type = 'prob')[,2]

app_df$stacked_random_forest <- rf_stack_pred

app_df2 <- gather(app_df, appetency, 'prediction')
names(app_df2) <- c('true_value', 'algorithm', 'prediction')


app_roc_df <- make_roc(app_df2, ens_response$appetency)
# plot results
ggplot(data = app_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                              colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Appetency Models')

make_auc(app_df2, ens_response$appetency, 0.853)

# In_House              0.8530000
# random_forest2        0.8324733
# logistic_regression   0.8057657
# naive_bayes           0.7793469
# logistic_regression2  0.8204741
# vote_ensemble         0.8357444
# stacked_random_forest 0.8291562

save(list = c('rf_stack_pred', 'app_vote'), file = 'ensembles/app.RData')