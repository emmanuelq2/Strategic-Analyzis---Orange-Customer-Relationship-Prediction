library(ROCR)
library(ggplot2)
library(randomForest)



for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}

# load the predictions for everyone
load("churn_lreg_jay.RData")
load("churn_nb_sandra.RData")
load("churn_rf_manjari.RData")
load("churn_lreg_udy.RData")
load("churn_rf_jay.RData")

source('kdd_tools.r')
source('impute_0.r')
load("data_transformations.RData")

my_colors <- c('#1f78b4', '#33a02c', '#e31a1c', '#ff7f00',
               '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a',
               '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99')

# find the mean of the scaled predictions as simple ensemble
# for some reason the niave bayes modelis giving the same output for all
# so I did not include it in the ensemble
model_responses <- data.frame(
  churn = test_response$churn,
  churn_rf_predictions,
  churn_lreg_fitLASSO_predictions,
  # churn_nb_predictions,
  churn_lreg_predictions
)

model_responses_ens <- data.frame(
  # churn = test_response$churn,
  churn_rf_predictions = churn_ens_rf_predictions,
  churn_lreg_fitLASSO_predictions = churn_ens_lreg_fitLASSO_predictions,
  # churn_nb_predictions = churn_ens_nb_predictions,
  churn_lreg_predictions = churn_ens_lreg_predictions
)

churn_ens_rf <- randomForest(factor(churn) ~ ., data = model_responses,
                             nodesize = 4, ntree = 100,
                             strata = factor(model_responses$churn),
                             sampsize = c(500, 500))

churn_ens_rf_predictions <- predict(churn_ens_rf, model_responses_ens,
                                    type = 'prob')[,2]

churn_vote <- rowSums(data.frame(
  scale_vec(churn_ens_rf_predictions),
  scale_vec(churn_ens_lreg_fitLASSO_predictions),
  # scale_vec(churn_ens_nb_sandra_predictions),
  scale_vec(churn_ens_lreg_predictions)
))/3

# combine all predictions
churn_df <- data.frame(churn = ens_response$churn,
                       logistic_regression = churn_ens_lreg_predictions,
                       logistic_regression2 = churn_ens_lreg_fitLASSO_predictions,
                       #  naive_bayes = churn_ens_nb_predictions,
                       random_forest = churn_ens_rf_predictions,
                       vote_ensemble = churn_vote,
                       ## rf_ensemble = churn_ens_rf_predictions)

library(tidyr)
churn_df2 <- gather(churn_df, churn, 'prediction')
names(churn_df2) <- c('true_value', 'algorithm', 'prediction')

churn_roc_df <- make_roc(churn_df2, ens_response$churn)
# plot results
ggplot(data = churn_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                                colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Churn Models')

make_auc(churn_df2, ens_response$churn, 0.7435)

## AUC
# In_House             0.7435000
# logistic_regression  0.7155615
# logistic_regression2 0.6120991
# random_forest        0.6957932
# vote_ensemble        0.7250015
# rf_ensemble          0.7035371
