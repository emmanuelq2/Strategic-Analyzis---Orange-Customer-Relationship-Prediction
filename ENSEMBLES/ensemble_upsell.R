library(ROCR)
library(ggplot2)
library(randomForest)
library(dplyr)
library(tidyr)
library(Matrix)



for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}
setwd("./R/data")

source('kdd_tools.r')
source('impute_0.r')
load("data_transformations.RData")

my_colors <- c('#1f78b4', '#33a02c', '#e31a1c', '#ff7f00',
               '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a',
               '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99')

load("upsell_lreg.RData")
load("upsell_rf.RData")
load("upsell_nb.RData")



upsell_vote <- rowSums(data.frame(
  scale_vec(upsell_ens_rf_predictions),
  scale_vec(upsell_ens_lreg_predictions),
  scale_vec(upsell_ens_nb_predictions),
))/3



upsell_df <- data.frame(
  upsell = ensemble_test$upsell,
  random_forest = upsell_ens_rf_predictions,
  naive_bayes = upsell_ens_nb_predictions,
  logistic_regression = upsell_ens_lreg_predictions,
  vote_ensemble = upsell_vote
)

upsell_train <- data.frame(
  upsell = test$upsell,
  random_forest = upsell_rf_predictions,
  naive_bayes = upsell_nb_predictions,
  logistic_regression = upsell_lreg_predictions
)

lreg_combiner <- glm(factor(upsell) ~ ., data = upsell_train,
                     family = 'binomial')

up_logistic_ens <- predict(lreg_combiner, upsell_df)
upsell_df$logistic_ensemble <- predict(lreg_combiner, upsell_df)

upsell_df2 <- gather(upsell_df, upsell, 'prediction')
names(upsell_df2) <- c('true_value', 'algorithm', 'prediction')

upsell_roc_df <- make_roc(upsell_df2, ens_response$upsell)
# plot results
ggplot(data = upsell_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                                 colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves upsell Models')

make_auc(upsell_df2, ens_response$upsell, 0.8975)

# In_House            0.8975000
# naive_bayes         0.7604086
# random_forest       0.8201009
# logistic_regression 0.8182955
# vote_ensemble       0.8465618
# logistic_ensemble   0.8462520

save(list = c('upsell_vote', 'up_logistic_ens'),
     file = 'upsell.RData')