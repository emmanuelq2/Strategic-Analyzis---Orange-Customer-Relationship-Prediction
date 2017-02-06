setwd("./R/data")


library(glmnet)

set.seed(123)
smp_size <- floor(0.70 * nrow(df))
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


library(glmnet)
source('kdd_tools.r')
df_mat <- make_mat(df)


app_lreg <- cv.glmnet(df_mat[train_ind,],
                factor(train$appetency), family = "binomial", nfolds = 4, type.measure = 'auc')


# make predictions
app_lreg_predictions <- predict(app_lreg, df_mat[test_ind,],
                                    type = 'response', s = 'lambda.min')[,1]


app_ens_lreg_predictions <- predict(app_lreg, df_mat[ens_ind,],
                                        type = 'response', s = 'lambda.min')[,1]

save(list = c('app_lreg', 'app_lreg_predictions',
              'app_ens_lreg_predictions'),
     file = 'app_lreg.RData')
