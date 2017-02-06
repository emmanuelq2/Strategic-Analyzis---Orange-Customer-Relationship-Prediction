# this model is a regularized logisitc regression model to predict upsell
# the regularization parameter is selected automatically using cross validation

library(Matrix)
library(glmnet)
library(ROCR)
# setwd("./R/data")



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

# squared numeric variables
df = (df[,c(1:212,384:386)])
new_names <- paste(colnames(select(df, 1:174)), '_squared', sep ='')
df2 <- select(df, 1:174)**2
colnames(df2) <- new_names

df <- cbind(df,df2)

# make some interaction variables
df$Var126_28 <- df$Var126 * df$Var28
df$Var28_153 <- df$Var28 * df$Var153
df$Var125_81 <- df$Var125 * df$Var81

# create a matrix
df_mat <- make_mat(df)

upsell_lreg <- cv.glmnet(df_mat[train_ind,],
                             factor(train$upsell), family = "binomial",
                             nfolds = 4, type.measure = 'auc')

# coef.glmnet(upsell_lreg_jay, s ='lambda.min')
par(mar=c(1,1,1,1))
plot(upsell_lreg_jay)

# make predictions
upsell_lreg_predictions <- predict(upsell_lreg, df_mat[test_ind,],
                                       type = 'response', s = 'lambda.min')[,1]

upsell_ens_lreg_predictions <- predict(upsell_lreg, df_mat[ens_ind,],
                                           type = 'response', s = 'lambda.min')[,1]

# pred <- prediction(upsell_lreg_predictions, test$upsell)
# perf <- performance(pred,'auc')

yhat <- predict(upsell_lreg, df_mat[test_ind ,], type = 'response')
pred <- prediction(yhat, factor(test$upsell))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf@y.values
# [1] 0.7927



p <- ggplot(data.frame(TPR = unlist(perf@y.values),
                       FPR = unlist(perf@x.values)),
            aes(FPR, TPR)) + geom_line(color = '#3b5b92', size = 1) +
  xlab('False Positive Rate') + ylab('True Positive Rate') +
  ggtitle('Logistic Regression ROC Curve') +
  theme(plot.title = element_text(lineheight=.8, face="bold")) +
  annotate("text", x = 0.75, y = 0.20, label = sprintf('AUC: %f',
                   attributes(performance(pred, 'auc'))$y.values[[1]]))
p


# save the output
save(list = c('upsell_lreg', 'upsell_lreg_predictions',
              'upsell_ens_lreg_predictions'),
     file = 'upsell_lreg.RData')