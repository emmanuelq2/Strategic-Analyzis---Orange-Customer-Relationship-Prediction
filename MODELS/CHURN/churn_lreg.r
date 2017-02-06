
# choose a script to load and transform the data
source('impute_0.r')

# the data needs to be in matrix form, so I'm using make_mat()
# from kdd_tools.r
source('kdd_tools.r')
df_mat <- make_mat(df)

library(glmnet)

library(Matrix)


churn_lreg <- cv.glmnet(df_mat[train_ind,],
                            factor(train$churn), family = "binomial",
                            nfolds = 4, type.measure = 'auc')
# make predictions

churn_lreg_predictions <- predict(churn_lreg, df_mat[test_ind,],
                                    type = 'response', s = 'lambda.min')[,1]

churn_ens_lreg_predictions <- predict(churn_lreg, df_mat[ens_ind,],
                                          type = 'response', s = 'lambda.min')[,1]


library(data.table)
library(knitr)
cv_coefs <- data.table(variable = row.names(coef(churn_lreg))[
  abs(as.vector(coef(churn_lreg))) > 1e-3],
  coeficient = coef(churn_lreg)[abs(coef(churn_lreg_jay)) > 1e-3])


# kable(cv_coefs[variable %like% '26'],
#      caption = "Variables Selected by Elastic-Net")


# kable(cv_coefs[abs(order(-coeficient)),],
#             caption = "Variables Selected by Elastic-Net")

kable(cv_coefs[order(abs(coeficient)),],
      caption = "Variables Selected by Elastic-Net")

# yhat <- predict(churn_lreg, df_mat[-train_ind,], type = 'response')
yhat <- predict(churn_lreg, df_mat[test_ind ,], type = 'response')

# Error in as.matrix(cbind2(1, newx) %*% nbeta) : 
#  error in evaluating the argument 'x' in selecting a method for function 'as.matrix': Error in cbind2(1, newx) %*% nbeta : 
#  Cholmod error 'out of memory' at file ../Core/cholmod_memory.c, line 147

pred <- prediction(yhat, factor(test$churn))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")

library(ggplot2)
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
save(list = c('churn_lreg', 'churn_lreg_predictions',
              'churn_ens_lreg_predictions'),
     file = 'churn_lreg.RData')
