
# ---- libraries ----
library(lattice)
library(plyr)
library(dplyr)
library(tidyr)
library(grid)
library(gridExtra)
library(ROCR)
library(e1071)
library(knitr)
library(ggplot2)
library(data.table)
library(caret)
library(rpart)
library(rpart.plot)
library(FSelector)
# library(Matrix)
# ----

# ---- read_data ----

for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}
# choose a script to load and transform the data
source('data_transformations/impute_0.r')


#check for factor variables
names(df)
f <- sapply(df, is.factor)
which(f)


#Data set containing predictors only
df_mat <- select(df, -churn, -appetency, -upsell)

#Creating separate variables for the different factors
for (i in names(df_mat)){
  if (class(df_mat[,i]) == 'factor'){
    for(level in unique(df_mat[,i])){
      df_mat[sprintf('%s_dummy_%s', i, level)] <- ifelse(df_mat[,i] == level, 1, 0)
    }
    df_mat[,i] <- NULL
  } else {
    # scale numeric variables
    # this is important for regularized logistic regression and KNN
    df_mat[,i] <- scale(df_mat[,i])
  }
}

# Create input matrix
df_mat <- data.matrix(df_mat)
# Convert to data.frame
df_mat.frame <- data.frame(df_mat)
df_mat.frame$appetency <- df$appetency
names(df_mat.frame)
dim(df_mat.frame)

# Create train and test data set for data frame

set.seed(123)
smp_size <- floor(0.70 * nrow(df_mat.frame))
test_ind <- seq_len(nrow(df_mat.frame))
train_ind <- sample(test_ind, size = smp_size)
# remove train observations from test
test_ind <- test_ind[! test_ind %in% train_ind]
# create an ensemble test set
set.seed(123)
smp_size <- floor(0.15 * nrow(df_mat.frame))
ens_ind <- sample(test_ind, size = smp_size)
# remove ensemble observations from test
test_ind <- test_ind[! test_ind %in% ens_ind]
# partition the data
ensemble_test <- df_mat.frame[ens_ind, ]
train <- df_mat.frame[train_ind, ]
test <- df_mat.frame[test_ind, ]

# ----
#Exploratory decision Tree (Classification) for appetency
library(rpart)
library(rpart.plot)
df_mat.frame$appetency <- as.factor(df_mat.frame$appetency)
AppTree <- rpart(appetency~.,method="class",data = df_mat.frame,
                 control=rpart.control(minsplit=10, minbucket=10, cp=0.001))
AppTree
printcp(AppTree)
plot(AppTree, uniform=TRUE)
text(AppTree, all=TRUE,cex=0.75, splits=TRUE, use.n=TRUE, xpd=TRUE)
?predict
p <- predict(AppTree,newdata=df_mat.frame,type="class")
table(actual=df_mat.frame$appetency,predicted=p)

par(mar = c(5, 10, 4, 2) + 0.1)
barplot(AppTree$variable.importance,horiz=T,las=1, cex.names = 0.75)
par(mar = c(5, 4, 4, 2) + 0.1)


#GOF logistic regression using variables from exploratory Tree 
# lrApp.tree <- glm(appetency~Var112+Var113+Var126+Var140+
#                    Var153+Var21+Var216_dummy_beK4AFX+Var218_dummy_cJvF+Var24+
#                    Var38+Var6+Var76+Var81,data = train, family = binomial)

lrApp.tree <- glm(appetency~Var126+Var218+Var6+Var204+
                    Var153+Var81+Var38+Var113+
                    Var194+Var201+Var189+Var73+
                    Var125+Var225+Var76+Var119+
                    Var123+Var133+Var83+Var112,data = train, family = binomial)

summary(lrApp.tree)

#Refitted with statitically significant variables from exploratory tree
lrApp.treeRe <- glm(appetency~Var126+Var204+Var218+Var125+Var225,
                    data = train, family = binomial)

summary(lrApp.treeRe)

anova(lrApp.treeRe,lrApp.tree,test="Chisq")


par(mfrow=c(1,1))
fit <- lrApp.tree$fitted
hist(fit)
par(mfrow=c(1,1))

# PERFORMING FEATURE SELECTION USING FSELECTOR

#use random.forest.importance to calculate weight of each attribute
library(randomForest)
library(FSelector) # not available on 32 bits version
app.rf <- randomForest(appetency ~., data=train, ntree=1000,
                       keep.forest=FALSE, importance=TRUE)

weights = importance(app.rf, type = 1)
print(weights)

#use cutoff to obtain the top 30 attributes
subset = cutoff.k(weights, 30)
f = as.simple.formula(subset, "appetency")
print(f)



# LOGISTIC REGRESSION WITH LASSO EDA/VARIABLE SELECTION
# ---- LASSO_appetency ----

library(glmnet)
# regularized logistic regression with LASSO

train.glm <- as.matrix(train[-213])
dim(train.glm)
app.LASSO <- glmnet(train.glm,train$appetency,alpha=1, family = "binomial")

plot(app.LASSO,xvar="lambda",label=TRUE)
grid()
# Show number of selected variables
app.LASSO

#Use cross validation to select the best lambda (minlambda) and lambda.1se
#Cross Validated LASSO
appCVlasso <- cv.glmnet(train.glm,train$appetency)
#plot(fit.lasso,xvar="dev",label=TRUE)
plot(appCVlasso)
coef(appCVlasso)

coef(appCVlasso, s="lambda.min") # Gives coefficients lambda.min
best_lambda <- appCVlasso$lambda.min
best_lambda

attach(df_mat.frame)

#GOF logistic regression models LASSO Variables selected
names(train)

# lrAppLASSO <- glm(appetency~Var28+Var34+Var38+Var44+Var58+Var64+Var67+Var75+
#                    Var81+Var84+Var95+Var124+Var125+
#                    Var126+Var140+Var144+Var152+Var162+Var171+Var177+
#                    Var181+Var126_missing+Var194_dummy_SEuy+
#                    Var197_dummy_0Xwj+Var197_dummy_487l+Var197_dummy_TyGl+
#                    Var197_dummy_z32l+Var204_dummy_YULl+Var204_dummy_15m3+
#                    Var204_dummy_4N0K+Var204_dummy_m_h1+Var205_dummy_VpdQ+
#                    Var205_dummy_sJzTlal+Var206_dummy_zm5i+Var206_dummy_43pnToF+
#                    Var208_dummy_kIsH+Var210_dummy_uKAI+Var210_dummy_other+
#                    Var211_dummy_L84s+Var211_dummy_Mtgm+
#                    Var212_dummy_NhsEn4L+Var216_dummy_other+
#                    Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+
#                    Var218_dummy_cJvF+Var218_dummy_UYBR+Var219_dummy_OFWH+
#                    Var223_dummy_M_8D+Var226_dummy_xb3V+Var226_dummy_fKCe+
#                    Var226_dummy_Aoh3+Var226_dummy_uWr3+Var226_dummy_7P5s+
#                    Var228_dummy_R4y5gQQWY8OodqDV,data = train, family = binomial)

lrAppLASSO <- glm(appetency~Var28+Var34+Var38+Var44+Var64+Var67+
                    Var81+Var84+Var95+Var124+Var125+
                    Var126+Var140+Var144+Var152+Var153+Var162+Var171+Var177+
                    Var181+Var197+Var203+Var204+Var205+
                    Var206+Var208+Var210+Var212+Var216+Var217+
                    Var218+Var226,data = train, family = binomial)


summary(lrAppLASSO)
# par(mfrow=c(2,2))
# plot(lrAppLASSO)
# par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrAppLASSORe1 <- glm(appetency~Var34+Var38+Var44+Var64+Var67+Var81+Var84+Var125+
                Var126+Var140+Var144+Var152+Var153+Var162+Var171+Var177+
                Var197+Var204+Var205+Var206+Var210+Var212+Var217+
                Var218+Var226,data = train, family = binomial)

summary(lrAppLASSORe1)

# variables 44, 64, 144, 152, 153, 162, 204 dropped because they are not statistically insignificant.Model refitted
lrAppLASSORe <- glm(appetency~Var34+Var38+Var67+Var81+Var84+Var125+
                      Var126+Var140+Var171+Var177+Var197+Var205+Var206+Var210+Var212+Var217+
                      Var218+Var226,data = train, family = binomial)


summary(lrAppLASSORe)


#Test for better fit between nested models: chi-sq
anova(lrAppLASSORe,lrAppLASSO,test="Chisq")
#Test for better fit between nested models: AIC
AIC(lrAppLASSO,lrAppLASSORe)


par(mfrow=c(1,1))
fit.LASSO <- lrAppLASSO$fitted
hist(fit)
par(mfrow=c(2,2))
plot(lrAppLASSO)
# lrAppLASSO
# par(mfrow=c(1,1))


#################################################
# Chi-square goodness of fit test for LASSO Variables
#################################################
# Calculate residuals across all individuals
r.LASSO <- (df_mat.frame$appetency - fit.LASSO)/(sqrt(fit.LASSO*(1-fit.LASSO)))
r.LASSO <- (train$appetency - fit.LASSO)/(sqrt(fit.LASSO*(1-fit.LASSO)))
# Sum of squares of these residuals follows a chi-square
sum(r.LASSO^2)
# [1] 48585.26
#Calculate the p-value from the test
# 1- pchisq(60207.87, df=49966)
1- pchisq(48585.26, df=34966)

# ----


# # view the Area Under the Curve for different values of lambda.
# plot(churn_lreg.cv)
# title('Cross Validation Curve Logistic Regression',line =+2.8)
#
#
#
# cv_coefs <- data.table(variable = row.names(coef(churn_lreg.cv))[
#   abs(as.vector(coef(churn_lreg.cv))) > 1e-5],
#   coeficient = coef(churn_lreg.cv)[abs(coef(churn_lreg.cv)) > 1e-5])
#
#
# kable(cv_coefs[variable %like% '26'],
#       caption = "Variables Selected by Elastic-Net")




#RANDOM FOREST EDA/VARIABLE SELECTION

# ---- rf_churn ----
library(randomForest)
set.seed(101)
?randomForest
appRf <- randomForest(appetency~.,data = train,importance = TRUE)
# Error: cannot allocate vector of size 133.5 Mb
rf.important <- importance(appRf, type = 1)
barplot(rf.important)
# ----
# ---- plot_rf_churn ----
varImpPlot(appRf, type = 1, main = 'Variable Importance appetency')
varImpPlot(appRf, type = 2, main = 'Variable Importance appetency')

par(mar = c(5, 10, 4, 2) + 0.1)
barplot(rf.important,horiz=T)
par(mar = c(5, 4, 4, 2) + 0.1)

# write the variable importance to a file that can be read into excel
fo <- file("rf.txt", "w")
imp <- importance(appRf)
write.table(imp, fo, sep="\t")
flush(fo)
close(fo)

#GOF logistic regression using variables from Tree selection
lrAppRF <- glm(appetency~Var126+Var6+Var81+Var113+Var119+Var28+Var25+
                 Var85++Var22+Var73+Var153+Var83+
                 Var133+Var123+Var109+Var160+Var125+Var21+ Var134+Var74+Var112+Var38+Var163+Var76+Var140,
                  data = train, family = binomial)

summary(lrAppRF)
par(mfrow=c(2,2))
# Error in plot.new() : figure margins too large
plot(lrlrAppRF)
par(mfrow=c(1,1))

# Refitted with statitically significant variables
# lrAppRFRe <- glm(appetency~Var126+Var73+Var125+Var218_dummy_cJvF+
#                   Var211_dummy_Mtgm+Var140,
#                 data = train, family = binomial)

lrAppRFRe <- glm(appetency~Var126+Var125+Var81,
                 data = train, family = binomial)


summary(lrAppRFRe)

#Test for better fit between nested models:
anova(lrAppRFRe,lrAppRF,test="Chisq")

par(mfrow=c(1,1))
fit.Rf <- lrAppRF$fitted
hist(fit.Rf)

#################################################
# Chi-square goodness of fit test for RandomForest Variables
#################################################
# Calculate residuals across all individuals
# r.Rf <- (df_mat.frame$appetency - fit.Rf)/(sqrt(fit.Rf*(1-fit.Rf)))
r.Rf <- (train$appetency - fit.Rf)/(sqrt(fit.Rf*(1-fit.Rf)))
# Sum of squares of these residuals follows a chi-square
sum(r.Rf^2)
# [1] 56239.05
# Sum of squares of these residuals follows a chi-square
# 1- pchisq(66473.61, df=49971)
1- pchisq(56239.05, df=34971)

# ----

# Create train and test data set for data frame

# set.seed(123)
# smp_size <- floor(0.75 * nrow(df_mat.frame))
# train_ind <- sample(seq_len(nrow(df_mat.frame)), size = smp_size)
# # making a "tiny" data set so I cn quickly test r markdown and graphical paramters
# # this will be removed in the submitted version
# tiny_ind <- sample(seq_len(nrow(df_mat.frame)), size = floor(0.01 * nrow(df)))
# # split the data
# train.frame <- df_mat.frame[train_ind, ]
# test.frame <- df_mat.frame[-train_ind, ]
# tiny.frame <- df_mat.frame[tiny_ind, ]

############################
#LOGISTIC MODELS
############################


#1. Logistic regression with decision tree selected variables
# lrApp.tree <- glm(appetency~Var112+Var113+Var126+Var140+
#                    Var153+Var21+Var216_dummy_beK4AFX+Var218_dummy_cJvF+Var24+
#                    Var38+Var6+Var76+Var81,data = train, family = binomial)

# lrApp.tree <- glm(appetency~Var126+Var218+Var6+Var204+
#                    Var153+Var81+Var38+Var113+
#                   Var194+Var201+Var189+Var73+
#                    Var125+Var225+Var76+Var119+
#                    Var123+Var133+Var83+Var112,data = train, family = binomial)

# lrfitDT <-glm(appetency~Var112+Var113+Var126+Var140+Var153+Var21+
#                Var216_dummy_beK4AFX+Var218_dummy_cJvF+Var24+Var38+
#                Var6+Var76+Var81,data=train,family = binomial)

lrfitDT <- glm(appetency~Var126+Var218+Var6+Var204+
                    Var153+Var81+Var38+Var113+Var194+Var201+Var189+Var73+
                   Var125+Var225+Var76+Var119+ Var123+Var133+Var83+Var112,
                  data = train, family = binomial)

summary(lrfitDT)

#Checking prediction quality on training
Plogit <- predict(lrfitDT,newdata=train,type = "response")
p.app <- round(Plogit)

require(e1071)
require(caret)
confusionMatrix(p.app,train$appetency)

#Checking prediction quality on test
PlogitTest <- predict(lrfitDT,newdata=test,type = "response")
p.AppTest <- round(PlogitTest)
confusionMatrix(p.AppTest,test$appetency)

# Confusion Matrix and Statistics

# Reference
#     0    1
# 0 7356  141
# 1    2    1

# Accuracy : 0.9809          
# 95% CI : (0.9776, 0.9839)
# No Information Rate : 0.9811          
# P-Value [Acc > NIR] : 0.5558          

# Kappa : 0.013           
 # Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.999728        
# Specificity : 0.007042        
# Pos Pred Value : 0.981192        
# Neg Pred Value : 0.333333        
# Prevalence : 0.981067        
# Detection Rate : 0.980800        
# Detection Prevalence : 0.999600        
# Balanced Accuracy : 0.503385        

# 'Positive' Class : 0     

#How good is the Logistic model in-sample (AUC)
DT.scores <- prediction(Plogit,train$appetency)
plot(performance(DT.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
DT.auc <- performance(DT.scores,'auc')
DT.auc

#How good is the Logistic model out-sample (AUC)
DT.scores.test <- prediction(PlogitTest,test$appetency)
#ROC plot for logistic regression
plot(performance(DT.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
DT.auc.test <- performance(DT.scores.test,'auc')
DT.auc.test
# Slot "y.values":
# [1] 0.8256769

#2. Logistic regression with LASSO selected variables
# lrfitLASSO <-glm(appetency~Var28+Var34+Var38+Var44+Var58+Var64+Var67+Var75+
#                   Var81+Var84+Var95+Var124+Var125+
#                   Var126+Var140+Var144+Var152+Var162+Var171+Var177+
#                   Var181+Var126_missing+Var194_dummy_SEuy+
#                   Var197_dummy_0Xwj+Var197_dummy_487l+Var197_dummy_TyGl+
#                   Var197_dummy_z32l+Var204_dummy_YULl+Var204_dummy_15m3+
#                   Var204_dummy_4N0K+Var204_dummy_m_h1+Var205_dummy_VpdQ+
#                   Var205_dummy_sJzTlal+Var206_dummy_zm5i+Var206_dummy_43pnToF+
#                   Var208_dummy_kIsH+Var210_dummy_uKAI+Var210_dummy_other+
#                   Var211_dummy_L84s+Var211_dummy_Mtgm+
#                   Var212_dummy_NhsEn4L+Var216_dummy_other+
#                   Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+
#                   Var218_dummy_cJvF+Var218_dummy_UYBR+Var219_dummy_OFWH+
#                   Var223_dummy_M_8D+Var226_dummy_xb3V+Var226_dummy_fKCe+
#                   Var226_dummy_Aoh3+Var226_dummy_uWr3+Var226_dummy_7P5s+
#                   Var228_dummy_R4y5gQQWY8OodqDV,data=train,family = binomial)

lrfitLASSO <- glm(appetency~Var28+Var34+Var38+Var44+Var64+Var67+
                    Var81+Var84+Var95+Var124+Var125+
                    Var126+Var140+Var144+Var152+Var153+Var162+Var171+Var177+
                    Var181+Var197+Var203+Var204+Var205+
                    Var206+Var208+Var210+Var212+Var216+Var217+
                    Var218+Var226,data = train, family = binomial)

summary(lrfitLASSO)

#Checking prediction quality on training
logitLASSO.train <- predict(lrfitLASSO,newdata =train,type = "response")
pLASSO.app <- round(logitLASSO.train)

require(e1071)
require(caret)
confusionMatrix(pLASSO.app,train$appetency)

# Reference
# Prediction     0     1
#         0   34390   605
#         1      2     3

# Accuracy : 0.9827         
# 95% CI : (0.9812, 0.984)
# No Information Rate : 0.9826         
# P-Value [Acc > NIR] : 0.4945         

# Kappa : 0.0095         
# Mcnemar's Test P-Value : <2e-16         

# Sensitivity : 0.999942       
# Specificity : 0.004934       
# Pos Pred Value : 0.982712       
# Neg Pred Value : 0.600000       
# Prevalence : 0.982629       
# Detection Rate : 0.982571       
# Detection Prevalence : 0.999857       
# Balanced Accuracy : 0.502438       

# 'Positive' Class : 0 


#Checking prediction quality on test
logitLASSO.test <- predict(lrfitLASSO,newdata=test,type = "response")
p.AppTest <- round(logitLASSO.test)
confusionMatrix(p.AppTest,test$appetency)

# sum(p.AppTest)
# [1] 0

#How good is the Logistic model in-sample:
LASSO.scores <- prediction(logitLASSO.train,train$appetency)
plot(performance(LASSO.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
LASSO.auc <- performance(LASSO.scores,'auc')
LASSO.auc
# [1] 0.8241394

#How good is the Logistic model out-sample:
LASSO.scores.test <- prediction(logitLASSO.test,test$appetency)
#ROC plot for logistic regression
plot(performance(LASSO.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
LASSO.auc.test <- performance(LASSO.scores.test,'auc')
LASSO.auc.test
# [1] 0.8208571

#3. Logistic regression with Random Forest selected variables
lrfitRf <- glm(appetency~Var126+Var6+Var81+Var113+Var119+Var28+Var25+
                 Var85++Var22+Var73+Var153+Var83+
                 Var133+Var123+Var109+Var160+Var125+Var21+ Var134+Var74+Var112+Var38+Var163+Var76+Var140,
               data = train, family = binomial)

summary(lrfitRf)

#Checking prediction quality on training
logitRf.train <- predict(lrfitRf,newdata =train,type = "response")
pRf.app <- round(logitRf.train)
confusionMatrix(pRf.app,train$appetency)

# Confusion Matrix and Statistics

# Reference
# Prediction     0     1
#            0 34391   607
#            1     1     1

# Accuracy : 0.9826         
# 95% CI : (0.9812, 0.984)
# No Information Rate : 0.9826         
# P-Value [Acc > NIR] : 0.5108         

# Kappa : 0.0032         
# Mcnemar's Test P-Value : <2e-16         

# Sensitivity : 0.999971       
# Specificity : 0.001645       
# Pos Pred Value : 0.982656       
# Neg Pred Value : 0.500000       
# Prevalence : 0.982629       
# Detection Rate : 0.982600       
# Detection Prevalence : 0.999943       
# Balanced Accuracy : 0.500808       

# 'Positive' Class : 0 


#Checking prediction quality on test
logitRf.test <- predict(lrfitRf,newdata=test,type = "response")
p.AppTest <- round(logitRf.test)
confusionMatrix(p.AppTest,test$appetency)


# Confusion Matrix and Statistics

# Reference
# Prediction    0    1
#          0 7356  141
#          1    2    1

# Accuracy : 0.9809          
# 95% CI : (0.9776, 0.9839)
# No Information Rate : 0.9811          
# P-Value [Acc > NIR] : 0.5558          

# Kappa : 0.013           
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.999728        
# Specificity : 0.007042        
# Pos Pred Value : 0.981192        
# Neg Pred Value : 0.333333        
# Prevalence : 0.981067        
# Detection Rate : 0.980800        
# Detection Prevalence : 0.999600        
# Balanced Accuracy : 0.503385        

# 'Positive' Class : 0  



#How good is the Logistic model in-sample:
Rf.scores <- prediction(logitRf.train,train$appetency)
plot(performance(Rf.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
Rf.auc <- performance(Rf.scores,'auc')
Rf.auc
# [1] 0.7690781


#How good is the Logistic model out-sample:
Rf.scores.test <- prediction(logitRf.test,test$appetency)
#ROC plot for logistic regression
plot(performance(Rf.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
Rf.auc.test <- performance(Rf.scores.test,'auc')
Rf.auc.test
# [1] 0.7588043


#plot multiple ROC curves
?plot.performance
library(ROCR)

predapp <- prediction(logitRf.test,test$appetency)
perfapp <- performance(predapp, "tpr", "fpr")
plot(perfapp)
abline(0,1,lty=8,col='grey')

#4. Logistic regression with FSelector Random Forest selected variables
class(train$appetency)
train$appetency <- as.factor(train$appetency)
# m1 <- glm(appetency~Var28+Var34+Var38+Var44+Var58+Var64+Var67+Var75+
#            Var81+Var84+Var95+Var124+Var125+
#            Var126+Var140+Var144+Var152+Var162+Var171+Var177+
#            Var181+Var126_missing+Var194_dummy_SEuy+
#            Var197_dummy_0Xwj+Var197_dummy_487l+Var197_dummy_TyGl+
#            Var197_dummy_z32l+Var204_dummy_YULl+Var204_dummy_15m3+
#            Var204_dummy_4N0K+Var204_dummy_m_h1+Var205_dummy_VpdQ+
#            Var205_dummy_sJzTlal+Var206_dummy_zm5i+Var206_dummy_43pnToF+
#            Var208_dummy_kIsH+Var210_dummy_uKAI+Var210_dummy_other+
#            Var211_dummy_L84s+
#            Var212_dummy_NhsEn4L+Var216_dummy_other+
#            Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+
#            Var218_dummy_cJvF+Var218_dummy_UYBR+Var219_dummy_OFWH+
#            Var223_dummy_M_8D+Var226_dummy_xb3V+Var226_dummy_fKCe+
#            Var226_dummy_Aoh3+Var226_dummy_uWr3+Var226_dummy_7P5s+
#            Var228_dummy_R4y5gQQWY8OodqDV,data=train, family = binomial)

m1 <- glm(appetency~Var28+Var34+Var38+Var44+Var64+Var67+
                    Var81+Var84+Var95+Var124+Var125+
                    Var126+Var140+Var144+Var152+Var153+Var162+Var171+Var177+
                    Var181+Var197+Var203+Var204+Var205+
                    Var206+Var208+Var210+Var212+Var216+Var217+
                    Var218+Var226,data = train, family = binomial)

summary(m1)

#Checking prediction quality on training
logitRf2.train <- predict(m1,newdata=train,type = "response")
# pRf2.app <- round(logitRf2.train)
confusionMatrix(logitRf2.train,train$appetency)

#Checking prediction quality on test
logitRf2.test <- predict(m1,newdata=test,type = "response")
# p.AppTest2 <- round(logitRf2.test)
confusionMatrix(logitRf2.test,test$appetency)

#How good is the Logistic model in-sample:
logitRf2.train <- as.numeric(logitRf2.train)
Rf2.scores <- prediction(logitRf2.train,train$appetency)
plot(performance(Rf2.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
Rf2.auc <- performance(Rf2.scores,'auc')
Rf2.auc
# [1] 0.8241394

#How good is the Logistic model out-sample:
Rf2.scores.test <- prediction(logitRf2.test,test$appetency)
#ROC plot for logistic regression
plot(performance(Rf2.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
Rf2.auc.test <- performance(Rf2.scores.test,'auc')
Rf2.auc.test
# [1] 0.8208571

#plot multiple ROC curves
?plot.performance
library(ROCR)

predapp2 <- prediction(logitRf2.test,test$appetency)
perfapp2 <- performance(predapp2, "tpr", "fpr")
plot(perfapp2)
abline(0,1,lty=8,col='grey')

# make logistic regression predictions
app_lreg_fitLASSO_pred <- predict(lrfitLASSO, test,
                             type = 'response')

app_ens_lreg_fitLASSO_pred <- predict(lrfitLASSO, ensemble_test,
                                 type = 'response')


# save the output
save(list = c('lrfitLASSO', 'app_lreg_fitLASSO_pred',
              'app_ens_lreg_fitLASSO_pred'),
     file = 'R/data/models/app_lreg_fitLASSO.RData')