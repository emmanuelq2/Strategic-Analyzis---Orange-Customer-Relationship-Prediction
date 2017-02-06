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
# ----

# ---- read_data ----
# read in the data to R
# I'm using na.stings = '' to replace blanks with na

for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}
# choose a script to load and transform the data
source('impute_0.r')

#check for factor variables
names(df)
f <- sapply(df, is.factor)
which(f)

truedf <- df[,0:230]

truedf$appetency <- appetency$appetency
truedf$upsell <- upsell$upsell
truedf$churn <- churn$churn
# str(truedf)
# summary(truedf)
# summary(truedf) -> 18 missing columns (8,15,20,31,32,39,42,48,52,55,79,141,167,169,175,185,209,230)
#
data = (truedf[,c(1:212,231:233)])


#Data set containing predictors only
df_matrix <- select(data, -churn, -appetency, -upsell)

for (i in names(df_matrix)){
  if (class(df_matrix[,i]) == 'factor'){
    for(level in unique(df_matrix[,i])){
      df_matrix[sprintf('%s_dummy_%s', i, level)] <- ifelse(df_matrix[,i] == level, 1, 0)
    }
    df_matrix[,i] <- NULL
  } else {
    # scale numeric variables
    # this is important for regularized logistic regression and KNN
    df_matrix[,i] <- scale(df_matrix[,i])
  }
}

df_matrix <- data.matrix(df_matrix)


#Creating separate variables for the different factors
# for (i in names(df_mat)){
#  if (class(df_mat[,i]) == 'factor'){
#    for(level in unique(df_mat[,i])){
#      df_mat[sprintf('%s_dummy_%s', i, level)] <- ifelse(df_mat[,i] == level, 1, 0)
#    }
#    df_mat[,i] <- NULL
#  } else {
    # scale numeric variables
    # this is important for regularized logistic regression and KNN
#    df_mat[,i] <- scale(df_mat[,i])
#  }
# }

#Create input matrix
df_matrix <- data.matrix(df_matrix)
# #Convert to data.frame
df_mat.frame <- data.frame(df_matrix)
df_mat.frame$churn <- df$churn
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
df_mat.frame$churn <- factor(df_mat.frame$churn)

# churnTree <- rpart(churn~.,method="class",data = df_mat.frame,
#                   control=rpart.control(minsplit=10, minbucket=10, cp=0.001))

# 1) root 50000 3672 0 (0.9265600 0.0734400) *
# printcp(churnTree)

# Classification tree:
#  rpart(formula = churn ~ ., data = df_mat.frame, method = "class", 
#        control = rpart.control(minsplit = 10, minbucket = 10, cp = 0.005))

# Variables actually used in tree construction:
#  character(0)

# Root node error: 3672/50000 = 0.07344

# n= 50000 

# CP nsplit rel error xerror xstd
# 1  0      0         1      0    0

churnTree <- rpart(churn~.,method="class",data = df_mat.frame,
                   control=rpart.control(minsplit=10, minbucket=5, cp=0.0001))
churnTree

printcp(churnTree)

# Classification tree:
#  rpart(formula = churn ~ ., data = df_mat.frame, method = "class", 
#        control = rpart.control(minsplit = 10, minbucket = 5, cp = 1e-04))

# Variables actually used in tree construction:
#  [1] Var109 Var112 Var113 Var119 Var123 Var125 Var126 Var13  Var132 Var133 Var134 Var140 Var144 Var146 Var149 Var152 Var153
# [18] Var160 Var163 Var17  Var181 Var188 Var189 Var19  Var192 Var193 Var194 Var195 Var197 Var200 Var203 Var204 Var205 Var206
# [35] Var207 Var208 Var21  Var210 Var211 Var212 Var216 Var217 Var218 Var219 Var22  Var221 Var223 Var225 Var226 Var227 Var228
# [52] Var229 Var24  Var25  Var28  Var3   Var35  Var37  Var38  Var5   Var51  Var53  Var57  Var6   Var61  Var65  Var68  Var7  
# [69] Var72  Var73  Var74  Var76  Var78  Var81  Var83  Var84  Var85  Var9   Var94 

# Root node error: 3672/50000 = 0.07344

# n= 50000 

# CP nsplit rel error xerror     xstd
# 1  0.00070806      0   1.00000 1.0000 0.015885
# 2  0.00068083     22   0.97821 1.0027 0.015905
# 3  0.00063544     29   0.97304 1.0044 0.015917
# 4  0.00058656     35   0.96814 1.0093 0.015952
# 5  0.00054466     60   0.95289 1.0408 0.016180
# 6  0.00049020    113   0.91885 1.0536 0.016271
# 7  0.00047658    133   0.90605 1.0637 0.016342
# 8  0.00045389    149   0.89733 1.0694 0.016382
# 9  0.00042795    156   0.89406 1.0937 0.016551
# 10 0.00040850    176   0.88399 1.1043 0.016624
# 11 0.00038126    225   0.85975 1.1174 0.016713
# 12 0.00036311    251   0.84858 1.1329 0.016818
# 13 0.00034041    257   0.84641 1.1446 0.016897
# 14 0.00029956    289   0.83197 1.1487 0.016924
# 15 0.00027233    323   0.81944 1.2315 0.017465
# 16 0.00024207    475   0.76961 1.2437 0.017543
# 17 0.00023343    510   0.75871 1.2704 0.017711
# 18 0.00022694    532   0.75109 1.2756 0.017744
# 19 0.00021786    539   0.74946 1.2914 0.017842
# 20 0.00020425    549   0.74728 1.2938 0.017857
# 21 0.00019452    563   0.74428 1.3115 0.017966
# 22 0.00018155    594   0.73693 1.3273 0.018062
# 23 0.00017021    615   0.73230 1.3404 0.018141
# 24 0.00016642    637   0.72849 1.3453 0.018171
# 25 0.00015562    678   0.71950 1.3475 0.018184
# 26 0.00014265    686   0.71814 1.4161 0.018589
# 27 0.00013617    711   0.71432 1.4175 0.018597
# 28 0.00012104    913   0.67838 1.4262 0.018647
# 29 0.00010893    927   0.67620 1.4360 0.018704
# 30 0.00010000    932   0.67565 1.4390 0.018721

churnTree <- rpart(churn~.,method="class",data = df_mat.frame,
                   control=rpart.control(minsplit=10, minbucket=5, cp=0.0005))
churnTree
printcp(churnTree)

# Classification tree:
#  rpart(formula = churn ~ ., data = df_mat.frame, method = "class", 
#        control = rpart.control(minsplit = 10, minbucket = 5, cp = 5e-04))

# Variables actually used in tree construction:
#  [1] Var112 Var113 Var119 Var123 Var125 Var126 Var13  Var133 Var134 Var140 Var144 Var149 Var153 Var160 Var163 Var181 Var189
# [18] Var192 Var197 Var204 Var205 Var206 Var207 Var21  Var210 Var212 Var216 Var217 Var218 Var223 Var225 Var226 Var229 Var25 
# [35] Var28  Var38  Var53  Var57  Var73  Var74  Var76  Var81  Var9   Var94 

# Root node error: 3672/50000 = 0.07344

# n= 50000 

# CP nsplit rel error xerror     xstd
# 1 0.00070806      0   1.00000 1.0000 0.015885
# 2 0.00068083     22   0.97821 1.0125 0.015976
# 3 0.00063544     29   0.97304 1.0128 0.015978
# 4 0.00058656     35   0.96814 1.0136 0.015984
# 5 0.00054466     60   0.95289 1.0400 0.016174
# 6 0.00050000    113   0.91885 1.0542 0.016275

churnTree <- rpart(churn~.,method="class",data = df_mat.frame,
                   control=rpart.control(minsplit=10, minbucket=5, cp=0.0005, maxdepth = 6))

printcp(churnTree)

# Classification tree:
#  rpart(formula = churn ~ ., data = df_mat.frame, method = "class", 
#        control = rpart.control(minsplit = 10, minbucket = 5, cp = 5e-04, maxdepth = 6))

# Variables actually used in tree construction:
#   [1] Var113 Var126 Var160 Var217 Var225 Var226 Var229 Var73 

# Root node error: 3672/50000 = 0.07344

# n= 50000 

# CP nsplit rel error  xerror     xstd
# 1 0.00070806      0   1.00000 1.00000 0.015885
# 2 0.00068083      5   0.99646 1.00054 0.015889
# 3 0.00054466      8   0.99428 1.00054 0.015889
# 4 0.00050000      9   0.99374 0.99973 0.015883


par(mar=c(1,1,1,1))
plot(churnTree, uniform=TRUE)
text(churnTree, all=TRUE,cex=0.75, splits=TRUE, use.n=TRUE, xpd=TRUE)
?predict
p <- predict(churnTree,newdata=df_mat.frame,type="class")
table(actual=df_mat.frame$churn,predicted=p)

#        predicted
#  actual     0     1
#        0 46260    68
#        1  3306   366
par(mar=c(1,1,1,1))
par(mar = c(5, 10, 4, 2) + 0.1)
barplot(churnTree$variable.importance,horiz=T,las=1, cex.names = 0.75)
par(mar = c(5, 4, 4, 2) + 0.1)


#GOF logistic regression using variables from Tree selection
# lrchurn.tree <- glm(churn~Var126+Var126_missing+Var217_dummy_other+
#                      Var221_dummy_zCkv+Var229_dummy_missing+Var28+
#                      Var65+Var73,data = train, family = binomial)

lrchurn.tree <- glm(churn~Var112+Var113+Var119+Var123+Var125+Var126+Var13+Var133+Var134+ 
                      Var140+Var144+Var149+Var153+Var160+Var163+Var181+Var189+
                      Var192+Var197+Var204+Var205+Var206+Var207+Var21+Var210+
                      Var212+Var216+Var217+Var218+Var223+Var225+Var226+Var229+Var25+
                      Var28+Var38+Var53+Var57+Var73+Var74+Var76+Var81+Var9+Var94, 
                      data = train, family = binomial)

summary(lrchurn.tree)

#Refitted with statitically significant variables
lrchurn.treeRe <- glm(churn~Var113+Var126+Var181+Var206+Var207+Var212+Var216+Var217+Var218+
                        Var223+Var28+Var229+Var73+Var74+Var81,
                        data = train, family = binomial)
summary(lrchurn.treeRe)

anova(lrchurn.treeRe,lrchurn.tree,test="Chisq")


# par(mfrow=c(1,1))
par(mar=c(1,1,1,1))
fit <- lrApp.tree$fitted
hist(fit)
par(mfrow=c(1,1))

# #################################################
# # Chi-square goodness of fit test
# #################################################
# # Calculate residuals across all individuals
# r.tree <- (df_mat.frame$appetency - fit)/(sqrt(fit*(1-fit)))
# # Sum of squares of these residuals follows a chi-square
# sum(r.tree^2)
# #Calculate the p-value from the test
# 1- pchisq(65795.97, df=49996)

#Exploratory logistic Regression with LASSO - Udy's code changes
# ---- LASSO_appetency ----
library(glmnet)
# regularized logistic regression with LASSO

train.glm <- as.matrix(train[-213])
dim(train.glm)
churn.LASSO <- glmnet(train.glm,train$churn,alpha=1, family = "binomial")

plot(churn.LASSO,xvar="lambda",label=TRUE)
grid()
# Show number of selected variables
churn.LASSO
# 84 variables

#Use cross validation to select the best lambda (minlambda) and lambda.1se
#Cross Validated LASSO
churnCVlasso <- cv.glmnet(train.glm,train$churn)
#plot(fit.lasso,xvar="dev",label=TRUE)
plot(churnCVlasso)
coef(churnCVlasso)

# coef(churnCVlasso, s="lambda.1se") # Gives coefficients lambda.1se
# best_lambda <- churnCVlasso$lambda.1se
# best_lambda

#GOF logistic regression models LASSO Variables selected
names(train)
lrchurnLASSO <- glm(churn~Var7+Var73+Var113+Var126,data = train, family = binomial)

summary(lrchurnLASSO)
par(mfrow=c(2,2))
plot(lrchurnLASSO)
# par(mfrow=c(1,1))

#Refitted after dropping variables that could not be estimated
lrchurnLASSO1 <- glm(churn~Var7+Var73+Var113+Var126+Var22_missing+
                       Var28_missing+Var126_missing+
                       Var205_dummy_sJzTlal+Var206_dummy_IYzP+Var210_dummy_g5HH+
                       Var212_dummy_NhsEn4L+Var217_dummy_other+Var218_dummy_cJvF+
                       Var229_dummy_missing,data = train,
                     family = binomial)


summary(lrchurnLASSO1)

#Refitted with significant variables
# lrchurnLASSO2 <- glm(churn~Var7+Var73+Var113+Var126+Var28_missing+Var126_missing+
#                       Var205_dummy_sJzTlal+Var210_dummy_g5HH+
#                       Var212_dummy_NhsEn4L+Var217_dummy_other+Var218_dummy_cJvF+
#                       Var229_dummy_missing,data = train,
#                     family = binomial)

# lrchurnLASSO2 same as lrchurnLASSO
lrchurnLASSO2 <- glm(churn~Var7+Var73+Var113+Var126,data = train, family = binomial)
summary(lrchurnLASSO2)


#Test for better fit between nested models: chi-sq
anova(lrchurnLASSO2,lrchurnLASSO1,test="Chisq")
#Test for better fit between nested models: AIC
AIC(lrchurnLASSO1,lrchurnLASSO2)


par(mfrow=c(1,1))
fit.LASSO.churn <- lrchurnLASSO2$fitted
hist(fit.LASSO.churn)
par(mfrow=c(2,2))
plot(lrchurnLASSO2)
par(mfrow=c(1,1))

## Random Forest
?randomForest
# ---- rf_churn ----
library(randomForest)
set.seed(101)
churnRf <- randomForest(churn~.,data = train,importance = TRUE)
#  The response has five or fewer unique values.  Are you sure you want to do regression?
churn.important <- importance(churnRf, type = 1, scale = FALSE)
varImp(churnRf)
# ----

# ---- plot_rf_churn ----
varImpPlot(churnRf, type = 1, main = 'Variable Importance churn')
varImpPlot(churnRf, type = 2, main = 'Variable Importance churn')

# write the variable importance to a file that can be read into excel
fo <- file("rf.txt", "w")
imp <- importance(churnRf)
write.table(imp, fo, sep="\t")
flush(fo)
close(fo)


#GOF logistic regression using variables from Tree selection
lrchurnRF <- glm(churn~Var113+Var73+Var126+Var6+Var119+Var153+Var25+Var160+
                   Var133+Var81+Var112+Var13+Var125+Var109+Var38+
                   Var21+Var24+Var123+Var83+Var28+
                   Var22+Var144+Var85+Var140+Var74+Var134+Var126+
                   Var76,data = train, family = binomial)

summary(lrchurnRF)
par(mfrow=c(2,2))
plot(lrchurnRF)
par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrchurnRF1 <- glm(churn~Var113+Var73+Var126+Var112+Var81+Var25+
                    Var28+Var74+Var123,
                  data = train, family = binomial)

summary(lrchurnRF1)

#Test for better fit between nested models:
anova(lrchurnRF1,lrchurnRF,test="Chisq")
AIC(lrchurnRF1,lrchurnRF)



#Decision Tree Variables:
# lrfitchurnDT <-glm(churn~Var126+Var126_missing+Var217_dummy_other+
#                     Var221_dummy_zCkv+Var229_dummy_missing+Var28+Var65+
#                     Var73,data=train,family = binomial)

lrfitchurnDT <-glm(churn~Var113+Var126+Var181+Var206+Var207+Var212+Var216+Var217+Var218+
                    Var223+Var28+Var229+Var73+Var74+Var81,data=train,family = binomial)

summary(lrfitchurnDT)

#Checking prediction quality on training
Pchurn <- predict(lrfitchurnDT,newdata =train,type = "response")
p.churn <- round(Pchurn)

require(e1071)
require(caret)
# confusionMatrix(p.churn,train$churn)

# Confusion Matrix and Statistics

# Reference
# Prediction     0     1
#         0  32454  2533
#         1     8     5

# Accuracy : 0.9274          
# 95% CI : (0.9246, 0.9301)
# No Information Rate : 0.9275          
# P-Value [Acc > NIR] : 0.5299          

# Kappa : 0.0032          
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.99975         
# Specificity : 0.00197         
# Pos Pred Value : 0.92760         
# Neg Pred Value : 0.38462         
# Prevalence : 0.92749         
# Detection Rate : 0.92726         
# Detection Prevalence : 0.99963         
# Balanced Accuracy : 0.50086         

# 'Positive' Class : 0               


#Checking prediction quality on test
PchurnTest <- predict(lrfitchurnDT,newdata=test,type = "response")
p.churnTest <- round(PchurnTest)
confusionMatrix(p.churnTest,test$churn)


# Confusion Matrix and Statistics

# Reference
# Prediction    0    1
#           0 6935  561
#           1    4    0

# Accuracy : 0.9247          
# 95% CI : (0.9185, 0.9305)
# No Information Rate : 0.9252          
# P-Value [Acc > NIR] : 0.5806          

# Kappa : -0.0011         
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.9994          
# Specificity : 0.0000          
# Pos Pred Value : 0.9252          
# Neg Pred Value : 0.0000          
# Prevalence : 0.9252          
# Detection Rate : 0.9247          
# Detection Prevalence : 0.9995          
# Balanced Accuracy : 0.4997          

# 'Positive' Class : 0  

#How good is the Logistic model in-sample (AUC)
DT.churnscores <- prediction(Pchurn,train$churn)
plot(performance(DT.churnscores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
DT.churnauc <- performance(DT.churnscores,'auc')
DT.churnauc
# [1] 0.6539597

#How good is the Logistic model out-sample (AUC)
DT.churnscores.test <- prediction(PchurnTest,test$churn)
#ROC plot for logistic regression
plot(performance(DT.churnscores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
DT.churnauc.test <- performance(DT.churnscores.test,'auc')
DT.churnauc.test
# [1] 0.6507785 

#LASSO
# lrchurnLASSO <-glm(churn~Var7+Var73+Var113+Var126+Var28_missing+
#                     Var126_missing+Var205_dummy_sJzTlal+Var210_dummy_g5HH+
#                     Var212_dummy_NhsEn4L+Var217_dummy_other+Var218_dummy_cJvF+
#                     Var229_dummy_missing,data=train,family = binomial)

lrchurnLASSO <- glm(churn~Var7+Var73+Var113+Var126, data = train, family = binomial)
summary(lrchurnLASSO)

#Checking prediction quality on training
pchurnLASSO <- predict(lrchurnLASSO,newdata =train,type = "response")
pLASSO.churn <- round(pchurnLASSO)

require(e1071)
require(caret)
confusionMatrix(pLASSO.churn,train$churn)

# Confusion Matrix and Statistics

#          Reference
# Prediction     0     1
#          0 32456  2535
#          1     6     3

# Accuracy : 0.9274          
# 95% CI : (0.9246, 0.9301)
# No Information Rate : 0.9275          
# P-Value [Acc > NIR] : 0.5299          

# Kappa : 0.0018          
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.999815        
# Specificity : 0.001182        
# Pos Pred Value : 0.927553        
# Neg Pred Value : 0.333333        
# Prevalence : 0.927486        
# Detection Rate : 0.927314        
# Detection Prevalence : 0.999743        
# Balanced Accuracy : 0.500499        

# 'Positive' Class : 0



#Checking prediction quality on test
churnLASSO.test <- predict(lrchurnLASSO,newdata=test,type = "response")
p.churnTest <- round(churnLASSO.test)
confusionMatrix(p.churnTest,test$churn)


#          Reference
# Prediction    0    1
#          0 6935  561
#          1    4    0

# Accuracy : 0.9247          
# 95% CI : (0.9185, 0.9305)
# No Information Rate : 0.9252          
# P-Value [Acc > NIR] : 0.5806          

# Kappa : -0.0011         
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.9994          
# Specificity : 0.0000          
# Pos Pred Value : 0.9252          
# Neg Pred Value : 0.0000          
# Prevalence : 0.9252          
# Detection Rate : 0.9247          
# Detection Prevalence : 0.9995          
# Balanced Accuracy : 0.4997          

# 'Positive' #Class : 0    



#How good is the Logistic model in-sample:
LASSO.churnscores <- prediction(pchurnLASSO,train$churn)
plot(performance(LASSO.churnscores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
LASSO.churnauc <- performance(LASSO.churnscores,'auc')
LASSO.churnauc
# [1] 0.6220781

#How good is the Logistic model out-sample:
LASSO.churnscores.test <- prediction(churnLASSO.test,test$churn)
#ROC plot for logistic regression
plot(performance(LASSO.churnscores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
LASSO.churnauc.test <- performance(LASSO.churnscores.test,'auc')
LASSO.churnauc.test
# [1] 0.6154169

#Random Forest
# lrchurnRf <-glm(churn~Var113+Var73+Var126+Var6+Var81+Var210_dummy_g5HH+
#                  Var28+Var74+Var126_missing+Var211_dummy_L84s,data=train,
#                family = binomial)

lrchurnRf <- glm(churn~Var113+Var73+Var126+Var112+Var81+Var25+
                    Var28+Var74+Var123, data = train, family = binomial)

summary(lrchurnRf)

#Checking prediction quality on training
churnRf.train <- predict(lrchurnRf,newdata =train,type = "response")
pchurnRf.app <- round(churnRf.train)
confusionMatrix(pchurnRf.app,train$churn)

# Confusion Matrix and Statistics

# Reference
# Prediction     0     1
#        0    32458  2535
#        1      4     3

# Accuracy : 0.9275          
# 95% CI : (0.9247, 0.9302)
# No Information Rate : 0.9275          
# P-Value [Acc > NIR] : 0.5135          

# Kappa : 0.002           
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.999877        
# Specificity : 0.001182        
# Pos Pred Value : 0.927557        
# Neg Pred Value : 0.428571        
# Prevalence : 0.927486        
# Detection Rate : 0.927371        
# Detection Prevalence : 0.999800        
# Balanced Accuracy : 0.500529        

# 'Positive' Class : 0 


#Checking prediction quality on test
churnRf.test <- predict(lrchurnRf,newdata=test,type = "response")
p.churnTest <- round(churnRf.test)
confusionMatrix(p.churnTest,test$churn)

# Confusion Matrix and Statistics

# Reference
#      Prediction    0    1
#                0 6935  561
#                1    4    0

# Accuracy : 0.9247          
# 95% CI : (0.9185, 0.9305)
# No Information Rate : 0.9252          
# P-Value [Acc > NIR] : 0.5806          

# Kappa : -0.0011         
# Mcnemar's Test P-Value : <2e-16          

# Sensitivity : 0.9994          
# Specificity : 0.0000          
# Pos Pred Value : 0.9252          
# Neg Pred Value : 0.0000          
# Prevalence : 0.9252          
# Detection Rate : 0.9247          
# Detection Prevalence : 0.9995          
# Balanced Accuracy : 0.4997          

# 'Positive' Class : 0 


#How good is the Logistic model in-sample:
churnRf.scores <- prediction(churnRf.train,train$churn)
plot(performance(churnRf.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
Rf.churnauc <- performance(churnRf.scores,'auc')
Rf.churnauc
# [1] 0.6251731

#How good is the Logistic model out-sample:
Rf.churnscores.test <- prediction(churnRf.test,test$churn)
#ROC plot for logistic regression
plot(performance(Rf.churnscores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
Rf.churnauc.test <- performance(Rf.churnscores.test,'auc')
Rf.churnauc.test
# [1] 0.626054
#plot ROC curves

?plot.performance
library(ROCR)

predchurn <- prediction(churnLASSO.test,test$churn)

perfchurn <- performance(predchurn, "tpr", "fpr")

plot(perfchurn, main="ROC of churn")

abline(0,1,lty=8,col='blue')


# make logistic regression predictions
churn_lreg_fitLASSO_predictions <- predict(lrchurnLASSO, test,
                                      type = 'response')

churn_ens_lreg_fitLASSO_predictions <- predict(lrchurnLASSO, ensemble_test,
                                          type = 'response')

# churn_svm_predictions <- predict(lrfit, df_mat[-train_ind,],
#                                       type = 'response')


# save the output
save(list = c('lrchurnLASSO', 'churn_lreg_fitLASSO_predictions',
              'churn_ens_lreg_fitLASSO_predictions'),
     file = 'models/churn_lreg_fitLASSO.RData')
