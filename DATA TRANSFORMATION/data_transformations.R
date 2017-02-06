df <- read.csv('orange_small_train.data', header = TRUE,
               sep = '\t', na.strings = '')

# get the response variables
churn <- read.csv('orange_small_train_churn.labels', header = FALSE)
appetency <- read.csv('orange_small_train_appetency.labels',
                       header = FALSE)
upsell <- read.csv('orange_small_train_upselling.labels',
                    header = FALSE)

# change -1 to 0
churn[churn$V1 < 0,] <- 0
appetency[appetency$V1 < 0,] <- 0
upsell[upsell$V1 < 0,] <- 0

# add response variables to the data
df$churn <- churn$V1
df$appetency <- appetency$V1
df$upsell <- upsell$V1

# this portion of the code should be copied exactly
# in every data transformation script
# that way we will all be using the same training/testing data
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

# create response dataframe
test_response <- test[,c('upsell', 'churn', 'appetency')]
ens_response <- ensemble_test[,c('upsell', 'churn', 'appetency')]

# save the test response vectors
save(list = c("test_response", 'ens_response'),
     file = 'data_transformations.RData')