library(dplyr)
library(ROCR)


make_mat <- function(df_mat){
  # this function turns a dataframe into matrix format
  # it assumes that the response varaibles have not been removed
  
  df_mat <- dplyr::select(df_mat, -churn, -appetency, -upsell)
  
  for (i in names(df_mat)){
    if (class(df_mat[,i]) == 'factor'){
      for(level in unique(df_mat[,i])[2:length( unique(df_mat[,i]))]){
        df_mat[sprintf('%s_dummy_%s', i, level)] <- ifelse(df_mat[,i] == level, 1, 0)
      }
      df_mat[,i] <- NULL
    } else {
      # scale numeric variables
      # this is important for regularized logistic regression and KNN
      df_mat[,i] <- scale(df_mat[,i])
    }
  }
  
  return(data.matrix(df_mat))
  
}


multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  # Multiple plot function
  #
  # ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
  # - cols:   Number of columns in layout
  # - layout: A matrix specifying the layout. If present, 'cols' is ignored.
  #
  # If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
  # then plot 1 will go in the upper left, 2 will go in the upper right, and
  # 3 will go all the way across the bottom.
  #
  # http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


all_interactions <- function(df_mat, var){
  # input:
  #   - df_mat: data matrix
  #   - var: variable in that matrix
  # output:
  #   data matrix with interaction between all variables and var
  
  # this is a very brute force method of finding interaction terms
  
  int_mat <- df_mat * df_mat[,var]
  
  for (i in 1:dim(int_mat)[2]){
    dimnames(int_mat)[[2]][i]<- sprintf('%s_%s_inter', var,
                                        dimnames(int_mat)[[2]][i])
  }
  
  return(cbind(df_mat, int_mat))
}


scale_vec <- function(x){
  # input:
  #   - vector
  # output:
  #   - copy of vector scaled from 0 to 1
  
  (x - min(x))/diff(range(x))
}


make_roc <- function(df, response_vec){
  # transform data frame to be ready to plot roc curves
  
  df_list = list()
  # cycle through algorithms and append them to the dataframe
  for (alg in unique(df$algorithm)){
    pred <- prediction(df[df$algorithm == alg, 'prediction'], response_vec)
    perf <- performance(pred, measure = "tpr", x.measure = "fpr")
    df_list[[alg]] <- data.frame(
      algorithm = rep(alg, length(perf@y.values)),
      TPR = unlist(perf@y.values),
      FPR = unlist(perf@x.values))
  }
  # rbind all the dataframes
  return(do.call("rbind", df_list))
}


make_auc <- function(df, response_vec, in_house){
  # funciton to help me build AUC comparison tables
  auc_table = data.frame(In_House = c(in_house))
  # cycle through algorithms and append them to the dataframe
  for (alg in unique(df$algorithm)){
    pred <- prediction(df[df$algorithm == alg, 'prediction'], response_vec)
    perf <- performance(pred, measure = "auc")
    auc_table[,alg] <- perf@y.values[[1]]
  }
  auc_table <- t(auc_table)
  colnames(auc_table) <- c("AUC")
  return(auc_table)
}


vec_auc <- function(yhat, y){
  # calculate the AUC for a vector and truth vector
  pred <- prediction(yhat, y)
  perf <- performance(pred, measure = "auc")
  return(perf@y.values[[1]])
}