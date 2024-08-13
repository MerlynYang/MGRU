# We use the lassovar package to estimate the VAR model with lasso regularization.
rm(list=ls())
# please make sure have installed the require packages
# package 'lassovar' can be found in https://github.com/lcallot/lassovar

library(lassovar)
library(doParallel)
library(foreach)

# ! readers should change the path
setwd('/Users/merlyn/Documents/Projects/MGRU/MGRU')
data <- read.csv('data/realized_volatility/log_rv_table.csv', 
                 row.names = 1, header = TRUE)
stock_colnames <- colnames(data)
index_rownames <- rownames(data)
save_folder <- 'results/forecasts'
stock_num <- 152 # total number of stocks
# [245, 244, 244, 244, 243, 244, 243, 243, 242, 242]
train_size <- 1950 # previous 9 years (8 for training and 1 for validation)
val_size <- 242
out_sample_size <- 242 # last year
rolling_steps <- 22
in_sample_size <- train_size + val_size
data_size <- in_sample_size + out_sample_size

get_forecast <- function(forecast_horizon, max_lag) {
    set.seed(0)
    
    start_index <- 1
    var_model <- NULL
    results <- NULL
    
    all_forecasts <- array(rep(0, (out_sample_size - forecast_horizon + 1) * stock_num),
                           dim=c((out_sample_size - forecast_horizon + 1), stock_num))
    
    while (TRUE) {
        # estimation
        Finished <- FALSE
        end_index = start_index+in_sample_size-1
        in_sample_data <- data[seq(start_index, end_index),]
        var_model <- lassovar(as.matrix(in_sample_data), lags=max_lag, ic='AIC', horizon=forecast_horizon)
        params <- t(var_model$coefficients)
        for (i in seq(1, rolling_steps)) {
            last_few_matrix <- as.matrix(data[seq((end_index+i-1), (end_index-max_lag+i), -1),])
            X_vec <- rbind(c(1), matrix(t(last_few_matrix), ncol=1))
            if ((start_index - 1 + i) > (out_sample_size - forecast_horizon + 1)) {
                Finished <- TRUE
                break
            }
            all_forecasts[start_index - 1 + i,] = (params %*% X_vec)[,1]
        }
        if (Finished) {
            break
        } else {
            start_index <- start_index + rolling_steps
        }
    }
    colnames(all_forecasts) <- stock_colnames
    rownames(all_forecasts) <- index_rownames[seq(in_sample_size+forecast_horizon, in_sample_size+out_sample_size)]
    filename <- paste(paste('VAR', 'h', forecast_horizon, sep = '_'), 'csv', sep='.')
    write.csv(as.data.frame(all_forecasts), file=paste(save_folder, filename, sep='/'), row.names = TRUE)
    return
}

cl <- makeCluster(3)
clusterSetRNGStream(cl, 0) #set seed
registerDoParallel(cl)

foreach(forecast_horizon = c(1, 5, 22), .packages='lassovar') %dopar% {
    get_forecast(forecast_horizon, 5)
    print('finished')
}

stopCluster(cl)
