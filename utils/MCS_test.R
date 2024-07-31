rm(list=ls())
library(MCS)
setwd('/Users/merlyn/Documents/Projects/MGRU/MGRU_v2')

# target : [out_sample_size, stock_nums]
# forecast : [out_sample_size, stock_nums]
# loss : [out_sample_size] vector

in_sample_size = 2192
out_sample_size = 242

loss <- function(model_name, forecast_horizon, metric) {
    all_data <- read.csv('data/realized_volatility/log_rv_table.csv', 
        row.names = 1, header = TRUE)
    target <- all_data[seq(in_sample_size+forecast_horizon, in_sample_size+out_sample_size), ]
    filename <- paste(paste(model_name, 'h', forecast_horizon, sep='_')
                      , 'csv', sep='.')
    forecast <- read.csv(paste('results', 'forecasts', filename, sep='/')
                         , header=TRUE, row.names=1)
    if (metric == 'MSE') {
        loss = sqrt((rowMeans((target - forecast)^2)))
    } else if (metric == 'MAE') {
        loss = (rowMeans(abs(target - forecast)))
    }
    return(array(loss))
}

MCS_test <- function(metric, forecast_horizon) {
    cat(metric, forecast_horizon)
    model_list = c('VAR', 'LSTM', 'GRU', 'MLSTMF', 'MGRUF')
    loss_matrix = lapply(model_list, loss, forecast_horizon=forecast_horizon,
                         metric=metric)
    loss_matrix <- do.call(cbind, loss_matrix)
    test_results <- MCSprocedure(as.matrix(loss_matrix), alpha = 0.2)
    
    return(test_results)
}

test_results <- MCS_test('MSE', 1)
test_results <- MCS_test('MAE', 1)
test_results <- MCS_test('MSE', 5)
test_results <- MCS_test('MAE', 5)
test_results <- MCS_test('MSE', 22)
test_results <- MCS_test('MAE', 22)
