import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# valid stock should satisfy two conditions:
# 1. start date before specific start date
# 2. have more the nan_threshold nans
def filter_col_date(nan_threshold, date):
    rv_table = pd.read_csv('realized_volatility/rv_table.csv', index_col=0)
    date_valid_col = hs300_stocks[hs300_stocks['start_date'] <= date]['code'] # filter column (stock)
    rv_table_valid = rv_table.loc[rv_table.index > date, date_valid_col] # filter date

    na_valid_col = rv_table_valid.columns[rv_table_valid.isna().sum() < nan_threshold]
    rv_table_valid = rv_table_valid[na_valid_col]
    
    return rv_table_valid

# use the median of non-nan values of all stocks at the same date to fill the nan values
def filter_impute_nan(rv_table):
    for date in rv_table.index:
        non_nan_list = rv_table.loc[date, :].dropna().values
        fill_number = np.median(non_nan_list)
        for col in rv_table.columns:
            if np.isnan(rv_table.loc[date, col]) or rv_table.loc[date, col] == 0:
                rv_table.loc[date, col] = fill_number
    return rv_table

def time_plot(rv_table, seed=546):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from statsmodels.graphics.tsaplots import plot_acf
    import matplotlib as mpl

    mpl.rcParams['font.size'] = 16
    
    np.random.seed(seed)
    selected_columns = rv_table.sample(n=4, axis=1).columns
    # these four stocks are randomly selected from the valid stocks
    selected_columns = ['sz.300274', 'sz.002236', 'sh.601800', 'sh.600188']

    fig, axes = plt.subplots(4, 2, figsize=(26, 12))
    axes = axes.ravel()  # flatten axes

    for i, column in enumerate(selected_columns):
        axes[2 * i].plot(rv_table.index, rv_table[column])
        axes[2 * i].set_title(f'Time Plot of Stock {column}')
        axes[2 * i].xaxis.set_major_locator(mdates.YearLocator())  # set x-axis major ticks to the first of every year
        plot_acf(rv_table[column], auto_ylims=True, zero=False, lags=100, ax=axes[2 * i + 1], title=f'ACF Plot of Stock {column}')

    plt.tight_layout()
    plt.savefig('../images/rv_time_plot.pdf')

def cross_correlation(lag):
    data = pd.read_csv('realized_volatility/log_rv_table.csv', index_col=0)
    data_lag = data.shift(lag)
    
    
    data = np.array(data[lag:])
    data_lag = np.array(data_lag[lag:])
    corr_matrix = np.corrcoef(data.T, data_lag.T)[:152, 152:]
    
    columns = data.columns
    hs300_stocks = pd.read_csv('hs300_stocks.csv', index_col=0)
    sectors = hs300_stocks[hs300_stocks['code'].isin(columns)]['sector_code'].values
    unique_sectors = np.unique(sectors)
    sector_dict_num = {sector: np.sum(sectors == sector) for sector in unique_sectors}
    
    ax = sns.heatmap(corr_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
    ax.set(xlabel=None, ylabel=None)
    index = 0
    for key, value in sector_dict_num.items():
        offset_y = 0
        if key == 'InformationTechnology':
            offset_x = -73
        elif key == 'Materials':
            offset_x = -41
        elif key == 'RealEstate':
            offset_x = -46
            offset_y = 0
        elif key == 'TelecommunicationServices':
            offset_x = -69
            offset_y = 3
        elif key == 'Utilities':
            offset_x = -value + 1
            offset_y = -value / 2 - 15
        else:
            offset_x = 6
        if key == 'Utilities':
            rotation = 90
        else:
            rotation = 0
        plt.text(index + value + offset_x, index + value / 2 + offset_y, key, va='center', fontsize=8.5, rotation = rotation)
        plt.hlines(index, index, index + value, colors='black')
        plt.hlines(index + value, index, index + value, colors='black')
        plt.vlines(index, index, index + value, colors='black')
        plt.vlines(index + value, index, index + value, colors='black')
        index += value
    
    plt.savefig('../../images/heatmap.pdf')
    
    return
    
if __name__ == '__main__':
    hs300_stocks = pd.read_csv('hs300_stocks.csv', index_col=0)
    # filter out valid stocks
    # 1. stock start date should before 2014-01-01
    # 2. stock should have less than 100 nan values (non-trading day)
    rv_table_valid = filter_col_date(100, '2014-01-01')
    # impute nan values
    rv_table_valid = filter_impute_nan(rv_table_valid)
    # log transformation
    log_rv_table_valid = np.log(rv_table_valid)
    # plot time series and acf
    time_plot(rv_table_valid)
    # plot lag order cross_correlation matrix
    cross_correlation(1)
    # save the results
    log_rv_table_valid.to_csv('realized_volatility/log_rv_table.csv')