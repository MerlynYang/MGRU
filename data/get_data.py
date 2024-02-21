import pandas as pd
import numpy as np
import time
import os

# we use baostock to get data
# for more details, please refer to http://baostock.com/baostock/index.php/
import baostock as bs

def get_constituen_stocks():
    lg = bs.login()
    rs = bs.query_hs300_stocks(date='2024-1-6')

    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():
        hs300_stocks.append(rs.get_row_data())
    hs300 = pd.DataFrame(hs300_stocks, columns=rs.fields)
    
    bs.logout()
    hs300.to_csv('hs300_stocks.csv', index=True)
    return hs300

def code_order_id(code):
    if code[:3] == 'sh.':
        return code[3:] + '.XSHG'
    elif code[:3] == 'sz.':
        return code[3:] + '.XSHE'
    else:
        Warning('code is not valid')
        return None

# add abbrev_symbol, sector_code, sector_code_name, industry_code, industry_name
def add_extra_info():
    hs300_stocks = pd.read_csv('hs300_stocks.csv', index_col=0)
    all_stocks = pd.read_csv('overall_description.csv')

    for code in hs300_stocks['code']:
        order_id = code_order_id(code)
        stock_info = all_stocks[all_stocks['order_book_id'] == order_id]
        if len(stock_info) == 0:
            print(f'stock info not found for {code}/{order_id}')
            continue
        hs300_stocks.loc[hs300_stocks['code'] == code, 'abbrev_symbol'] = stock_info['abbrev_symbol'].item()
        hs300_stocks.loc[hs300_stocks['code'] == code, 'sector_code'] = stock_info['sector_code'].item()
        hs300_stocks.loc[hs300_stocks['code'] == code, 'sector_code_name'] = stock_info['sector_code_name'].item()
        hs300_stocks.loc[hs300_stocks['code'] == code, 'industry_code'] = stock_info['industry_code'].item()
        hs300_stocks.loc[hs300_stocks['code'] == code, 'industry_name'] = stock_info['industry_name'].item()
        
    hs300_stocks.to_csv('hs300_stocks.csv', index=True)

def group_by_industry():
    hs300_stocks = pd.read_csv('hs300_stocks.csv', index_col=0)
    grouped = hs300_stocks.sort_values('sector_code').reset_index()
    grouped = grouped.rename(columns={'index' : 'original_index'})
    grouped.to_csv('hs300_stocks.csv', index=True)

def get_highfreq_data_stock(code):
    start_time = time.time()
    rs = bs.query_history_k_data_plus(code,
        "date, time, code, open, high, low, close, volume",
        start_date='2014-01-01', end_date='2023-12-31',
        frequency="5", adjustflag="1")
    print(f'recived data for {code}, time: {(time.time() - start_time):.1f}')
    
    start_time = time.time()
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result.to_csv(f'./5mins_data/{code}.csv', index=False)
    print(f'convert data to dataframe, time: {(time.time() - start_time):.1f}')
    
    return

def get_highfreq_data():
    hs300 = pd.read_csv('hs300_stocks.csv', index_col=0)
    bs.login()

    code_list = hs300['code']
    for i, code in enumerate(code_list):
        print(f'get data for {code} ({i + 1}/300)')
        if os.path.exists(f'./5mins_data/{code}.csv'):
            print(f'{code} already exists, skip')
            continue
        get_highfreq_data_stock(code)

    bs.logout()

# change the time format from 20190101093000000000 to 09:30:00
def reformulate_time(code):
    stock_data = pd.read_csv(f'./5mins_data/{code}.csv')
    stock_data['time'] = pd.to_datetime(stock_data['time'], format='%Y%m%d%H%M%S%f')
    stock_data['time'] = stock_data['time'].dt.strftime('%H:%M')
    stock_data.to_csv(f'./5mins_data/{code}.csv', index=False)

# add start_date and end_date of trading to hs300_stocks.csv
# reformulate the time format of 5 mins data
def add_date_info():
    hs300 = pd.read_csv('hs300_stocks.csv', index_col=0)
    for code in hs300['code']:
        reformulate_time(code)
        stock_data = pd.read_csv(f'./5mins_data/{code}.csv')
        start_date = stock_data['date'].iloc[0]
        end_date = stock_data['date'].iloc[-1]
        hs300.loc[hs300['code'] == code, 'start_date'] = start_date
        hs300.loc[hs300['code'] == code, 'end_date'] = end_date
    hs300.to_csv('hs300_stocks.csv')

# calculate the realized volatility of each day
def cal_rv_stock(code):
    stock_data = pd.read_csv(f'./5mins_data/{code}.csv')
    stock_data['log_return'] = np.log(stock_data['close']) - np.log(stock_data['close'].shift(1))
    date_list = stock_data['date'].unique()
    rv_list = []
    filter_time = stock_data[stock_data['time'] >= '10:00']
    for date in date_list:
        filter1 = filter_time['date'] == date
        log_returns = filter_time[filter1]['log_return'] * 100 # percentage
        rv = np.sum(log_returns ** 2)
        rv_list.append(rv)
    
    return date_list, rv_list

def cal_rv_all():
    hs300 = pd.read_csv('hs300_stocks.csv', index_col=0)
    rv_columns = []
    
    for i, code in enumerate(hs300['code']):
        start_time = time.time()
        date_list, rv_list = cal_rv_stock(code)
        rv_column = pd.DataFrame(rv_list, index=date_list, columns=[code])
        rv_columns.append(rv_column)
        print(f'{code} ({i + 1}/300) calculated, time used: {(time.time() - start_time):.1f}')

    rv_table = pd.concat(rv_columns, axis=1)
    rv_table = rv_table.sort_index()
    rv_table.to_csv('realized_volatility/rv_table.csv')

if __name__=='__main__':
    # get the constituent stocks of CSI 300
    get_constituen_stocks()
    # add abbrev_symbol, sector_code, sector_code_name, industry_code, industry_name to hs300_stocks.csv
    add_extra_info()
    # rearrange the order of the stocks by sector_code
    group_by_industry()
    # download 5-mins data for each stock
    get_highfreq_data()
    # calculate realized volatility of each stock
    cal_rv_all()