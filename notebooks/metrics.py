import pandas as pd

#Columns
exchange_contract_col = 'exchange-contract'
exchange_col = 'exchange'
contract_col = 'contract'
basis_col = 'basis'
timestamp_col = 'timestamp'
current_btc_price_col = 'current_btc_price'

def calc_correlation(df,btc_price_percent_change):
    percent_change_df = df.copy()
    correlation = percent_change_df.corrwith(btc_price_percent_change)
    return correlation

def mask(df,threshold):
    """
    Masks DataFrame values based on threshold, returns new DataFrame with NaN and sign.

    Args:
    df: DataFrame
    threshold: numeric threshold

    Returns:
    DataFrame with masked values.
    """
    df_mask  = df.where(df.abs() > threshold)
    df_mask = df_mask/df_mask.abs()
    return df_mask
def calc_hit_rate(df,price_change):
    """
    This function calculates the hit rate of a given DataFrame and a price change. The hit rate is defined as the percentage of times that the price change is greater than 0 for each row of the DataFrame.

    Args:
    df: A pandas DataFrame containing the prices
    price_change: A pandas DataFrame or Series containing the price change for each row of the input DataFrame

    Returns:
    A float representing the hit rate, i.e, the percentage of times that the price change is greater than 0 for each row of the input DataFrame.
    """
    percent_change_df = df.copy()
    hit_rate = percent_change_df.multiply(price_change,axis=0).gt(0).mean()
    return hit_rate
def calc_actual_profit(df,btc_price_percent_change,actual_profit_thresh=0):
    """
    Calculates the actual profit of a given DataFrame using a given threshold and a btc price change.

    Args:
    df: A pandas DataFrame containing the prices
    actual_profit_thresh: A numeric threshold to mask the values between -thresh and thresh. (default 0)
    btc_price_percent_change: A pandas DataFrame or Series containing the btc price change for each row of the input DataFrame

    Returns:
    A float representing the actual profit by simulating trading by multiplying the mask dataframe and btc price change.
    """
    df_copy = df.copy()
    #Create a dataframe that masks the values between -thresh and thresh. 
    multiplier_df = mask(df_copy,threshold=actual_profit_thresh)

    #Multiply the multiplier_df to the btc_price to simulate trading (same sign = profit, diffrent sign = loss)
    trades_df = multiplier_df.multiply(btc_price_percent_change,axis=0)
    #Get the sum of the trades
    return trades_df

def calc_profit_ratio(trades_df):
    """
    Calculates the profit ratio of a given DataFrame of trades.
    The profit ratio is calculated as the ratio of the average gain per winning trade to the average loss per losing trade.

    Args:
    trades_df: A pandas DataFrame containing the trades.

    Returns:
    A float representing the profit ratio.
    """

    #calculates the total gain of the trades
    total_gain = trades_df.mul(trades_df.gt(0)).sum()

    #calculates the total loss of the trades
    total_loss = abs(trades_df.mul(~trades_df.gt(0)).sum())

    #calculates the number of winning trades
    num_winning_trades = trades_df.gt(0).sum()

    #calculates the number of losing trades
    num_losing_trades = trades_df.lt(0).sum()
    
    #calculates the profit ratio as the ratio of the average gain per winning trade to the average loss per losing trade
    
    profit_ratio = (total_gain/num_winning_trades)/(total_loss/num_losing_trades)
    return profit_ratio

def calc_profit_factor(df):
    trades_df = df.copy()

    #calculates the number of winning trades
    num_winning_trades = trades_df.gt(0).sum()

    #calculates the number of losing trades
    num_losing_trades = trades_df.lt(0).sum()


    return num_winning_trades/num_losing_trades

def filter_every_n_min(df,n,timestamp_col):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col],unit='ms').dt.floor('T')
    time_mask = df[timestamp_col].dt.minute % n == (df[timestamp_col][0].minute % n)
    return df[time_mask]
def get_percent_change(df,timestamp_col,exchange_contract_col,target_col):
    #Get the percent change of all the Exchange Contract pairs
    percent_change_df = df.set_index([timestamp_col, exchange_contract_col])[target_col].unstack([exchange_contract_col]).pct_change()
    percent_change_df = percent_change_df.iloc[1:]
    return percent_change_df


def get_metrics(percent_change_df,btc_price_percent_change,actual_profit_thresh,correlation_thresh = 0):

    # btc_price_percent_change = df.set_index([timestamp_col, exchange_contract_col])[current_btc_price_col].unstack([exchange_contract_col]).pct_change()['BinanceBTCUSDT_230331']
    # btc_price_percent_change = btc_price_percent_change.iloc[1:]
    correlation = calc_correlation(percent_change_df,btc_price_percent_change)

    #Apply correlation
    if correlation_thresh is not None:
        #Calculate the correlation using default pandas correlation function  
        masked_correlation = mask(correlation,correlation_thresh)
        #Multiply the mask to the percent_change_df and drop the null columns
        percent_change_df = percent_change_df.multiply(masked_correlation).dropna(axis=1)
    
    #Calculate Hit rate
    hit_rate = calc_hit_rate(percent_change_df,btc_price_percent_change)

    #Calculate actual profit
    trades_df = calc_actual_profit(
        df = percent_change_df,
        btc_price_percent_change=btc_price_percent_change,
        actual_profit_thresh=actual_profit_thresh)
    actual_profit = trades_df.sum()
    profit_ratio = calc_profit_ratio(trades_df)
    profit_factor = calc_profit_factor(trades_df)

    final_df = pd.concat([correlation,hit_rate,actual_profit,profit_ratio,profit_factor],
                    axis=1,
                    keys = ["correlation","hit_rate","actual_profit","profit_ratio","profit_factor"])
    return final_df.T
if __name__ == '__main__':
    df = pd.read_csv('a.csv')
    final_df = get_metrics(df,True)
    final_df.to_csv('result.csv')

