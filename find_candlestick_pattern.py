import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    csvs = glob.glob('data/*.csv')

    # lists to save patterns to
    positive_patterns = []
    negative_patterns = []

    N_PAST_DAYS = 6
    N_FUTURE_DAYS = 3

    for csv in csvs:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        df = df.sort_index()
        # loop through the time series, store the last 5 days worth of candle sticks and find the pattern indicates a buy
        # where we can sell a few days for a profit above a tolerance
        # use the columns for: open, high, low, close, volume
        # use the index for: time

        sell_tolerance = 0.03 # % change to use when selling

        # loop over historical data, find 5 dates in past
        # if we buy on most recent past date, can we sell for a profit on the next day?
        for i in tqdm(range(N_PAST_DAYS, len(df)-N_FUTURE_DAYS)):
            # get the past 5 days
            past_df = df.iloc[i-N_PAST_DAYS:i]
            # get the future 1 days
            future_df = df.iloc[i:i+N_FUTURE_DAYS]
            # get the most recent past date
            past_date = past_df.index[-1]
            # get the open, high, low, close, volume for the most recent past date
            past_open = past_df['open'].iloc[-1] # when we buy
            past_high = past_df['high'].iloc[-1]
            past_low = past_df['low'].iloc[-1]
            past_close = past_df['close'].iloc[-1]
            past_volume = past_df['volume'].iloc[-1]

            # compare future open prices, are any above the sell_tolerance?
            # if so save the pattern for training
            future_open = future_df['open']
            future_high = future_df['high']

            # compute percent change in price compared to past_date
            high_percent_change = (future_high - past_open) / past_open

            # combine future and past data into single df
            combined_df = pd.concat([past_df, future_df])

            # are any of the percent changes above the sell_tolerance?
            if any(high_percent_change > sell_tolerance):
                # save the pattern as a positive pattern
                positive_patterns.append(combined_df)
            else:
                # save the pattern as a negative pattern
                negative_patterns.append(combined_df)
        #break

    # prepare data for training by converting into percent changes
    X = []
    y = []
    for pattern in positive_patterns:
        # get other features and normalize by colum
        c_pattern = pattern[['volume', 'SMA', 'EMA', 'RSI', 'SlowK', 'SlowD', 'CCI']]
        c_pattern = c_pattern[:N_PAST_DAYS-1]
        c_pattern = c_pattern / c_pattern.iloc[-1] - 1
        # only take the open, high, low, close columns
        pattern = pattern[['open', 'high', 'low', 'close']]
        # only deal with past data since last row is future data
        pattern = pattern[:N_PAST_DAYS]
        # convert to percent change relative to buy price on oldest date
        pattern = pattern / pattern.iloc[-1]['open'] - 1
        # flatten into a 1D array and remove features for the day of buy
        f_pattern = pattern.values.flatten()[:-4]
        # add volume features
        f_pattern = np.append(f_pattern, c_pattern.values[:-1])
        # save to X and remove high, low, close values on day of buy
        X.append(f_pattern)
        y.append(1) # 1 for positive pattern
    
    for pattern in negative_patterns:
        # get other features and normalize by colum
        c_pattern = pattern[['volume', 'SMA', 'EMA', 'RSI', 'SlowK', 'SlowD', 'CCI']]
        c_pattern = c_pattern[:N_PAST_DAYS-1]
        c_pattern = c_pattern / c_pattern.iloc[-1] - 1
        # only take the open, high, low, close columns
        pattern = pattern[['open', 'high', 'low', 'close']]
        # only deal with past data since last row is future data
        pattern = pattern[:N_PAST_DAYS]
        # convert to percent change relative to buy price on oldest date
        pattern = pattern / pattern.iloc[-1]['open'] - 1
        # flatten into a 1D array and remove features for the day of buy
        f_pattern = pattern.values.flatten()[:-4]
        # add volume features
        f_pattern = np.append(f_pattern, c_pattern.values[:-1])
        # save to X and remove high, low, close values on day of buy
        X.append(f_pattern)
        y.append(0) # 0 for negative pattern/ do not buy

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # remove rows with nans
    nanmask = np.isnan(X).any(axis=1)
    X = X[~nanmask]
    y = y[~nanmask]

    # remove rows with inf
    infmask = np.isinf(X).any(axis=1)
    X = X[~infmask]
    y = y[~infmask]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # calculate sample weights
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 1] = 1/sum(y_train == 1)
    sample_weights[y_train == 0] = 1/sum(y_train == 0)

    # train a decision tree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # evaluate the model classification tp, tn, fp, fn rates
    y_pred = model.predict(X_test)
    tp = sum((y_pred == 1) & (y_test == 1))/sum(y_test == 1)
    tn = sum((y_pred == 0) & (y_test == 0))/sum(y_test == 0)
    fp = sum((y_pred == 1) & (y_test == 0))/sum(y_test == 0)
    fn = sum((y_pred == 0) & (y_test == 1))/sum(y_test == 1)
    print('tp: {:.2f}, tn: {:.2f}, fp: {:.2f}, fn: {:.2f}'.format(tp, tn, fp, fn))
    # total accuracy
    print('accuracy: {:.2f}'.format((tp+tn)/(tp+tn+fp+fn)))

    # # plot feature importances
    # plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # plt.show()

    # print out the tree
    #print(export_text(model))

    # What if we were to randomly buy and sell?
    pos_rate = len(y_test[y_test == 1])/len(y_test)
    print('positive rate from randomly guessing: {:.2f}'.format(pos_rate))