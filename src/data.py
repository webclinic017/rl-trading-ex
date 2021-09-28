import pandas as pd
import pandas_ta as ta  # TODO: Add additional indicators to dataset
from sklearn.preprocessing import StandardScaler


def get_data(train_path, test_path, scaling=False):

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # pandas_ta requires the DataFrame index to be a DatetimeIndex for some indicators
    df_train.set_index(pd.DatetimeIndex(df_train["Date"]), inplace=True)
    df_test.set_index(pd.DatetimeIndex(df_test["Date"]), inplace=True)

    for df in df_train, df_test:
        df.ta.rsi(close="ETF Price", cumulative=True, append=True)
        df.ta.stochrsi(close="ETF Price", cumulative=True, append=True)
        df.ta.mom(close="ETF Price", cumulative=True, append=True)
        df.ta.ema(close="ETF Price", cumulative=True, append=True)
        df.ta.macd(close="ETF Price", cumulative=True, append=True)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(df.shape)

        if scaling:
            for col in df.columns[3:].values:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df_train, df_test


if __name__ == "__main__":
    pass
