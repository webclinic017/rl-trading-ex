import pandas as pd
from sklearn.preprocessing import StandardScaler

# import pandas_ta as ta # TODO: Add additional indicators to dataset


def process_data(train_path: str, test_path: str, scaling: bool = False) -> None:

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # pandas_ta requires the DataFrame index to be a DatetimeIndex for some indicators
    df_train.set_index(pd.DatetimeIndex(df_train["Date"]), inplace=True)
    df_test.set_index(pd.DatetimeIndex(df_test["Date"]), inplace=True)

    for df in df_train, df_test:
        df.rsi.log_return(cumulative=True, append=True)  # relative strength index
        df.stochrsi.log_return(
            cumulative=True, append=True
        )  # stochastic relative strength index
        df.stoch.log_return(cumulative=True, append=True)  # stochastic oscillator
        df.sma.log_return(cumulative=True, append=True)  # simple moving average
        df.ema.log_return(cumulative=True, append=True)  # exponential moving average
        df.macd.percent_return(
            cumulative=True, append=True
        )  # moving average convergence divergence

        if scaling:
            for col in df.columns[3:]:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[col])


if __name__ == "__main__":
    pass
