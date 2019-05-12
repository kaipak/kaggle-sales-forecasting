
import pandas as pd
import SalesForecaster as sf




def main():
    df = pd.read_csv('../data/train.csv')
    df_samp = df[(df.Store==1) & (df.Dept==1)]


    my_test = sf.SalesForecaster(df_samp, freq='7D', periods=3)
    print(my_test.get_test_df())
    print(my_test.get_train_df())
    print(my_test.get_holiday_df())
    forecast = my_test.get_forecast()
    df_se = my_test.get_results(forecast)





if __name__ == main():
    main()
