
import multiprocessing as mp
import pandas as pd
import SalesForecaster as sf

from functools import partial


def get_store_dept_dict(df):
    """ Get unique store-dept dict from dataframe so we can
        differentiate time series within complete dataframe

    """
    df_store_dept = df[['Store', 'Dept']].copy()
    df_store_dept.drop_duplicates(inplace=True)
    store_dept_dict = df_store_dept.to_dict('records')

    return(store_dept_dict)

def run_model(df, store_dept):
    """ Get forecast for single Store-Dept combination

        Args:
            df(dataframe): Dataframe containing data for forecast we
                want to run
            store_dept(tuple): a two integer tuple describing a
                location.
    """
    store = store_dept['Store']
    dept = store_dept['Dept']
    df_singleton = df[(df.Store==store) & (df.Dept==dept)]

    salesforecast = sf.SalesForecaster(df_singleton, freq='7D',
                                       periods=3)
    forecast = salesforecast.get_forecast()
    results = salesforecast.get_results(forecast)
    RMSE = salesforecast.get_RMSE()

    return(RMSE)

def main():
    df = pd.read_csv('../data/train.csv')
    sd_combs = get_store_dept_dict(df)
    print(sd_combs)
    pool = mp.Pool(processes=mp.cpu_count())

    run_model_partial = partial(run_model, df)
    results = pool.map(run_model_partial, sd_combs)

    pool.close()
    pool.join()

    print(results)




    #df_samp = df[(df.Store==1) & (df.Dept==1)]



   # my_test = sf.SalesForecaster(df_samp, freq='7D', periods=3)
   # print(my_test.get_test_df())
   # print(my_test.get_train_df())
   # print(my_test.get_holiday_df())
   # forecast = my_test.get_forecast()
   # df_se = my_test.get_results(forecast)
   # print(df_se)
   # print(my_test.get_RMSE())
#
#
   # print(get_store_dept_dict(df))





if __name__ == main():
    main()
