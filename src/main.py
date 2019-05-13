import multiprocessing as mp
import pandas as pd
import SalesForecaster as sf

from functools import partial

OUTPUT_COLS = ['Store', 'Dept', 'RMSE', 'Mean_Error_Perc']

def get_store_dept_dict(df):
    """ Get unique store-dept dict from dataframe so we can
        differentiate time series within complete dataframe

        Args:
            df(dataframe): Input dataframe that contains sales data for
                variety of Store/Dept combinations.
        Returns:
            store_dept(list of dicts): A list of Store/Dept dicts.

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

        Returns:

    """
    store = store_dept['Store']
    dept = store_dept['Dept']
    df_singleton = df[(df.Store==store) & (df.Dept==dept)]

    salesforecast = sf.SalesForecaster(df_singleton, freq='7D',
                                       periods=3)
    forecast = salesforecast.get_forecast()
    results = salesforecast.get_results(forecast)
    RMSE = salesforecast.get_RMSE()
    mean_err_perc = salesforecast.get_mean_err_perc()

    df_errors = pd.DataFrame(
        [[store, dept, RMSE, mean_err_perc]], columns=OUTPUT_COLS
        )

    return(df_errors)

def main():
    df = pd.read_csv('../data/train.csv')
    sd_combs = get_store_dept_dict(df)
    pool = mp.Pool(processes=mp.cpu_count())
    run_model_partial = partial(run_model, df)
    results = pool.map(run_model_partial, sd_combs)
    pool.close()
    pool.join()


    df_forecasts = pd.DataFrame()

    for result in results:
        df_forecasts = df_forecasts.append(result)

    print(df_forecasts)
    df_forecasts.to_csv('../data/output.csv')


if __name__ == main():
    main()
