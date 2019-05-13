import numpy as np
import pandas as pd

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

class SalesForecaster:


    def __init__(self, df, freq='7D', periods=3):
        """ Forecast weekly sales on input dataframe df. Initializes
            dataframe and separates into train/test as well as
            generating a holiday dataframe for Prophet consumption.

            Args:
                df (dataframe): Input dataframe to process.
                freq (str): Pandas compliant time frequency abbreviation
                periods (int): How far out to run forecast. Length of
                    time is a multiple of freq.

        """
        self.freq = freq
        self.periods = periods

        # Prepare dataframe
        df.Date = pd.to_datetime(df.Date)
        df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'}, inplace=True)

        # Save last three obs for calculating error
        self.df_train = df[0:-3]
        self.df_test = df[-3:]

        # Get holidays dataframe
        self.df_holidays = pd.DataFrame({
            'holiday': 'holiday',
            'ds': df[df.IsHoliday == True].ds
            })

    def get_results(self, forecast):
        """ Return dataframe with date, actual metric, forecasted metric,
            and squared error

        """
        self.df_results = self.df_test.copy(deep=True)
        self.df_results.rename(columns={'ds': 'Date',
                                        'y': 'Weekly_Sales_Act'},
                               inplace=True)
        # Assumption here being that last 'periods' rows of forecast
        # contain the result we're calculating error against.
        # If the forecast fails we'll handle that
        try:
            self.df_results['Weekly_Sales_FC'] = (
                forecast['yhat'][-self.periods:].copy(deep=True).tolist()
                )
            self.df_results['Error'] = (
                self.df_results['Weekly_Sales_Act'] -
                self.df_results['Weekly_Sales_FC']
                )
            self.df_results['Squared_Error'] = (
                self.df_results['Weekly_Sales_Act'] -
                self.df_results['Weekly_Sales_FC']
                )**2

            self.df_results.reset_index(inplace=True)
        except Exception as e:
            print(e)

        return(self.df_results)

    def get_RMSE(self):
        """ Return a discrete RMSE value from self.df_results if it
            exists

        """
        try:
            self.RMSE = (
                np.sqrt(mean_squared_error(self.df_results['Weekly_Sales_Act'],
                                           self.df_results['Weekly_Sales_FC']))
                )
        except Exception as e:
            print(e, "Are you sure a model was successfully run?")
            self.RMSE = np.NaN

        return self.RMSE

    def get_mean_err_perc(self):
        """ Calculate and return percentage difference between RMSE and
            mean of test set.

            Returns:
                err_percentage(float): fraction of RMSE to mean of test
                    set.

        """
        try:
            mean = np.mean(self.df_results['Weekly_Sales_Act'])
            self.err_percentage = self.RMSE / mean
        except Exception as e:
            print(e)
            self.err_percentage = np.NaN

        return(self.err_percentage)



    def get_test_df(self):
        """ Return test df

            Returns:
                df_test(dataframe): Test set formatted for Prophet
                    consumption.

        """
        return(self.df_test)

    def get_train_df(self):
        """ Return training df

            Returns:
                df_train(dataframe): Training set formatted for Prophet
                    consumption.

        """
        return(self.df_train)

    def get_holiday_df(self):
        """ Return holiday df

            Returns:
                df_holidays(dataframe): Holidays dataset formatted for
                    Prophet consumption.

        """
        return(self.df_holidays)


    def get_forecast(self):
        """ Run Prophet model on df and get the forecast

        """
        model = Prophet(holidays=self.df_holidays)

        try:
            model.fit(self.df_train)
            future = model.make_future_dataframe(periods=self.periods,
                                                 freq=self.freq)
            forecast = model.predict(future)
        except Exception as e:
            print(e)
            forecast = None

        return(forecast)
