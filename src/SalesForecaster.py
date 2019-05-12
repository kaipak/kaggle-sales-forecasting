import numpy as np
import pandas as pd
from fbprophet import Prophet

class SalesForecaster:


    def __init__(self, df, freq='7D', periods=3):
        """

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
        self.df_results = self.df_test.copy()
        self.df_results.rename(columns={'ds': 'Date',
                                        'y': 'Weekly_Sales_Act'},
                               inplace=True)

        # Assumption here being that last 'periods' rows of forecast
        # contain the result we're calculating error against.
        self.df_results['Weekly_Sales_FC'] = (
            forecast['yhat'][-(self.periods):]
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

        return(self.df_results)

    def get_RMSE(self):
        """ Return a discrete RMSE value from self.df_results if it
            exists

        """
        try:
            RMSE = np.sqrt(np.sum(self.df_results['Squared_Error']) / len(self.df_test))
        except Exception as e:
            print(e, "Are you sure a model was run?")
            RMSE = 0
        return RMSE


    def get_test_df(self):
        """ Helper method to spot check

        """
        return(self.df_test)


    def get_train_df(self):
        """ Helper method to spot check

        """
        return(self.df_train)

    def get_holiday_df(self):
        """ Helper method to spot check

        """
        return(self.df_holidays)


    def get_forecast(self):
        """ Run Prophet model on df and get the forecast

        """
        model = Prophet(holidays=self.df_holidays)
        model.fit(self.df_train)
        future = model.make_future_dataframe(periods=self.periods,
                                             freq=self.freq)
        forecast = model.predict(future)

        return(forecast)




