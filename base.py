import datetime
import os

import pandas as pd

import settings


def add_relativetime(df, tm_column, d_str, base_time, parser_func):
    df_convert = pd.to_datetime(df[tm_column].apply(parser_func))
    df_converted = (df_convert - base_time) / pd.Timedelta('1s')
    df.loc[:, 'O_RELATIVETIME'] = df_converted
    return df


def other_data(d_str, format='%Y-%m-%d'):
    d_date = datetime.datetime.strptime(d_str, format).date()
    weather = settings.weather_201710[int(d_str[-2:])]
    workday = (d_date > datetime.date(2017, 10, 8)) & (d_date.isoweekday() < 6)
    weekday = d_date.isoweekday() if d_date > datetime.date(2017, 10, 8) else 8
    return d_date, weather, workday, weekday


class BaseMixin(object):
    predict_df = pd.DataFrame()
    predict_columns = None
    predict_dt = None
    predict_lines = None
    predict_terminals = None

    def get_predict_info(self):
        file_path = os.path.join(settings.PredictPath, settings.PredictFile)
        df = pd.read_csv(file_path)
        self.predict_columns = df.columns
        df.loc[:, 'O_ORDER'] = df.index
        self.predict_dt = sorted(df.O_DATA.drop_duplicates().tolist())
        self.predict_lines = sorted(df.O_LINENO.drop_duplicates().tolist())
        self.predict_terminals = df.O_TERMINALNO.drop_duplicates().tolist()
        self.predict_df = df
