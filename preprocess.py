import datetime
import operator
import os

import numpy as np
import pandas as pd
# from reshape import toBePredict
from sklearn.ensemble import IsolationForest

import settings
from base import BaseMixin, other_data


# from sklearn.neighbors import LocalOutlierFactor


def calc_buses(csv_file):
    """
    for test
    :param csv_file:
    :return:
    """
    start_time = time.time()
    # split file path use os.path.split
    df_date = pd.read_csv(csv_file)
    buses_records_count = df_date['O_LINENO'].value_counts()
    csv_path, csv_file_name = os.path.split(csv_file)
    parse_file = 'predict_line_counts.csv'
    buses_records_count.to_csv(parse_file, index_label='O_LINENO',
                               header=['records_number', ])
    print('use time', time.time() - start_time)
    return


def _write(df, file_name, index=False):
    df.to_csv(file_name, index=index)
    return


def concat_df(df, df_new):
    if df.size:
        df = pd.concat([df, df_new])
    else:
        df = df_new
    return df


def first_stop_according_up(x):
    return x == 1 or x == -1


def first_stop_according_nextstationno(x, max_stop):
    return x < max(-0.5 * max_stop, -10)


def _subsection(df):
    size = df.shape[0]
    stop_args = np.array([max(df.O_NEXTSTATIONNO)]) * size
    stop_index = df[
        df.O_NEXTSTATIONNO.diff().apply(first_stop_according_nextstationno,
                                        args=(stop_args))].index.tolist()
    # according up and down
    up_index = df[df.O_UP.diff().apply(first_stop_according_up)].index.tolist()
    time_index = df[df.O_RELATIVETIME.diff() > 3600].index.tolist()
    updown_index = sorted(list(set(up_index + time_index + stop_index)))
    # + [0, size - 1]
    return [0] + updown_index + [size - 1]


def calc_weekday(x):
    if not isinstance(x,str):
        x = str(x)
    d_date = datetime.date(int(x[:4]), int(x[4:6]), int(x[6:]))
    if d_date > datetime.date(2017, 10, 8):
        return d_date.isoweekday()
    else:
        return 8


def process_terminal(df_terminal, keep='first'):
    df_terminal_processed = pd.DataFrame()
    # reset index
    df_terminal.index = pd.Series(range(0, df_terminal.shape[0]))
    # get stop index from O_UP and O_NEXTSTATIONNO
    updown_index = _subsection(df_terminal)
    print('shape:', df_terminal.shape, 'updown_index:', updown_index)
    for it, start in enumerate(updown_index[:-1]):
        end = updown_index[it + 1]
        if operator.eq(start, end):
            end += 1
        df_once_raw = df_terminal[df_terminal.index.isin(range(start, end))]
        df_once = df_once_raw.drop_duplicates(subset=['O_NEXTSTATIONNO'],
                                              keep=keep)
        df_once.loc[:, 'stop_diff'] = df_once['O_NEXTSTATIONNO'].diff()
        df_stop = df_once[
            (df_once['stop_diff'] > 0) | (np.isnan(df_once['stop_diff']))].drop(
            'stop_diff', 1)
        df_stop.loc[:, 'O_USETIME'] = df_stop['O_RELATIVETIME'].diff()
        if it > 0 and start > 0:
            try:
                head_index = df_stop.head(n=1).index.tolist()[0]
            except:
                if df_once.size:
                    import pdb
                    pdb.set_trace()
            head_usetime = df_stop.at[head_index, 'O_RELATIVETIME'] - \
                           df_terminal.at[last_index, 'O_RELATIVETIME']
            df_stop.loc[head_index, 'O_USETIME'] = head_usetime
        try:
            last_index = df_stop.tail(n=1).index.tolist()[0]
        except:
            if df_once.size:
                import pdb
                pdb.set_trace()
        df_terminal_processed = concat_df(df_terminal_processed, df_stop)
    return df_terminal_processed


class PreProcess(BaseMixin):
    def __init__(self):
        self.date_str = None
        self.date_base = None
        self.get_predict_info()

    def _get_date(self, it):
        self.date_str = '201710{}'.format(str(it).zfill(2))
        self.date_base = datetime.datetime.combine(
            datetime.datetime.strptime(self.date_str, '%Y%m%d').date(),
            datetime.time(5, 0, 0))

    def _first_step(self, df_raw, keep='first'):
        """

        :param it:
        :param file_path:
        :param keep: drop duplicates method, default is first
        :return:
        """
        # read csv file by pandas
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

        # df_opt = df_raw.drop_duplicates(subset=settings.duplicate_cols,
        #                                 keep=keep).copy()
        df_opt = df_raw.loc[:, ['O_LINENO', 'O_TERMINALNO', 'O_TIME', 'O_UP',
                                'O_NEXTSTATIONNO']]
        if self.predict_terminals:
            df_opt = df_opt[df_opt['O_TERMINALNO'].isin(self.predict_terminals)]

        def date_parser(time):
            return ' '.join([self.date_str, time])

        df_convert = pd.to_datetime(df_opt['O_TIME'].apply(date_parser))
        df_converted = ((df_convert - self.date_base) / pd.Timedelta('1s'))
        df_opt.loc[:, 'O_RELATIVETIME'] = df_converted
        return df_opt

    def _second_step(self, df):
        terminals = df['O_TERMINALNO'].drop_duplicates().tolist()
        df_processed = pd.DataFrame()
        # terminals = [906184]
        for terminal in terminals:
            print('terminal:', terminal)
            df_terminal = df[df['O_TERMINALNO'] == terminal].sort_values(
                by='O_RELATIVETIME')
            df_terminal_processed = process_terminal(df_terminal)
            df_processed = concat_df(df_processed, df_terminal_processed)
        return df_processed

    def process(self, it, file_path):
        start_time = time.time()
        self._get_date(it)

        df_raw = pd.read_csv('{}/train{}.csv'.format(file_path, self.date_str),
                             dtype=settings.dtype_raw)
        print('read csv, use time', time.time() - start_time)
        df_opt = self._first_step(df_raw)
        print('first step, use time', time.time() - start_time)
        df_processed = self._second_step(df_opt)
        _write(df_processed,
               '{}/processed_t_{}.csv'.format(file_path, self.date_str))
        return


def outlier_detection(df):
    if df.empty:
        return df
    df_detection = df.dropna()
    df_train = df_detection.loc[:, features]
    odf_iforest = IsolationForest(n_estimators=200, random_state=666, n_jobs=-1)
    X_train = np.array(df_train)
    odf_iforest.fit(X_train)
    y_train_predict = odf_iforest.predict(X_train)
    df_detection.loc[:, 'O_OUTLIER_IFOREST'] = y_train_predict
    odf_lof = LocalOutlierFactor(n_neighbors=30)
    # odf_lof.fit(X_train)
    y_train_predict_lof = odf_lof.fit_predict(X_train)
    df_detection.loc[:, 'O_OUTLIER_LOF'] = y_train_predict_lof
    df_detection.loc[:, 'O_BOTH_OUTLIER'] = (y_train_predict_lof == -1) & (
        y_train_predict == -1)
    df_detection.loc[:, 'O_BOTH_INLIER'] = (y_train_predict_lof == 1) & (
        y_train_predict == 1)
    return df_detection[df_detection['O_BOTH_INLIER']]


def _is_workday(it):
    d_date = datetime.date(2017, 10, it)
    return (d_date > datetime.date(2017, 10, 8)) & (d_date.isoweekday() < 6)


def _get_weather(d_str):
    return settings.weather_201710[int(d_str[-2:])]

def calc_weekday(x):
    if not isinstance(x,str):
        x = str(x)
    d_date = datetime.date(int(x[:4]), int(x[4:6]), int(x[6:]))
    if d_date > datetime.date(2017, 10, 8):
        return d_date.isoweekday()
    else:
        return 8

def _calc_workday(x):
    return _is_workday(int(str(x)[-2:]))



def _add_weekday(lineno):
    file_name = os.path.join('fit', 'onestop_{}.csv'.format(lineno))
    df = pd.read_csv(file_name)
    df.loc[:, 'O_WEEKDAY'] = df.O_DATE.apply(calc_weekday)
    df.loc[:,'IS_WORKDAY'] = df.O_DATE.apply(_calc_workday)
    df.to_csv(file_name, index=False)


class LineAnalysis(BaseMixin):
    def __init__(self):
        self.start = None
        self.end = None
        self.file_path = ''
        self.result_file_path = 'fit'
        self.get_predict_info()

    def set_file(self, file_path):
        self.file_path = file_path

    def set_lineno(self, lineno):
        self.lineno = lineno
        self.raw_file = os.path.join(self.result_file_path,
                                     '{}.csv'.format(lineno))
        self.onestop_file = os.path.join(self.result_file_path,
                                         'onestop_{}.csv'.format(lineno))
        self.onestop_outlier_file = os.path.join(self.result_file_path,
                                                 'onestop_{}_out.csv'.format(
                                                     lineno))

    def set_date_range(self, start, end):
        assert start < end
        self.start = start
        self.end = end

    def line_process(self):
        # raw record
        df_line = pd.DataFrame()
        df_onestop = pd.DataFrame()
        for jt in range(self.start, self.end + 1):
            print(jt)
            date_str = '201710{}'.format(str(jt).zfill(2))
            d_date, weather, is_work, weekday = other_data(date_str, '%Y%m%d')
            file_name = os.path.join(self.file_path,
                                     'processed_t_{}.csv'.format(date_str))
            df_day = pd.read_csv(file_name).loc[:,
                     ['O_LINENO', 'O_TERMINALNO', 'O_TIME', 'O_UP',
                      'O_NEXTSTATIONNO', 'O_RELATIVETIME', 'O_USETIME']]
            df_day_line = df_day[df_day['O_LINENO'] == self.lineno]
            if not df_day_line.empty:
                min_stop = df_day_line['O_NEXTSTATIONNO'].min()
                df_day_line.loc[:, 'O_DATE'] = date_str
                df_day_line.loc[:, 'O_HOUR'] = df_day_line['O_TIME'].apply(
                    lambda x: x.split(':')[0])
                df_day_line.loc[:, 'O_WEEKDAY'] = is_work
                df_day_line.loc[:, 'O_WEATHER'] = weather
                df_line = concat_df(df_line, df_day_line)
                terminals = df_day_line[
                    'O_TERMINALNO'].drop_duplicates().tolist()
                for terminal in terminals:
                    df_terminal = df_day_line[
                        df_day_line.O_TERMINALNO == terminal].sort_values(
                        by='O_RELATIVETIME')

                    df_terminal_onestop = df_terminal[
                        (df_terminal['O_NEXTSTATIONNO'].diff() == 1.0) | (
                            df_terminal['O_NEXTSTATIONNO'] == min_stop)]
                    df_onestop = concat_df(df_onestop, df_terminal_onestop)

            else:
                print(date_str, ' has not line:{} data'.format(self.lineno))
        _write(df_line, self.raw_file)
        _write(df_onestop, self.onestop_file)
        return df_line

    def drop_outlier(self):
        df = pd.read_csv(
            os.path.join('fit', 'onestop_{}.csv'.format(self.lineno)))
        df_detect = outlier_detection(df)
        _write(df_detect, os.path.join('fit', 'drop_outlier',
                                       '{}.csv'.format(self.lineno)))

    def add_weekday(self):
        for it,line in enumerate(self.predict_lines):
            print(it,line)
            _add_weekday(line)


if __name__ == '__main__':
    # _for_predict()
    # process = PreProcess()
    # for it in range(17, 19):
    #     process.process(it, file_path='train', init=True)
    # process.process(25, file_path='train', init=True)
    import time
    start = time.time()
    la = LineAnalysis()
    la.add_weekday()


    print('use time', time.time() - start)
