import copy
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import settings
from base import BaseMixin, other_data

def concat_df(df, df_new):
    if df.size:
        df = pd.concat([df, df_new])
    else:
        df = df_new
    return df


def _convert(record, df_train):
    df_train_need = df_train[
        df_train["O_TERMINALNO"] == record['O_TERMINALNO']].sort_values(
        by='O_RELATIVETIME')
    max_stop = max(df_train_need.O_NEXTSTATIONNO)
    record['O_MAXSTOP'] = max_stop
    record['O_HOUR'] = int(record['predHour'].split(':')[0])
    tm_diff = record['predRelative'] - df_train_need['O_RELATIVETIME']
    df_train_valid = df_train_need[(tm_diff > 0) & (tm_diff < 3600 * 3)]
    train_after = df_train_need[tm_diff < 0].head(1).to_dict(orient='records')
    train_before = df_train_need[tm_diff > 0].tail(1).to_dict(orient='records')
    check_item = copy.deepcopy(record)
    if train_after:
        item = train_after[0]
        check_item.update({'after_stop': int(item['O_NEXTSTATIONNO']),
                           'after_time': item['O_RELATIVETIME'],
                           'after_up': item['O_UP']})
    if train_before:
        item = train_before[0]
        check_item.update({'before_stop': int(item['O_NEXTSTATIONNO']),
                           'before_time': item['O_RELATIVETIME'],
                           'before_up': item['O_UP'],
                           'time_delta': record['predRelative'] - item[
                               'O_RELATIVETIME']})
    else:
        check_item.update({'before_stop': 1000})
    return check_item, df_train_valid


def _predict(record, model):
    results = []
    base_time = record['before_time']
    stop_diff = record['stop_diff']
    if stop_diff < 0 and (record['before_up'] != record['O_UP']):
        for it in range(record['before_stop'] + 1, int(record['O_MAXSTOP']) + 1):
            inputs = np.array([record['before_up'], record['O_TERMINALNO'], it,
                               int(record['O_HOUR']) - 1, record['O_WEEKDAY']])
            y = model.predict(inputs.reshape(1, -1))
            base_time += int(y)
        for it in range(1, record['pred_start_stop_ID']):
            record['O_NEXTSTATIONNO'] = it + 1
            inputs = np.array([record['O_UP'], record['O_TERMINALNO'], it + 1,
                               int(record['O_HOUR']) - 1, record['O_WEEKDAY']])
            y = model.predict(inputs.reshape(1, -1))
            base_time += int(y)
    elif stop_diff > 0:
        # before : 3 ,pre:3
        for it in range(record['before_stop'], record['pred_start_stop_ID']):
            record['O_NEXTSTATIONNO'] = it + 1
            inputs = np.array([record['O_UP'], record['O_TERMINALNO'],
                               record['O_NEXTSTATIONNO'],
                               int(record['O_HOUR']) - 1, record['O_WEEKDAY']])
            y = model.predict(inputs.reshape(1, -1))
            base_time += int(y)

    for it in range(record['pred_start_stop_ID'],
                    record['pred_end_stop_ID'] + 1):
        record['O_NEXTSTATIONNO'] = it + 1
        inputs = np.array([record[feature] for feature in settings.features])
        y = model.predict(inputs.reshape(1, -1))
        base_time += int(y[0])
        results.append(base_time - record['predRelative'])
    result = ';'.join(map(str, map(int, results)))
    return result


def adjust_y(y, data):
    try:
        min_, max_, med = map(int,data)
    except:
        return y
    if min_ != max_ and (y < min_ or y > max_):
        print('outlet',min_,max_,med,y)
        return med
    else:
        return y


def get_similar_data(record):
    similar_data = {}
    similar = record.get('similar', None)
    if similar:
        if not isinstance(similar,str):
            return similar_data
        records = similar.split(';')
        for record in records:
            stop, data = record.split('|')
            similar_data[int(stop)] = map(float,data.split(','))
    return similar_data

def _predict_adjust_check_similar(record,model,adjust_model):
    similar_data = get_similar_data(record)
    results = []
    base_time = record['before_time']
    stop_diff = record['stop_diff']
    if stop_diff < 0 and (record['before_up'] != record['O_UP']):
        for it in range(record['before_stop'] + 1, int(record['O_MAXSTOP']) + 1):
            inputs = np.array([record['before_up'], record['O_TERMINALNO'], it,
                               int(record['O_HOUR']) - 1, record['O_WEEKDAY']])
            y = model.predict(inputs.reshape(1, -1))
            # ['O_UP', 'O_NEXTSTATIONNO', 'O_HOUR', 'O_WEEKDAY', 'O_TERMINALNO']
            imputs = np.array([record['O_UP'],it,int(record['O_HOUR']) - 1,record['O_WEEKDAY'],record['O_TERMINALNO']])
            t_y = adjust_model.predict(imputs.reshape(1,-1))
            simi = similar_data.get(it-1,None)
            if simi:
                y = adjust_y(int(y[0])-int(t_y[0]),simi)
            else:
                y=int(y[0])-int(t_y[0])
            base_time += int(y)
        for it in range(1, record['pred_start_stop_ID']):
            record['O_NEXTSTATIONNO'] = it + 1
            inputs = np.array([record['O_UP'], record['O_TERMINALNO'], it + 1,
                               int(record['O_HOUR']) - 1, record['O_WEEKDAY']])
            y = model.predict(inputs.reshape(1, -1))
            imputs = np.array([record['O_UP'], it+1, int(record['O_HOUR']) - 1,
                               record['O_WEEKDAY'], record['O_TERMINALNO']])
            t_y = adjust_model.predict(imputs.reshape(1, -1))
            simi = similar_data.get(it, None)
            if simi:
                y = adjust_y(int(y[0]) - int(t_y[0]), simi)
            else:
                y = int(y[0]) - int(t_y[0])
            base_time += int(y)
    elif stop_diff > 0:
        # before : 3 ,pre:3
        for it in range(record['before_stop'], record['pred_start_stop_ID']):
            record['O_NEXTSTATIONNO'] = it + 1
            inputs = np.array([record['O_UP'], record['O_TERMINALNO'],
                               record['O_NEXTSTATIONNO'],
                               int(record['O_HOUR']) - 1, record['O_WEEKDAY']])
            y = model.predict(inputs.reshape(1, -1))
            imputs = np.array(
                [record['O_UP'], it + 1, int(record['O_HOUR']) - 1,
                 record['O_WEEKDAY'], record['O_TERMINALNO']])
            t_y = adjust_model.predict(imputs.reshape(1, -1))
            simi = similar_data.get(it, None)
            if simi:
                y = adjust_y(int(y[0]), simi)
            else:
                y = int(y[0])
            base_time += int(y)
    return


def _predict_check_similar(record, model):
    similar_data = get_similar_data(record)
    results = []
    base_time = record['before_time']
    stop_diff = record['stop_diff']
    if stop_diff < 0 and (record['before_up'] != record['O_UP']):
        for it in range(record['before_stop'] + 1, int(record['O_MAXSTOP']) + 1):
            inputs = np.array([record['before_up'], record['O_TERMINALNO'], it,
                               int(record['O_HOUR']) - 1, record['IS_WORKDAY']])
            y = model.predict(inputs.reshape(1, -1))

            simi = similar_data.get(it-1,None)
            if simi:
                y = adjust_y(int(y[0]),simi)
            else:
                y=int(y[0])
            base_time += int(y)
        for it in range(1, record['pred_start_stop_ID']):
            record['O_NEXTSTATIONNO'] = it + 1
            inputs = np.array([record['O_UP'], record['O_TERMINALNO'], it + 1,
                               int(record['O_HOUR']) - 1, record['IS_WORKDAY']])
            y = model.predict(inputs.reshape(1, -1))
            simi = similar_data.get(it, None)
            if simi:
                y = adjust_y(int(y[0]),simi)
            else:
                y = int(y[0])
            base_time += int(y)
    elif stop_diff > 0:
        # before : 3 ,pre:3
        for it in range(record['before_stop'], record['pred_start_stop_ID']):
            record['O_NEXTSTATIONNO'] = it + 1
            inputs = np.array([record['O_UP'], record['O_TERMINALNO'],
                               record['O_NEXTSTATIONNO'],
                               int(record['O_HOUR']) - 1, record['IS_WORKDAY']])
            y = model.predict(inputs.reshape(1, -1))
            simi = similar_data.get(it, None)
            if simi:
                y = adjust_y(int(y[0]), simi)
            else:
                y = int(y[0])
            base_time += int(y)

    for it in range(record['pred_start_stop_ID'],
                    record['pred_end_stop_ID'] + 1):
        record['O_NEXTSTATIONNO'] = it + 1
        inputs = np.array([record[feature] for feature in settings.features])
        y = model.predict(inputs.reshape(1, -1))
        simi = similar_data.get(it, None)
        if simi:
            y = adjust_y(int(y[0]), simi)
        else:
            y = int(y[0])
        base_time += int(y)
        results.append(base_time - record['predRelative'])
    result = ';'.join(map(str, map(int, results)))
    return result


def calc_mediannum(a_list):
    a_list.sort()
    half = len(a_list) // 2
    median = (float(a_list[half]) + float(a_list[~half])) / 2
    return median


def _outlet(df, columns):
    def _adjust(x):
        data = map(int, map(float, x['pred_timeStamps'].split(';')))
        if data[0] < 0:
            new_data = [item + x['time_delta'] for item in data]
            if new_data[0] < 0:
                new_data = [item - new_data[0] for item in new_data]
            return ';'.join(map(str, map(int, new_data)))
        else:
            return x['pred_timeStamps']

    df.loc[:, 'pred_timeStamps'] = df.apply(_adjust, axis=1)
    df = df.sort_values(by='O_ORDER')
    # df_out = df.rename(columns={'pred_Stamp': 'pred_timeStamps'})
    return df.loc[:, columns]


class Predict(BaseMixin):
    def __init__(self):
        self.check_file = os.path.join(settings.PredictPath,
                                       'predict_check.csv')
        self.valid_file = os.path.join(settings.PredictPath,
                                       'predict_valid.csv')
        self.result_file = os.path.join(settings.ResultPath, 'result.csv')
        self.output_file = os.path.join(settings.ResultPath, 'output.csv')
        self.valid_clean_file = os.path.join(settings.PredictPath,
                                             'predict_valid_clean.csv')
        self.valid_result_file = os.path.join(settings.PredictPath,
                                              'predict_valid_result.csv')
        self.check_similar_file = os.path.join(settings.PredictPath,
                                               'predict_check_similar.csv')
        self.outlet_columns = ['O_DATA','O_LINENO','O_TERMINALNO','predHour','pred_start_stop_ID','pred_end_stop_ID','pred_timeStamps']
        self.get_predict_info()

    def convert(self):
        """
        convert toBePredicted_forUser.csv file for fit
        :return:
        """
        df_valid = pd.DataFrame()
        check_records = []
        for d in self.predict_dt:
            print('tm:', d)
            d_str = '-'.join(['2017', d])
            d_date = datetime.datetime.strptime(d_str, '%Y-%m-%d').date()
            d_date, weather, workday_type, workday_no = other_data(d_str)
            date_base = datetime.datetime.combine(d_date,
                                                  datetime.time(5, 0, 0))
            train_file = os.path.join(settings.TrainPath,
                                      'processed_t_{}.csv'.format(
                                          ''.join(d_str.split('-'))))

            df_predict = self.predict_df[self.predict_df['O_DATA'] == d]
            df_predict.loc[:, 'IS_WORKDAY'] = workday_type
            df_predict.loc[:, 'O_WEEKDAY'] = workday_no
            df_predict.loc[:, 'O_WEATHER'] = weather

            def date_parser(s_time):
                hour, minute, second = map(int, s_time.split(':'))
                d_time = datetime.time(hour, minute, second)
                return datetime.datetime.combine(d_date, d_time)

            ds_tm = pd.to_datetime(df_predict['predHour'].apply(date_parser))
            ds_relative = (ds_tm - date_base) / pd.Timedelta('1s')
            df_predict.loc[:, 'predRelative'] = ds_relative
            dt_predict = df_predict.to_dict(orient='records')
            df_train = pd.read_csv(train_file).loc[:, settings.predict_info]
            df_train.loc[:, 'O_DATA'] = np.array([d] * df_train.shape[0])

            for it, record in enumerate(dt_predict):
                print(it)
                new_record, df_train_valid = _convert(record, df_train)
                check_records.append(new_record)
                if not df_train_valid.empty:
                    df_train_valid.loc[:, 'O_ORDER'] = np.array(
                        [record['O_ORDER']] * df_train_valid.shape[0])
                    df_train_valid.loc[:, 'IS_WORKDAY'] = workday_type
                    df_train_valid.loc[:, 'O_WEEKDAY'] = workday_no
                    df_train_valid.loc[:, 'O_WEATHER'] = weather
                    df_train_valid.loc[:,
                    'O_HOUR'] = df_train_valid.O_TIME.apply(
                        lambda x: int(x.split(':')[0]))
                    df_train_valid.loc[:,
                    'O_USETIME'] = df_train_valid.O_RELATIVETIME.diff()
                    df_valid = concat_df(df_valid, df_train_valid)
        # save result
        df_valid.to_csv(self.valid_file, index=False)
        df_check = pd.DataFrame(check_records)
        df_check.to_csv(self.check_file, index=False)
        return

    def predict(self):
        df_predict = pd.read_csv(self.check_similar_file)
        new_records = []
        adjust_model = joblib.load(os.path.join('output','adjust.pickle'))
        for line in self.predict_lines:
            print(line)
            linemodel = joblib.load(
                os.path.join('output', '{}.pickle'.format(line)))

            df_line = df_predict[df_predict['O_LINENO'] == line]
            df_line.loc[:, 'stop_diff'] = df_line['pred_start_stop_ID'] - \
                                          df_line['before_stop']
            records = df_line.to_dict(orient='records')
            for record in records:
                # record['pred_timeStamps'] = _predict_adjust_check_similar(record,
                #                                                    linemodel,adjust_model)
                record['pred_timeStamps'] = _predict_check_similar(record, linemodel)
                # record['pred_timeStamps'] = _predict(record, linemodel)
                new_records.append(record)
        df_result = pd.DataFrame(new_records)
        df_result.sort_values(by='O_ORDER')
        df_result.to_csv(self.result_file, index=False)
        df_out = _outlet(df_result, self.outlet_columns)
        df_out.to_csv(self.output_file, index=False)
        return

    def outlet(self):
        df_result = pd.read_csv(self.result_file)
        df_out = _outlet(df_result, self.outlet_columns)
        df_out.to_csv(self.output_file, index=False)
        return


    def clean_valid(self):
        df_valid = pd.read_csv(self.valid_file)
        df_sorted = df_valid.sort_values(
            by=['O_DATA', 'O_TERMINALNO', 'O_RELATIVETIME'])
        df_sorted.loc[:, 'stop_diff'] = df_sorted.O_NEXTSTATIONNO.diff()
        df_step1 = df_sorted[df_sorted['O_HOUR'].diff() < 2]
        df_cleaned = df_step1[(df_step1['O_NEXTSTATIONNO'] == 2) | (
            df_step1['stop_diff'] == 1)].dropna()
        df_cleaned.to_csv(self.valid_clean_file, index=False)
        return

    def valid(self):
        df_valid = pd.read_csv(self.valid_clean_file)
        new_records = []
        for line in self.predict_lines:
            linemodel = joblib.load(
                os.path.join('output', '{}.pickle'.format(line)))

            df_line = df_valid[df_valid['O_LINENO'] == line]
            records = df_line.to_dict(orient='records')
            for record in records:
                inputs = np.array(
                    [record[feature] for feature in settings.features])
                y = linemodel.predict(inputs.reshape(1, -1))
                record['pred_usetime'] = int(y[0])
                record['time_diff'] = record['pred_usetime'] - record[
                    'O_USETIME']
                new_records.append(record)
        df = pd.DataFrame(new_records)
        df.to_csv(self.valid_result_file, index=False)

    def parse_valid_result(self):
        df = pd.read_csv(self.valid_result_file)
        import pdb
        pdb.set_trace()

    def similar_data(self):
        predict_df = pd.read_csv(self.check_file)
        new_record = []
        for line in self.predict_lines:
            print(line)
            df = predict_df[predict_df['O_LINENO'] == line]
            records = df.to_dict(orient='records')
            df_onestop = pd.read_csv(
                os.path.join(settings.FitPath, 'onestop_{}.csv'.format(line)))
            for record in records:
                import pdb
                pdb.set_trace()
                similar = []
                df_tl = df_onestop[
                    df_onestop['O_TERMINALNO'] == record['O_TERMINALNO']]
                df_tl_s = df_tl[df_tl['O_UP'] == record['O_UP']]
                df_hr_1_s = df_tl_s[
                    df_tl_s['O_HOUR'] == int(record['O_HOUR']) - 1]
                if not df_tl.empty:
                    stop_diff = record['pred_start_stop_ID'] - record[
                        'before_stop']
                    if stop_diff < 0:
                        df_tl_q = df_tl[df_tl['O_UP'] != record['O_UP']]
                        df_hr_1_q = df_tl_q[
                            df_tl_q['O_HOUR'] == int(record['O_HOUR']) - 1]
                        if not df_hr_1_q.empty:
                            for it in range(record['before_stop'] + 1,
                                            int(record['O_MAXSTOP']) + 1):
                                df_rs = df_hr_1_q[
                                    df_hr_1_q['O_NEXTSTATIONNO'] == it]
                                if not df_rs.empty:
                                    use_time = df_rs['O_USETIME'].tolist()
                                    result = [min(use_time), max(use_time),
                                              calc_mediannum(use_time)]
                                    similar.append(str(it) + '|' + ','.join(
                                        map(str, result)))
                            for it in range(1, record['pred_start_stop_ID']):
                                df_rs = df_hr_1_s[
                                    df_hr_1_s['O_NEXTSTATIONNO'] == it + 1]
                                if not df_rs.empty:
                                    use_time = df_rs['O_USETIME'].tolist()
                                    result = [min(use_time), max(use_time),
                                              calc_mediannum(use_time)]
                                    similar.append(str(it) + '|' + ','.join(
                                        map(str, result)))
                    if stop_diff > 0:
                        for it in range(record['before_stop'],
                                        record['pred_start_stop_ID']):
                            df_rs = df_hr_1_s[
                                df_hr_1_s['O_NEXTSTATIONNO'] == it + 1]
                            if not df_rs.empty:
                                use_time = df_rs['O_USETIME'].tolist()
                                result = [min(use_time), max(use_time),
                                          calc_mediannum(use_time)]
                                similar.append(
                                    str(it) + '|' + ','.join(map(str, result)))

                    for it in range(record['pred_start_stop_ID'],
                                    record['pred_end_stop_ID'] + 1):
                        df_rs = df_hr_1_s[
                            df_hr_1_s['O_NEXTSTATIONNO'] == it + 1]
                        if not df_rs.empty:
                            use_time = df_rs['O_USETIME'].tolist()
                            result = [min(use_time), max(use_time),
                                      calc_mediannum(use_time)]
                            similar.append(
                                str(it) + '|' + ','.join(map(str, result)))
                if similar:
                    # print(similar)
                    record['similar'] = ';'.join(similar)
                new_record.append(record)
        df = pd.DataFrame(new_record)
        df.to_csv(self.check_similar_file, index=False)
        return



if __name__ == '__main__':
    predict = Predict()
    # convert predict file:toBePredicted_forUser.csv
    predict.convert()
    # predict result and outlet
    predict.predict()
