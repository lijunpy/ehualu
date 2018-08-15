"""
project:
http://www.dcjingsai.com/common/cmpt/%E5%85%AC%E4%BA%A4%E7%BA%BF%E8%B7%AF%E5%87%86%E7%82%B9%E9%A2%84%E6%B5%8B_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html

pandas api:http://pandas.pydata.org/pandas-docs/stable/api.html#timedeltaindex
"""

dtype_raw = {'O_LINENO': int, 'O_TERMINALNO': int, 'O_TIME': object,
             'O_LONGITUDE': float, 'O_LATITUDE': float, 'O_SPEED': int,
             'O_MIDDOOR': int, 'O_REARDOOR': int, 'O_FRONTDOOR': int,
             'O_UP': int, 'O_RUN': int, 'O_NEXTSTATIONNO': int}

duplicate_cols = ['O_LINENO', 'O_TERMINALNO', 'O_LONGITUDE', 'O_LATITUDE',
                  'O_SPEED', 'O_MIDDOOR', 'O_REARDOOR', 'O_FRONTDOOR', 'O_UP',
                  'O_RUN', 'O_NEXTSTATIONNO']

fit_columns = ['O_LINENO', 'O_TERMINALNO', 'O_UP', 'O_NEXTSTATIONNO',
               'O_RELATIVETIME', 'O_USETIME', 'O_HOUR']

predict_info = ['O_LINENO', 'O_TERMINALNO', 'O_UP', 'O_TIME', 'O_NEXTSTATIONNO',
                'O_RELATIVETIME']

weather_201710 = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 2, 10: 1,
                  11: 0, 12: 0, 13: 0, 14: 1, 15: 0, 16: 0, 17: 0, 18: 1, 19: 0,
                  20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
                  29: 0, 30: 0, 31: 0}

peak = ((2 * 3600, 4 * 3600), (12 * 3600, 14 * 3600))
# features = ['O_UP', 'O_TERMINALNO', 'O_NEXTSTATIONNO', 'O_HOUR', 'O_WEEKDAY']
features = ['O_UP', 'O_TERMINALNO', 'O_NEXTSTATIONNO', 'O_HOUR', 'IS_WORKDAY']
adjust_feature = ['O_UP', 'O_NEXTSTATIONNO', 'O_HOUR', 'O_WEEKDAY','O_TERMINALNO']

FitPath = 'fit'
TrainPath = 'train'
PredictPath = 'predict'
ModelPath = 'output'
TestPath = 'test'
ResultPath = 'result'
PredictFile = 'toBePredicted_forUser.csv'