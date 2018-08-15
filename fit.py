import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import settings
from base import BaseMixin


def _load_lines(features, target, data_file):
    """
    load and return the lines dateset(regression)
    :param return_X_y:
    :return:
    """
    try:
        df = pd.read_csv(data_file).dropna()
        df = df[df['O_USETIME'] >= 30]
        data = np.array(df.loc[:, features])
        target = np.array(df.loc[:, target])
        return data, target
    except:
        print('load data error!')
        return np.array([]), np.array([])


def load_data(file_name, features=settings.features, target='O_USETIME'):
    # features = ['O_UP', 'O_NEXTSTATIONNO', 'O_RELATIVETIME', 'O_WEEKDAY']
    # X, y = _load_lines(features, target, 'lineno_652_for_fit.csv')
    X, y = _load_lines(features, target, file_name)
    if len(X) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=666)
    return X_train, y_train, X_test, y_test


def fit(X_train, y_train, X_test, y_test):
    "raw"
    gbdt0 = GradientBoostingRegressor(loss='huber', learning_rate=0.2,
                                      n_estimators=500, subsample=0.7,
                                      min_samples_split=100,
                                      min_samples_leaf=20)
    gbdt0.fit(X_train, y_train)
    y_predict = gbdt0.predict(X_test)
    score = r2_score(y_test, y_predict)
    print('raw fit r2 score:', score)
    mse = mean_squared_error(y_test, y_predict)
    print('mes:', mse)
    y_predict_train = gbdt0.predict(X_train)
    score_raw = r2_score(y_train, y_predict_train)
    print('train r2 score:', score_raw)

    mse_raw = mean_squared_error(y_train, y_predict_train)
    print('train mse:', mse_raw)
    print('feature importance:', gbdt0.feature_importances_)
    return gbdt0


class Train(BaseMixin):
    def __init__(self):
        """
        for all lines
        """
        self.models = {}
        self.get_predict_info()

    def fit(self):
        for line in self.predict_lines:
            model_name = os.path.join('output', '{}.pickle'.format(line))
            print('line:', line)
            file_name = os.path.join('fit', 'onestop_{}.csv'.format(line))
            X, y, X_test, y_test = load_data(file_name)
            if len(X) != 0:
                my_model = fit(X, y, X_test, y_test)
                joblib.dump(my_model, model_name)

    def adjust_fit(self):
        import time
        start = time.time()
        file_name = os.path.join(settings.PredictPath,
                                 'predict_valid_result.csv')
        X_train, y_train, X_test, y_test = load_data(file_name,
                                                     settings.adjust_feature,
                                                     'time_diff')
        if len(X_train) != 0:
            # ad_model = fit(X,y,X_test,y_test)
            gbdt0 = GradientBoostingRegressor(loss='huber', learning_rate=0.3,
                                              n_estimators=600, subsample=0.7,
                                              min_samples_split=50,
                                              min_samples_leaf=5)
            gbdt0.fit(X_train, y_train)
            y_predict = gbdt0.predict(X_test)
            score = r2_score(y_test, y_predict)
            print('raw fit r2 score:', score)
            mse = mean_squared_error(y_test, y_predict)
            print('mes:', mse)
            y_predict_train = gbdt0.predict(X_train)
            score_raw = r2_score(y_train, y_predict_train)
            print('train r2 score:', score_raw)

            mse_raw = mean_squared_error(y_train, y_predict_train)
            print('train mse:', mse_raw)
            print('feature importance:', gbdt0.feature_importances_)
            joblib.dump(gbdt0,
                        os.path.join(settings.ModelPath, 'adjust_1.pickle'))
        print('use time:', time.time() - start)
        return


if __name__ == "__main__":
    train = Train()
    train.fit()
