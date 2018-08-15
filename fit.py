import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import settings


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
    gbdt0 = GradientBoostingRegressor(loss='huber', learning_rate=0.4,
                                      n_estimators=600, subsample=0.7,
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


def search_parameters(X, y):
    """
    gridSearchCV
    :return:
    """
    # step 1
    param = {'n_estimators': range(300, 450, 50)}
    search_estimators = GridSearchCV(
        estimator=GradientBoostingRegressor(loss='huber', learning_rate=0.5,
                                            min_samples_split=100,
                                            min_samples_leaf=20, max_depth=8,
                                            subsample=0.8), param_grid=param,
        scoring='roc_auc', n_jobs=-1)
    search_estimators.fit(X, y)
    n_estimators = search_estimators.cv_results_['params'][
        search_estimators.best_index_].get('n_estimators', 81)
    print('search_estimators', n_estimators)

    param2 = {'max_depth': range(2, 10),
              'min_samples_split': range(50, 301, 50)}
    search_tree_depth = GridSearchCV(
        estimator=GradientBoostingRegressor(n_estimators=n_estimators,
                                            loss='huber', learning_rate=0.1,
                                            min_samples_leaf=20, subsample=0.8),
        param_grid=param2, scoring='roc_auc', n_jobs=-1)
    search_tree_depth.fit(X, y)
    result = search_tree_depth.cv_results_['params'][
        search_tree_depth.best_index_]
    print('search_tree_depth', result)
    max_depth = result.get('max_depth')
    # step 3
    param3 = {'min_samples_split': range(50, 301, 100),
              'min_samples_leaf': range(20, 401, 10)}
    search_tree_node = GridSearchCV(
        estimator=GradientBoostingRegressor(n_estimators=n_estimators,
                                            loss='huber', learning_rate=0.1,
                                            subsample=0.8, max_depth=max_depth),
        param_grid=param3, scoring='roc_auc', n_jobs=-1)
    search_tree_node.fit(X, y)
    result = search_tree_node.cv_results_['params'][
        search_tree_depth.best_index_]
    print(result)
    import pdb
    pdb.set_trace()
    return result


def parse(y_test, y_predict):
    y_diff = y_test - y_predict
    y_diff_arg = np.argsort(y_diff)


class Train:
    def __init__(self):
        """
        for all lines
        """
        self.models = {}

    def get_lines(self,
                  file=os.path.join('predict', 'toBePredicted_forUser.csv')):
        df = pd.read_csv(file)
        self.lines = sorted(df['O_LINENO'].drop_duplicates().tolist())

    def fit(self):
        it = 0
        for line in [652, 808, 856, 800, 912, 665, 850]:
            # for jt in range(75):
            #     line = self.lines[it*75+jt]
            # for line in self.lines:
            #     filename = os.path.join('output','{}.pickle'.format(line))
            # if os.path.exists(filename):
            #     continue
            print('line:', line)
            # file_name = os.path.join('fit','onestop_{}.csv'.format(line))
            file_name = os.path.join('fit', 'onestop_{}.csv'.format(line))
            X, y, X_test, y_test = load_data(file_name)
            if len(X) != 0:
                my_model = fit(X, y, X_test, y_test)
                # joblib.dump(my_model, filename)

    def adjust_fit(self):
        import time
        start = time.time()
        file_name = os.path.join(settings.PredictPath,'predict_valid_result.csv')
        X_train,y_train,X_test,y_test = load_data(file_name,settings.adjust_feature,'time_diff')
        if len(X_train)!=0:
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
            joblib.dump(gbdt0, os.path.join(settings.ModelPath,'adjust_1.pickle'))
        print('use time:',time.time()-start)
        return


    def search_param(self):
        target = 'O_USETIME'
        for line in [652, 808, 856, 800, 912, 665, 850]:
            filename = os.path.join('output', 'sp', '{}.pickle'.format(line))
            # if os.path.exists(filename):
            #     continue
            print('line:', line)
            # file_name = os.path.join('fit','onestop_{}.csv'.format(line))
            file_name = os.path.join('fit', 'onestop_{}.csv'.format(line))
            X, y = _load_lines(settings.features, target, file_name)
            if len(X) != 0:
                my_model = search_parameters(X, y)
                joblib.dump(my_model, filename)


if __name__ == "__main__":
    train = Train()
    # train.get_lines()
    train.adjust_fit()
    # train.fit()
    # train.search_param()
