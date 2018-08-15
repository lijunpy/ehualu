import os

import pandas as pd


def read_csv(file_name, dtype=None, file_path=None):
    if file_path:
        file_name = os.path.join(file_path, file_name)
    df = pd.read_csv(file_name)
    if dtype:
        try:
            df = df.astype(dtype)
        except:
            print('pandas transform dtype error!')
    return df


def anaysis_memory_usage(df):
    # anaysis memory usage according https://www.dataquest.io/blog/pandas-big-data/
    # https://blog.csdn.net/wally21st/article/details/77688755
    for dtype in ['float', 'int', 'object']:
        selected_dtype = df.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).sum()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,
                                                                        mean_usage_mb))
    return


def reduce_memory_usage(df):
    # reducing memory usage according https://www.dataquest.io/blog/pandas-big-data/
    # https://blog.csdn.net/wally21st/article/details/77688755
    # convert int
    selected_int = df.select_dtypes(include=['int'])
    convert_int = selected_int.apply(pd.to_numeric, downcast='unsigned')
    df[convert_int.columns] = convert_int
    # convert float
    selected_float = df.select_dtypes(include=['float'])
    convert_float = selected_float.apply(pd.to_numeric, downcast='float')
    df[convert_float.columns] = convert_float
    return df
