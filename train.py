from fit import Train
from preprocess import PreProcess, LineAnalysis


def train(start=1, end=31):
    # preprocess training data
    preprocess = PreProcess()
    preprocess.set_date_range(start, end)
    preprocess.process()
    la = LineAnalysis()
    la.set_date_range(start, end)
    la.process()
    # train model
    train = Train()
    train.fit()
    return


if __name__ == "__main__":
    # train data
    train()
