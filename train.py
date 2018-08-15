from fit import Train
from preprocess import PreProcess, LineAnalysis


def train():
    # preprocess training data
    preprocess = PreProcess()
    preprocess.process()
    la = LineAnalysis()
    la.set_date_range(1, 31)
    la.process()
    # train model
    train = Train()
    train.fit()
    return


if __name__ == "__main__":
    # train data
    train()
