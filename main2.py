# main, but modularized
import shutil

from utils import *

import xgboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

def main():
    # 데이터 및 실험 결과 PATH
    experiment_path = './report_experiment/'
    normal_list = './dataset/week_normal/'
    gamble_list = './dataset/week_gamble/'
    stock_normal_list = './dataset/total_normal/'
    stock_gamble_list = './dataset/total_gamble/'

    create_folder(stock_gamble_list)

    # TODO: 여러개 파일을 받는 이유?
    normal_file_list = load_dataset_list(normal_list)
    gamble_file_list = load_dataset_list(gamble_list)

    # TODO: 두개 파일이 다른 column으로 이루어져 있으면?
    normal_total_dataset = load_total_dataframe(normal_file_list)
    gamble_total_dataset = load_total_dataframe(gamble_file_list)

    # TODO: 왜 폴더이름은 하나?
    folder_name = gamble_file_list[0][-10:-4]

    model = xgboost.XGBRFClassifier()

    S = SelfLearningClassifier(model, x, y)

    return

def train(model, data):
    model.fit(X=x_train, y=y_train)
    return

class SelfLearningClassifier(object):
    # initialize
    def __init__(self, model, x, y):
        assert isinstance(model, xgboost.sklearn.XGBModel) and len(x) == len(y)

        # initialize members
        self.model = model
        self.x = x
        self.y = y
        self.cnt = 0

        # update model
        self.update_model()

    # update model
    def update_model(self):
        self.model.fit(x,y)
        self.cnt += 1       # increase week counter

    # self learning with new data instances
    def self_learning(self, new_x):
        new_y = self.model.predict(new_x)

        self.merge_data(new_x, new_y)
        self.train_model()

    # merge data with newcoming data
    def merge_data(self, new_x, new_y):
        # TODO: fix this
        self.x = x + new_x
        self.y = y + new_y
        pass

    # report results
    def report_all(self, output='out.log'):
        self.last_acc =
        test_acc =

        with open(output, 'a') as f:
            f.write('')
            f.write('\n')


    @staticmethod
    def test_model(model, test_x, test_y):
        pred_y = model.predict(test_x)
        test_acc = accuracy_score(test_y, pred_y)

        print("Test accuracy: ", test_acc)

    @staticmethod
    def cross_val(model, x, y, k=10, seed=0):
        cv = StratifiedKFold(k, random_state=seed)
        scores = cross_val_score(model, x, y, cv=cv, random_state=seed)

        print("Cross-val score:", np.mean(scores))
        print("min score:", min(scores))
        print("max score:", max(scores))


if __name__ == "__main__":
    main()