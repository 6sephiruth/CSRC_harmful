# main, but modularized
import shutil

from utils import *

import xgboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

# 데이터 및 실험 결과 PATH
#experiment_path = './report_experiment/'
#normal_list = './dataset/week_normal/'
#gamble_list = './dataset/week_gamble/'
#stock_normal_list = './dataset/total_normal/'
#stock_gamble_list = './dataset/total_gamble/'

#create_folder(stock_gamble_list)

#normal_file_list = load_dataset_list(normal_list)
#gamble_file_list = load_dataset_list(gamble_list)

#normal_total_dataset = load_total_dataframe(normal_file_list)
#gamble_total_dataset = load_total_dataframe(gamble_file_list)

#folder_name = gamble_file_list[0][-10:-4]

def main():
    # setup initial dataset
    df_gamb = load_total_dataframe(['./dataset/week_gamble/220117.csv'])
    df_norm = load_total_dataframe(['./dataset/week_normal/raw_white.csv'])

    df_gamb['label'] = 1
    df_norm['label'] = 0

    df_init = pd.concat([df_gamb, df_norm])
    df_init.fillna(0, inplace=True)

    col_names = [c for c in df_init.columns if c != 'label']    # is this necessary? yes!
    init_x = df_init[col_names]
    init_y = df_init['label']

    # define model
    model = xgboost.XGBRFClassifier()

    # initialize self-learning classifier
    S = SelfLearningClassifier(model, init_x, init_y)
    print(S.cnt)
    print(S.model)
    print(S.report_result())

    new_x = check_new()
    if new_x:
        S.self_learn(new_x)

    return


def check_new(data_dir='./dataset/week_gamble/'):
    """
    Checks if new data is available.
    """

    csv_files = sorted(glob.glob(f'{data_dir}/*.csv'))
    if len(csv_files):
        data = load_total_dataframe(csv_files)
        return data.fillna(0)

    return None


class SelfLearningClassifier:
    def __init__(self, model, init_x, init_y):
        """
        A self-learning classifier.

        :param model: Machine learning model. XGBoost model instance.
        :param init_x: Initial x data.
        :param init_y: Initial y data.
        """
        assert isinstance(model, xgboost.sklearn.XGBModel) and len(init_x) == len(init_y)

        # initialize members
        self.model = model
        self.x = init_x
        self.y = init_y
        self.cnt = -1

        # update model
        self.update_model()

    def update_model(self):
        """
        Update classifier.
        """
        self.model.fit(self.x,self.y)
        self.cnt += 1       # increase week counter

    def self_learn(self, new_x, new_y=None):
        """
        Self-learning with new instances.

        :param new_x: New x data.
        :param new_y (optional): New y data.
        """
        if new_y is not None:
            new_y = self.model.predict(new_x)

        self.merge_data(new_x, new_y)
        self.update_model()
        self.report_result()

    def merge_data(self, new_x, new_y):
        """
        Merge dataset with new dataset.

        :param new_x: New x data.
        :param new_y: New y data.
        """
        # TODO: fix this
        self.x = x + new_x
        self.y = y + new_y

    def report_result(self, out='out.tsv'):
        """
        Report self-learning results.

        :param out: Output file.
        """
        test_acc = self.test_model(self.model, self.x, self.y)
        return test_acc

        #with open(out,'a') as f:
        #    f.write('\t'.join())
        #    f.write('\n')


    @staticmethod
    def test_model(model, test_x, test_y):
        """
        Test model on the provided datasets.

        :param test_x: Test data x.
        :param test_y: Test data y.
        """
        pred_y = model.predict(test_x)
        test_acc = accuracy_score(test_y, pred_y)

        return test_acc

    @staticmethod
    def cross_val(model, x, y, k=10, seed=0):
        """
        Cross validation on the provided datasets.

        :param x: x
        :param y: y
        """
        cv = StratifiedKFold(k, random_state=seed)
        scores = cross_val_score(model, x, y, cv=cv, random_state=seed)

        return np.mean(scores)


if __name__ == "__main__":
    main()