import os, pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn as skl

from sklearn.metrics import *

class SelfTrainingClassifier:
    def __init__(self, model, init_date, init_x, init_y):
        """
        A self-training classifier.

        :param model: Machine learning model. XGBoost model instance.
        :param init_date: Initial date.
        :param init_x: Initial x data.
        :param init_y: Initial y data.
        """
        assert isinstance(model, xgb.sklearn.XGBModel) or \
               isinstance(model, skl.base.BaseEstimator)
        assert len(init_x) == len(init_y)

        # initialize members
        self.model = model
        self.date = init_date
        self.x = init_x
        self.y = init_y
        self.cnt = 0

        # update model
        self.update_model()

    def update_model(self, save_model=True):
        """
        Update classifier.

        :param save_model (optional): Indicator for saving model.
        """
        model_dir = f'trained_models/{self.cnt}_{self.date}'

        try:
            self.model = pickle.load(open(model_dir,'rb'))
            print(f'[*] Loaded model {self.cnt}')

        except:
            self.model.fit(self.x, self.y.values.ravel())
            print(f'[*] Training done for model {self.cnt}')

        if save_model:
            os.makedirs('trained_models', exist_ok=True)
            pickle.dump(self.model, open(model_dir,'wb'))

        # report results
        self.report_result()
        self.cnt += 1

    def self_train(self, date, new_x, new_y=None):
        """
        Self-training with new instances.

        :param new_x: New x data.
        :param new_y (optional): New y data.
        """
        new_x = self.format_data(new_x)
        idx = None

        while True:
            if idx is None:
                cur_x = new_x

            pred_y = self.model.predict(cur_x[self.x.columns])
            pred_y_ = self.model.predict_proba(cur_x[self.x.columns])

            # filter index
            idx = np.abs(pred_y_[:,0] - 0.5) >= 0.49
            if sum(idx) == 0:
                break

            new_y = pd.DataFrame(pred_y[idx])
            add_x = cur_x[idx]
            cur_x = cur_x[~idx]

            self.merge_data(add_x, new_y)
            self.date = date
            self.update_model()

    def format_data(self, new_x):
        """
        Format new data to match current model's input shape.

        :param new_x: New x data.
        """
        remain_col = [c for c in self.x.columns if c not in new_x.columns]

        empty_df = pd.DataFrame(np.zeros((len(new_x),len(remain_col))), columns=remain_col)
        empty_df.index = new_x.index

        return pd.concat([new_x,empty_df], axis=1)

    def merge_data(self, new_x, new_y):
        """
        Merge dataset with new dataset.

        :param new_x: New x data.
        :param new_y: New y data.
        """
        self.x = pd.concat([self.x, new_x]).fillna(0)
        self.y = pd.concat([self.y, new_y])

    def report_result(self, out='out.tsv'):
        """
        Report self-training results to a file.

        :param out (optional): Output file.
        """
        if not os.path.exists(out):
            cols = ['n_week', 'date', 'acc', 'fpr', 'fnr']
            with open(out,'w') as f:
                f.write('\t'.join(cols))
                f.write('\n')

        stats = self.test_model(self.model, self.x, self.y)
        if type(stats) != list:
            stats = [stats]
        stats = map(lambda s: f'{s:.4f}', stats)

        with open(out,'a') as f:
            f.write('\t'.join([str(self.cnt), self.date]) + '\t')
            f.write('\t'.join(stats))
            f.write('\n')

    def report_attr(self, data_x=None, attr_method='importance', attr_dir='./attributions'):
        """
        Get attribution values for current model.

        :param attr_method: Type of attribution method.
        """
        if data_x is None:
            data_x = self.x

        os.makedirs(f'{attr_dir}/{attr_method}', exist_ok=True)

        if attr_method == 'importance':
            attr = self.model.feature_importances_
            prefix = 'imp'
        if attr_method == 'shap':
            prefix = 'shap'
            pass
        if attr_method == 'lime':
            prefix = 'lime'
            pass

        col_names = self.x.columns
        num_nonzero = len([x for x in attr if x > 0])
        num_plot = min(num_nonzero, 50)

        # column names and importances
        col_imp = list(zip(col_names, attr))
        col_imp = sorted(col_imp, key=lambda x: x[1], reverse=True)

        keywords = [t[0] for t in col_imp[:num_plot]]
        scores = [t[1] for t in col_imp[:num_plot]]

        plt.figure(figsize=(15,10), dpi=300)
        plt.barh(keywords[::-1], scores[::-1])
        plt.savefig(f'{attr_dir}/{attr_method}/{prefix}_{self.date}.png')

    @staticmethod
    def test_model(model, test_x, test_y, apply_thresh=False):
        """
        Test model's performance on the provided test data.

        :param test_x: Test data x.
        :param test_y: Test data y.
        """
        col_names = model.feature_names_in_

        if not apply_thresh:
            pred_y = model.predict(test_x[col_names])

        else:
            # TODO: update for multi-label classification
            pred_y_ = model.predict_proba(test_x[col_names])
            # get optimal threshold
            opt_thresh = acc_thresh(test_y, pred_y_[:,1])
            pred_y = pred_y_[:,1] > opt_thresh

        return accuracy_score(test_y, pred_y)

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