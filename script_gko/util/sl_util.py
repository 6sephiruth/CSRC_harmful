import os, pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn as skl
import shap
import operator

from sklearn.metrics import *
from collections import Counter

class SelfTrainingClassifier:
    def __init__(self, model, init_date, init_x, init_y, seed=0):
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
        self.seed = seed

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

    def update_expl(self):
        self.expl = shap.TreeExplainer(self.model, seed=self.seed)

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
            self.update_expl()
            self.shap_keyword()

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

    def shap_keyword(self):
        """
        Get shap values for current model.
        """
        ref_pattern = make_pattern(xs=self.x, ys=self.y,
                                   explainer=self.expl, n_key=15)
        print(ref_pattern)
        # TODO: pattern should be generated for each label
        exit()

        train_pattern = get_pattern(xs=self.x, ys=self.y, xs_mean=self.x.mean(),
                                    explainer=self.expl, n_key=15)
        print(train_pattern)

        s_train = pattern_score(train_pattern, self.y, ref_ptn)

        print(s_train)
        exit()

        if data_x is None:
            data_x = self.x

        os.makedirs(f'{attr_dir}/{attr_method}', exist_ok=True)

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

# generate keyword pattern
def make_pattern(xs, ys, explainer, n_key=5):
    keywords = get_keywords(xs, ys, explainer, n_key, w_key=2)
    print(keywords)
    print(np.array(keywords).shape)
    exit()

    pattern = []
    for l in set(ys):
        idx_l = ys.values == l
        key_l = keywords[idx_l]   # extract keywords

        # extract most frequent keywords
        cnt_l = Counter(list(key_l.flatten()))
        freq_l = sorted(cnt_l.items(), key=operator.itemgetter(1), reverse=True)[:n_key]

        ptn_l = []
        for k,_ in freq_l:
            avg_all = np.mean(xs[k])
            avg_l = np.mean(xs[idx_l][k])
            sgn_k = avg_l >= avg_all

            ptn_l.append((k,sgn_k))

        pattern.append(ptn_l)

    return pattern

# get keyword pattern
def get_pattern(xs, ys, xs_mean, explainer, n_key=5):
    keywords = get_keywords(xs, ys, explainer, n_key, w_key=1)

    pattern = []
    for (_,x),ks in zip(xs.iterrows(),keywords):
        ptn = [(k,x[k]>=xs_mean[k]) for k in ks]
        pattern.append(ptn)

    return pattern

# get keywords
def get_keywords(xs, ys, explainer, n_key=5, w_key=3):
    shap_val = explainer.shap_values(xs, check_additivity=False)
    best_idx = np.argsort(shap_val)[...,-int(w_key*n_key):]

    key_all = np.array(xs.columns)[best_idx]
    key_label = [key_all[y,i] for i,y in enumerate(ys.values)]

    return np.array(key_label)

def pattern_score(ps, ys, ref_p):
    assert len(ps) == len(ys)
    assert len(ref_p) == len(set(ys))

    return [match_ptn(p,ref_p[y]) for p,y in zip(ps,ys)]

def match_ptn(p1, p2):
    # TODO: weighted scoring
    print(p1)
    print(p2)
    print(len(p1))
    print(len(p2))
    assert len(p1) == len(p2)

    score = 0
    for k,s in p1:
        if (k,s) in p2:
            score += 1
        elif (k,not s) in p2:
            score += 0.5

    return score