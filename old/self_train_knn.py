# main, but modularized
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import numpy as np
import pandas as pd

import pickle

import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

def load_df(filename):
    df = pd.read_csv(filename)
    df = pd.DataFrame(df.drop('Unnamed: 0', axis=1))

    return df

def train_test_split(df, seed=0):
    train = df.sample(frac=0.8, random_state=seed)
    test = df.drop(train.index)

    return train, test

def main():
    init_date = '220117'
    seed = 1

    # setup initial dataset
    df_gamb = load_df('./dataset/week_gamble/raw_gamble_recent.csv')
    df_norm = load_df('./dataset/raw_white.csv')
    df_adver = load_df('./dataset/raw_advertisement.csv')

    df_gamb['label'] = 0        # 도박
    df_norm['label'] = 1        # 정상
    df_adver['label'] = 2       # 광고

    gamb_tr, gamb_te = train_test_split(df_gamb, seed=seed)
    norm_tr, norm_te = train_test_split(df_norm, seed=seed)
    adver_tr, adver_te = train_test_split(df_adver, seed=seed)

    df_train = pd.concat([gamb_tr, norm_tr, adver_tr])
    df_train.fillna(0, inplace=True)
    df_test = pd.concat([gamb_te, norm_te, adver_te])
    df_test.fillna(0, inplace=True)

    tot_cols = df_train.columns

    xy_train = shuffle(df_train, random_state=seed)
    xy_test = shuffle(df_test, random_state=seed)

    x_train = np.array(xy_train.drop('label', axis=1))
    y_train = np.array(xy_train['label'])

    x_test = np.array(xy_test.drop('label', axis=1))
    y_test = np.array(xy_test['label'])

    """ Stats on keyword sum
    for i in [0,1,2]:
        idx_tr = y_train == i
        x_tr = x_train[idx_tr]

        idx_te = y_test == i
        x_te = x_test[idx_te]

        s_tr = np.sum(x_tr, axis=1)
        s_te = np.sum(x_te, axis=1)

        print(f"[*] {i}")
        print(np.max(s_tr), np.mean(s_tr), np.min(s_tr))
        print(np.max(s_te), np.mean(s_te), np.min(s_te))
    """

    """
    s_train = np.sum(x_train, axis=1)
    s_test = np.sum(x_test, axis=1)

    print(np.max(s_train), np.mean(s_train), np.min(s_train))
    print(np.max(s_test), np.mean(s_test), np.min(s_test))
    exit()
    """

    idx_tr = np.sum(x_train, axis=1) > 0
    x_train = x_train[idx_tr]
    y_train = y_train[idx_tr]

    idx_te = np.sum(x_test, axis=1) > 0
    x_test = x_test[idx_te]
    y_test = y_test[idx_te]

    # define model
    try:
        model = pickle.load(open('knn_pre.pt','rb'))

    except:
        model = xgb.XGBClassifier(n_estimators=200,
                                  max_depth=10,
                                  learning_rate=0.5,
                                  min_child_weight=0,
                                  tree_method='gpu_hist',
                                  sampling_method='gradient_based',
                                  reg_alpha=0.2,
                                  reg_lambda=1.5)
        model.fit(x_train, y_train)
        pickle.dump(model, open('knn_pre.pt','wb'))

    neigh = KNN(n_neighbors=10,
                metric='l1',
                n_jobs=-1)
    neigh.fit(x_train, y_train)

    print("[*] train 정확도")
    y_pred = model.predict(x_train)
    y_pred_proba = model.predict_proba(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    print("Train accuracy: %.2f" % (accuracy * 100.0))

    conf = np.max(y_pred_proba, axis=1)
    print(conf.shape)
    print(np.max(conf), np.min(conf), np.mean(conf))

    mis_x_tr = x_train[y_train != y_pred]
    mis_y_tr = y_train[y_train != y_pred]
    mis_yp_tr = y_pred[y_train != y_pred]

    print("[*] test 정확도")
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy: %.2f" % (accuracy * 100.0))

    conf = np.max(y_pred_proba, axis=1)
    print(conf.shape)
    print(np.max(conf), np.min(conf), np.mean(conf))

    mis_x_te = x_test[y_test != y_pred]
    mis_y_te = y_test[y_test != y_pred]
    mis_yp_te = y_pred_proba[y_test != y_pred]
    knn_pred_prob = neigh.predict_proba(mis_x_te)

    print(mis_y_te)
    print(y_pred[y_test != y_pred])
    print(mis_yp_te)
    print(knn_pred_prob)
    exit()

    neigh = KNN(n_neighbors=2,
                metric='l1',
                n_jobs=-1)
    neigh.fit(x_train, y_train)

    # load unlabeled samples
    future = ['220117', '220425','220502','220530','220606','220613','220620','220704']
    for f in future:
        new_df = load_df(f'./dataset/week_gamble/{f}.csv')
        l = len(new_df)

        df_new = pd.concat([df_train, new_df])
        df_new.fillna(0, inplace=True)
        df_neww = df_new[-l:][tot_cols]

        df_neww['label'] = 0

        x_new = np.array(df_neww.drop('label', axis=1))
        y_new = np.array(df_neww['label'])

        y_pred = neigh.predict(x_new)
        y_pred_ = model.predict(x_new)

        ind = y_pred == y_pred_

        knn_acc = accuracy_score(y_new[ind], y_pred[ind])
        model_acc = accuracy_score(y_new[ind], y_pred_[ind])
        print("KNN accuracy: %.2f" % (knn_acc * 100.0))
        print("Model accuracy: %.2f" % (model_acc * 100.0))

        #agree = accuracy_score(y_pred, y_pred_)
        #print("Agreements: %.2f" % (agree * 100.0))

    exit()

    # initialize self-training classifier
    S = SelfTrainingClassifier(model, init_date, init_x, init_y)

    ##### self-training #####
    future = ['220425','220502','220530','220606','220613','220620','220704']
    for d in future:
        new_x = load_total_dataframe([f'./dataset/week_gamble/{d}.csv'])
        S.self_train(d, new_x)

    ##### ground truth testing #####
    all_dates = ['220117','220425','220502','220530','220606','220613','220620','220704']
    all_csv = map(lambda d: f'./dataset/week_gamble/{d}.csv', all_dates)

    all_gamb = load_total_dataframe(list(all_csv))
    all_gamb['label'] = 1

    df_all = pd.concat([all_gamb, df_norm])
    df_all.fillna(0, inplace=True)

    all_x = df_all.drop('label', axis=1)
    all_y = df_all['label']

    print(S.test_model(S.model, all_x, all_y, apply_thresh=True))

    # define model
    model = xgb.XGBClassifier(n_estimators=200,
                              max_depth=10,
                              learning_rate=0.5,
                              min_child_weight=0,
                              tree_method='gpu_hist',
                              sampling_method='gradient_based',
                              reg_alpha=0.2,
                              reg_lambda=1.5)

    # initialize self-training classifier
    S = SelfTrainingClassifier(model, 'all', all_x, all_y)

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


class SelfTrainingClassifier:
    def __init__(self, model, init_date, init_x, init_y):
        """
        A self-training classifier.

        :param model: Machine learning model. XGBoost model instance.
        :param init_date: Initial date.
        :param init_x: Initial x data.
        :param init_y: Initial y data.
        """
        assert isinstance(model, xgb.sklearn.XGBModel) and len(init_x) == len(init_y)

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
            self.model.fit(self.x, self.y)
            print(f'[*] Training done for model {self.cnt}')

        if save_model:
            os.makedirs('trained_models', exist_ok=True)
            pickle.dump(self.model, open(model_dir,'wb'))

        # report results
        self.report_result()
        #self.report_attr()
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
            with open(out, 'w') as f:
                f.write('\t'.join(cols))
                f.write('\n')

        stats = self.test_model(self.model, self.x, self.y)
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
            pred_y_ = model.predict_proba(test_x[col_names])
            # get optimal threshold
            opt_thresh = acc_thresh(test_y, pred_y_[:,1])
            pred_y = pred_y_[:,1] > opt_thresh

            print(roc_auc_score(test_y, pred_y_[:,1]))

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

        return acc, fpr, fnr

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