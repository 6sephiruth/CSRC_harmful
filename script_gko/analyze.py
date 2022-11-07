from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import *

import pandas as pd
import numpy as np

import time

def get_invar(df, drop_field=[]):
    df_new = df[df.columns.drop(drop_field)]

    return list(df_new.columns[df_new.max(axis=0) == df_new.min(axis=0)])

def identity(df, a, b):
    return df

def minmax(df, mi, mx):
    ndf = df.copy()
    for c in df.columns:
        if mi[c] == mx[c]:
            ndf[c] = df[c] - mi[c]
        else:
            ndf[c] = (df[c] - mi[c]) / (mx[c] - mi[c])
    return ndf

def zscore(df, mean, std):
    ndf = df.copy()
    for c in df.columns:
        if std[c] == 0:
            ndf[c] = df[c] - mean[c]
        else:
            ndf[c] = (df[c] - mean[c]) / std[c]
    return ndf

def aug_norm(x, sigma=0.03):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def aug_pois(x, lam):
    return x + np.random.poisson(lam=lam, size=x.shape)

def aug_mixup(x, y, lamb):
    zeros = y == 0
    ones = y == 1
    cols = x.columns

    x_aug = np.zeros_like(x)
    x_aug[zeros] = mixup(x[zeros], lamb)
    x_aug[ones] = mixup(x[ones], lamb)

    return pd.DataFrame(x_aug, columns=cols)

def mixup(x, lamb=0.5):
    x_rand = x.sample(frac=1)
    return lamb*x + (1-lamb)*x_rand

def get_stats(y_t, y_p, weight=None):
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, sample_weight=weight).ravel()

    acc = (tp+tn)/(tn+fp+fn+tp)
    tpr = tp/(fp+tp)
    tnr = tn/(fn+tn)

    return np.array([acc,tpr,tnr])

def load_df(filename):
    df = pd.read_csv(filename)
    df = pd.DataFrame(df.drop('Unnamed: 0', axis=1))

    # remove all zeros
    s = np.sum(df, axis=1)

    return df[s > 0]

def train_test_split(df, frac=0.8, seed=0):
    train = df.sample(frac=frac, random_state=seed)
    test = df.drop(train.index)

    return train, test


### main df
def main():
    # set normalize method
    normalize = 'minmax'
    #normalize = 'identity'
    bench_name = 'result.tsv'
    model_name = 'RandomForestClassifier'
    cls_field = 'label'

    # setup initial dataset
    df_gam = load_df('./dataset/week_gamble/raw_gamble_recent.csv')
    df_adv = load_df('./dataset/raw_advertisement.csv')
    df_etc = load_df('./dataset/raw_white.csv')

    df_gam[cls_field] = 0         # 도박
    df_adv[cls_field] = 1         # 광고
    df_etc[cls_field] = 2         # 기타

    data_raw = pd.concat([df_gam, df_adv, df_etc])
    data_raw.fillna(0, inplace=True)

    weight = data_raw.value_counts(normalize=True, sort=False)

    # remove duplicates prior to split
    data_raw.drop_duplicates(inplace=True)
    data_raw['weight'] = weight.values

    # separate labels
    gam_raw = data_raw[data_raw[cls_field] == 0]
    adv_raw = data_raw[data_raw[cls_field] == 1]
    etc_raw = data_raw[data_raw[cls_field] == 2]

    seed = 8
    frac = 0.7

    # randomly split between labels
    gam_tr, gam_te = train_test_split(gam_raw, frac=frac, seed=seed)
    adv_tr, adv_te = train_test_split(adv_raw, frac=frac, seed=seed)
    etc_tr, etc_te = train_test_split(etc_raw, frac=frac, seed=seed)

    train_raw = pd.concat([gam_tr, adv_tr, etc_tr])
    test_raw = pd.concat([gam_te, adv_te, etc_te])

    # preprocessing
    invar_fields = get_invar(train_raw, drop_field=[cls_field])
    drop = invar_fields + ['weight', cls_field]
    valid_cols = train_raw.columns.drop(drop)

    # TODO: it takes too long to normalize
    # normalize & set test and training
    if normalize == 'zscore':
        norm_args = train_raw[valid_cols].mean(), train_raw[valid_cols].std()
    else:
        norm_args = train_raw[valid_cols].min(), train_raw[valid_cols].max()

    #x_train = eval(normalize)(train_raw[valid_cols], *norm_args)
    #x_test = eval(normalize)(test_raw[valid_cols], *norm_args)

    x_train = train_raw[valid_cols]
    y_train = train_raw[cls_field]
    w_train = train_raw['weight']
    x_test = test_raw[valid_cols]
    y_test = test_raw[cls_field]
    w_test = test_raw['weight']

    """
    # pca reduction
    pca = PCA(n_components='mle')
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    y_train = y_train.reset_index()[cls_field]
    y_test = y_test.reset_index()[cls_field]

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    """

    # training
    print('[*] training ...')
    model = eval(model_name)(random_state=seed, n_jobs=-1)
    s = time.time()
    model.fit(x_train, y_train)
    e = time.time()

    print('[*]', e-s, 'seconds taken')

    # evaluation
    y_pred_tr = model.predict(x_train)
    acc_tr = accuracy_score(y_train, y_pred_tr, sample_weight=w_train)

    y_pred_te = model.predict(x_test)
    acc_te = accuracy_score(y_test, y_pred_te, sample_weight=w_test)

    print(acc_tr, acc_te)

    y_pred_p = model.predict_proba(x_test)
    b_loss = log_loss(y_test, y_pred_p, sample_weight=w_test)

    print(b_tr, b_te)
    exit()

    std = x_train.std()

    # set random seed
    for n_aug in range(1,20):

        x_a, y_a, w_a = [], [], []
        for _ in range(n_aug):
            # augmentation
            #x_aug = aug_mixup(x_train, y_train, lamb=0.9)
            x_aug = x_train           # do not apply mixup
            x_aug = aug_norm(x_aug, sigma=std/50)
            y_aug = y_train.copy()
            w_aug = w_train.copy()

            x_a.append(x_aug)
            y_a.append(y_aug)
            w_a.append(w_aug)

        # concatenation
        x_train_ = pd.concat([x_train] + x_a)
        y_train_ = pd.concat([y_train] + y_a)
        w_train_ = pd.concat([w_train] + w_a)

        # training
        model_ = eval(model_name)(random_state=seed)
        model_.fit(x_train_, y_train_)

        # evaluation
        y_pred_tr = model_.predict(x_train_)
        a_tr = get_stats(y_train_, y_pred_tr, weight=w_train_)[0]

        y_pred_te = model_.predict(x_test)
        a_te = get_stats(y_test, y_pred_te, weight=w_test)[0]

        y_pred_p = model_.predict_proba(x_test)
        a_loss = log_loss(y_test, y_pred_p, sample_weight=w_test)

        # benchmark
        with open(bench_name,'a') as f:
            s = '\t'.join([model_name, str(n_aug), normalize]) + '\t'
            #s += '\t'.join(list(map(lambda t: f"{t:.4f}", b_te))) + '\t'
            #s += '\t'.join(list(map(lambda t: f"{t:.4f}", a_te))) + '\n'
            s += '\t'.join(list(map(lambda t: f"{t:.4f}",
                            [b_te, a_te, b_loss, a_loss]))) + '\n'
            f.write(s)


# run main
if __name__ == "__main__":
    main()