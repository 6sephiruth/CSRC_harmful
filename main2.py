# main, but modularized
import shutil

from utils import *

import xgboost
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

def main():
    init_date = '220117'

    # setup initial dataset
    df_gamb = load_total_dataframe([f'./dataset/week_gamble/{init_date}.csv'])
    df_norm = load_total_dataframe(['./dataset/week_normal/raw_white.csv'])

    #df_norm = df_norm[:len(df_gamb)]    # match length of normal instance

    df_gamb['label'] = 1
    df_norm['label'] = 0

    df_init = pd.concat([df_gamb, df_norm])
    df_init.fillna(0, inplace=True)

    col_names = [c for c in df_init.columns if c != 'label']
    init_x = df_init[col_names]
    init_y = df_init['label']

    # define model
    model = xgboost.XGBClassifier(n_estimators=200,
                                  max_depth=10,
                                  learning_rate=0.5,
                                  min_child_weight=0,
                                  tree_method='gpu_hist',
                                  sampling_method='gradient_based',
                                  reg_alpha=0.2,
                                  reg_lambda=1.5)

    # initialize self-learning classifier
    S = SelfLearningClassifier(model, init_date, init_x, init_y)

    ##### new iteration #####
    dates = ['220425','220502','220530','220606','220613','220620','220704']
    for d in dates:
        new_x = load_total_dataframe([f'./dataset/week_gamble/{d}.csv'])
        S.self_learn(d, new_x)

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
    def __init__(self, model, init_date, init_x, init_y):
        """
        A self-learning classifier.

        :param model: Machine learning model. XGBoost model instance.
        :param init_date: Initial date.
        :param init_x: Initial x data.
        :param init_y: Initial y data.
        """
        assert isinstance(model, xgboost.sklearn.XGBModel) and len(init_x) == len(init_y)

        # initialize members
        self.model = model
        self.date = init_date
        self.x = init_x
        self.y = init_y
        self.cnt = 0

        # update model
        self.update_model(save_model=True)

    def update_model(self, save_model=False):
        """
        Update classifier.
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
        self.report_attr()
        self.cnt += 1

    def self_learn(self, date, new_x, new_y=None):
        """
        Self-learning with new instances.

        :param new_x: New x data.
        :param new_y (optional): New y data.
        """
        new_x = self.format_data(new_x)
        if not new_y:
            pred_y = self.model.predict(new_x[self.x.columns])
            new_y = pd.DataFrame(pred_y)

        self.merge_data(new_x, new_y)
        self.date = date
        self.update_model(save_model=True)

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
        Report self-learning results to a file.

        :param out: Output file.
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
        if not data_x:
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
    def test_model(model, test_x, test_y):
        """
        Test model's performance on the provided test data.

        :param test_x: Test data x.
        :param test_y: Test data y.
        """
        pred_y = model.predict(test_x)

        # metrics
        acc = accuracy_score(test_y, pred_y)
        #prec = precision_score(test_y, pred_y)
        #rec = recall_score(test_y, pred_y)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

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