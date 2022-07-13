import os
import pandas as pd
import numpy as np

import shap
from shap import Explanation, Cohorts

def createFolder(dir_path):
    '''
    폴더 만들기
    '''
    os.makedirs(dir_path, exist_ok = True)

def load_dataset_list(dir_path):
    '''
    dir_path에 무슨 파일이 들어있는지
    input: dir_path
    output: [./A, ./B, ./C]
    '''
    dataset_list = []

    for (root, _, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            dataset_list.append(file_path)

    return dataset_list

def load_total_dataframe(file_list):
    '''
    폴더 내 모든 file list를 불러 온 뒤,
    dataFrame을 하나로 묶는 과정
    '''
    empty_dataframe = pd.DataFrame()

    for each_file in file_list:

        each_file = pd.read_csv(each_file)
        each_file = pd.DataFrame(each_file.drop('Unnamed: 0', axis=1))
        empty_dataframe = pd.concat([empty_dataframe, each_file])

    total_dataframe = empty_dataframe

    return total_dataframe

def distribution_dataset(normal_dataset, gamble_dataset):

    normal_dataset["correct_label"] = 0
    gamble_dataset["correct_label"] = 1

    total_dataset = pd.concat([normal_dataset, gamble_dataset])
    total_dataset = total_dataset.fillna(0)

    total_columns = total_dataset.columns

    total_dataset = np.where(total_dataset>0, 1, 0)
    
    total_dataset = pd.DataFrame(total_dataset, columns=total_columns)
    total_dataset = total_dataset.sample(frac=1)

    train = total_dataset.sample(frac=0.9)
    test = total_dataset.drop(train.index)

    y_train = train['correct_label'].to_numpy()
    x_train = pd.DataFrame(train.drop(['correct_label'], axis=1))
    x_train = np.array(x_train)

    y_test = test['correct_label'].to_numpy()
    x_test = pd.DataFrame((test.drop(['correct_label'], axis=1)))
    x_test = np.array(x_test)

    return (x_train, y_train), (x_test, y_test), total_columns

def report_shap(x_train, total_columns, XGB_model):

    explainer = shap.Explainer(XGB_model)
    shap_values = explainer(x_train)

    #####################
    # 그대로 복붙 ㅎㅎ;;
    # 출력: shap_name, shap_value 기여도 0 이상만 출력
    #####################

    from shap import Explanation, Cohorts
    
    clustering=None
    clustering_cutoff=0.5
    max_display= x_train.shape[1]
    order=Explanation.abs
    clustering=None
    clustering_cutoff=0.5
    merge_cohorts=False
    show_data="auto"
    show=True
    import warnings
    import matplotlib.pyplot as pl
    from shap import Explanation, Cohorts
    from shap.plots._labels import labels
    from shap.plots._utils import convert_ordering
    # convert_color, merge_nodes, get_sort_order, sort_inds, 
    from shap.plots import colors
    from shap.utils import format_value, ordinal_str
    if isinstance(shap_values, Explanation):
        cohorts = {"": shap_values}
    elif isinstance(shap_values, Cohorts):
        cohorts = shap_values.cohorts
    else:
        assert isinstance(shap_values, dict), "You must pass an Explanation object, Cohorts object, or dictionary to bar plot!"
    # unpack our list of Explanation objects we need to plot
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())

    # unpack our list of Explanation objects we need to plot
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
        assert isinstance(cohort_exps[i], Explanation), "The shap_values paramemter must be a Explanation object, Cohorts object, or dictionary of Explanation objects!"
        assert cohort_exps[i].shape == cohort_exps[0].shape, "When passing several Explanation objects they must all have the same shape!"
        # TODO: check other attributes for equality? like feature names perhaps? probably clustering as well.

    # unpack the Explanation object
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names

    if clustering is None:
        partition_tree = getattr(cohort_exps[0], "clustering", None)
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering
    if partition_tree is not None:
        assert partition_tree.shape[1] == 4, "The clustering provided by the Explanation object does not seem to be a partition tree (which is all shap.plots.bar supports)!"
    op_history = cohort_exps[0].op_history
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])

    if len(values[0]) == 0:
        raise Exception("The passed Explanation is empty! (so there is nothing to plot)")

    # we show the data on auto only when there are no transforms
    if show_data == "auto":
        show_data = len(op_history) == 0

    # TODO: Rather than just show the "1st token", "2nd token", etc. it would be better to show the "Instance 0's 1st but", etc
    if issubclass(type(feature_names), str):
        feature_names = [ordinal_str(i)+" "+feature_names for i in range(len(values[0]))]

    # build our auto xlabel based on the transform history of the Explanation object
    xlabel = "SHAP value"
    for op in op_history:
        if op["name"] == "abs":
            xlabel = "|"+xlabel+"|"
        elif op["name"] == "__getitem__":
            pass # no need for slicing to effect our label, it will be used later to find the sizes of cohorts
        else:
            xlabel = str(op["name"])+"("+xlabel+")"
    # find how many instances are in each cohort (if they were created from an Explanation object)
    cohort_sizes = []
    for exp in cohort_exps:
        for op in exp.op_history:
            if op.get("collapsed_instances", False): # see if this if the first op to collapse the instances
                cohort_sizes.append(op["prev_shape"][0])
                break


    # unwrap any pandas series
    if str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values[0]))])

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values[0]))
    max_display = min(max_display, num_features)

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(values[0]))]
    orig_values = values.copy()
    while True:
        feature_order = np.argsort(np.mean([np.argsort(convert_ordering(order, Explanation(values[i]))) for i in range(values.shape[0])], 0))
        if partition_tree is not None:

            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values).mean(0))

            # now relax the requirement to match the parition tree ordering for connections above clustering_cutoff
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, clustering_cutoff, feature_order)

            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if max_display < len(feature_order) and dist[feature_order[max_display-1],feature_order[max_display-2]] <= clustering_cutoff:
                #values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values).mean(0), partition_tree)
                for i in range(len(values)):
                    values[:,ind1] += values[:,ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break
    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    feature_names_new = []
    for pos,inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        else:
            full_print = " + ".join([feature_names[i] for i in inds])
            if len(full_print) <= 40:
                feature_names_new.append(full_print)
            else:
                max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
                feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds)-1))
    feature_names = feature_names_new

    attribution_name = []    
    attribution_value = []

    for i in range(len(values)):
        for j in range(len(y_pos)):
            ind = feature_order[j]
        
            attribution_value.append(values[i,ind])

    for find_value_zero in range(x_train.shape[1]):

        attribution_name.append(total_columns[feature_inds[find_value_zero]])

        if attribution_value[find_value_zero] == 0:
            break


    return attribution_name[:find_value_zero], attribution_value[:find_value_zero]