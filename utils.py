import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, zero_one_loss
from copy import deepcopy
from demv import DEMV


from fairlearn.metrics import MetricFrame

import matplotlib.pyplot as plt
import seaborn as sns

# METRICS


def statistical_parity(data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str):
    query = '&'.join([f'{k}=={v}' for k, v in group_condition.items()])
    label_query = label_name+'=='+str(positive_label)
    unpriv_group_prob = (len(data_pred.query(query + '&' + label_query))
                         / len(data_pred.query(query)))
    priv_group_prob = (len(data_pred.query('~(' + query + ')&' + label_query))
                       / len(data_pred.query('~(' + query+')')))
    return unpriv_group_prob - priv_group_prob


def disparate_impact(data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str):
    query = '&'.join([f'{k}=={v}' for k, v in group_condition.items()])
    label_query = label_name+'=='+str(positive_label)
    unpriv_group_prob = (len(data_pred.query(query + '&' + label_query))
                         / len(data_pred.query(query)))
    priv_group_prob = (len(data_pred.query('~(' + query + ')&' + label_query))
                       / len(data_pred.query('~(' + query+')')))
    if( (unpriv_group_prob == 0) ):
        return 1
    else:
        return min(unpriv_group_prob / priv_group_prob, priv_group_prob/unpriv_group_prob) if unpriv_group_prob != 0 else unpriv_group_prob / priv_group_prob


def _compute_tpr_fpr(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    return FPR, TPR


def average_odds_difference(data_true: pd.DataFrame, data_pred: pd.DataFrame, group_condition: str, label: str):
    unpriv_group_true = data_true.query(group_condition)
    priv_group_true = data_true.drop(unpriv_group_true.index)
    unpriv_group_pred = data_pred.query(group_condition)
    priv_group_pred = data_pred.drop(unpriv_group_pred.index)

    y_true_unpriv = unpriv_group_true[label].values.ravel()
    y_pred_unpric = unpriv_group_pred[label].values.ravel()
    y_true_priv = priv_group_true[label].values.ravel()
    y_pred_priv = priv_group_pred[label].values.ravel()

    fpr_unpriv, tpr_unpriv = _compute_tpr_fpr(
        y_true_unpriv, y_pred_unpric)
    fpr_priv, tpr_priv = _compute_tpr_fpr(
        y_true_priv, y_pred_priv)
    return (fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv)/2


def zero_one_loss_diff(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: list):
    mf = MetricFrame(metrics=zero_one_loss,
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_features)
    return mf.difference()

# TRAINING FUNCTIONS

def get_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame, groups_condition: dict, label: str, positive_label: str):
    y_test = df_true
    y_pred = df_pred
    metrics = defaultdict(list)
    metrics['stat_par'].append(statistical_parity(
        df_pred, groups_condition, label, positive_label))
    metrics['disp_imp'].append(disparate_impact(
        df_pred, groups_condition, label, positive_label=positive_label))

    y_true = y_test[label]    
    y_pred = y_pred[label]
    metrics['zero_one_loss'].append(zero_one_loss_diff(
        y_true=y_true, y_pred=y_pred, sensitive_features=df_true[groups_condition].values))

    metrics['acc'].append(accuracy_score(y_true, y_pred))
    return metrics

def _train_test_split(df_train, df_test, label):
    x_train = df_train.drop(label, axis=1).values
    y_train = df_train[label].values.ravel()
    x_test = df_test.drop(label, axis=1).values
    y_test = df_test[label].values.ravel()
    return x_train, x_test, y_train, y_test


def cross_val(classifier, data, label, groups_condition, sensitive_features, positive_label, debiaser=None, exp=False, n_splits=10):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    metrics = {
        'stat_par': [],
        'zero_one_loss': [],
        'disp_imp': [],
        'acc': [],
        'f1': []
    }
    for train, test in fold.split(data):
        data = data.copy()
        df_train = data.iloc[train]
        df_test = data.iloc[test]
        model = deepcopy(classifier)
        if debiaser:
            run_metrics = _demv_training(model, debiaser, groups_condition, label,
                                         df_train, df_test, positive_label, sensitive_features)
        else:
            run_metrics = _model_train(df_train, df_test, label, model, defaultdict(
                list), groups_condition, sensitive_features, positive_label, exp)
        for k in metrics.keys():
            metrics[k].append(run_metrics[k])
    return model, metrics

def cross_val2(classifier, data, label, groups_condition, sensitive_features, positive_label, debiaser=None, exp=False, n_splits=10):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    metrics = {
        'stat_par': [],
        'zero_one_loss': [],
        'disp_imp': [],
        'acc': [],
        'f1': []
    }
    pred = None
    for train, test in fold.split(data):
        data = data.copy()
        df_train = data.iloc[train]
        df_test = data.iloc[test]
        model = deepcopy(classifier)
        if debiaser:
            run_metrics = _demv_training(model, debiaser, groups_condition, label,
                                         df_train, df_test, positive_label, sensitive_features)
        else:
            run_metrics, predtemp = _model_train2(df_train, df_test, label, model, defaultdict(
                list), groups_condition, sensitive_features, positive_label, exp)
            pred = predtemp if pred is None else pred.append(predtemp)
        for k in metrics.keys():
            metrics[k].append(run_metrics[k])
    return model, metrics, pred


def eval_demv(k, iters, data, classifier, label, groups, sensitive_features, positive_label=None):
    ris = defaultdict(list)
    for i in range(0, iters+1, k):
        data = data.copy()
        demv = DEMV(1, debug=False, stop=i)
        _, metrics = cross_val(classifier, data, label, groups,
                               sensitive_features, debiaser=demv, positive_label=positive_label)
        #metrics = _compute_mean(metrics)
        ris['stop'].append(i)
        for k, v in metrics.items():
            val = []
            for i in v:
                val.append(np.mean(i))
            ris[k].append(val)
    return ris


def _demv_training(classifier, debiaser, groups_condition, label, df_train, df_test, positive_label, sensitive_features):
    metrics = defaultdict(list)
    for _ in range(30):
        df_copy = df_train.copy()
        data = debiaser.fit_transform(
            df_copy, [keys for keys in groups_condition.keys()], label)
        metrics = _model_train(data, df_test, label, classifier, metrics,
                               groups_condition, sensitive_features, positive_label)
    return metrics

def _model_train(df_train, df_test, label, classifier, metrics, groups_condition, sensitive_features, positive_label, exp=False):
    x_train, x_test, y_train, y_test = _train_test_split(
        df_train, df_test, label)
    model = deepcopy(classifier)
    model.fit(x_train, y_train,
              sensitive_features=df_train[sensitive_features]) if exp else model.fit(x_train, y_train)
    pred = model.predict(x_test)
    df_pred = df_test.copy()
    df_pred[label] = pred
    metrics['stat_par'].append(statistical_parity(
        df_pred, groups_condition, label, positive_label))
    metrics['disp_imp'].append(disparate_impact(
        df_pred, groups_condition, label, positive_label=positive_label))
    metrics['zero_one_loss'].append(zero_one_loss_diff(
        y_true=y_test, y_pred=pred, sensitive_features=df_test[sensitive_features].values))
    metrics['acc'].append(accuracy_score(y_test, pred))
    metrics['f1'].append(f1_score(y_test, pred, average='weighted'))
    return metrics    


def _model_train2(df_train, df_test, label, classifier, metrics, groups_condition, sensitive_features, positive_label, exp=False):
    x_train, x_test, y_train, y_test = _train_test_split(
        df_train, df_test, label)
    model = deepcopy(classifier)
    model.fit(x_train, y_train,
              sensitive_features=df_train[sensitive_features]) if exp else model.fit(x_train, y_train)
    pred = model.predict(x_test)
    df_pred = df_test.copy()
    df_pred['y_true'] = df_pred[label]
    df_pred[label] = pred
    metrics['stat_par'].append(statistical_parity(
        df_pred, groups_condition, label, positive_label))
    metrics['disp_imp'].append(disparate_impact(
        df_pred, groups_condition, label, positive_label=positive_label))
    metrics['zero_one_loss'].append(zero_one_loss_diff(
        y_true=y_test, y_pred=pred, sensitive_features=df_test[sensitive_features].values))
    metrics['acc'].append(accuracy_score(y_test, pred))
    metrics['f1'].append(f1_score(y_test, pred, average='weighted'))
    return metrics, df_pred


def print_metrics(metrics):
    print('Statistical parity: ', round(np.mean(
        metrics['stat_par']), 3), ' +- ', round(np.std(metrics['stat_par']), 3))
    print('Disparate impact: ', round(np.mean(
        metrics['disp_imp']), 3), ' +- ', round(np.std(metrics['disp_imp']), 3))
    print('Zero one loss: ', round(np.mean(
        metrics['zero_one_loss']), 3), ' +- ', round(np.std(metrics['zero_one_loss']), 3))
    print('F1 score: ', round(
        np.mean(metrics['f1']), 3), ' +- ', round(np.std(metrics['f1']), 3))
    print('Accuracy score: ', round(np.mean(
        metrics['acc']), 3), ' +- ', round(np.std(metrics['acc']), 3))


# PLOT FUNCTIONS

def plot_group_percentage(data, protected_vars: list, label_name, label_value):
    full_list = protected_vars.copy()
    full_list.append(label_name)
    perc = (data[full_list]
            .groupby(protected_vars)[label_name]
            .value_counts(normalize=True)
            .mul(100).rename('Percentage')
            .reset_index()
            )
    perc['Groups'] = perc[protected_vars].apply(
        lambda x: '('+','.join(x.astype(str))+')', axis=1)
    sns.barplot(data=perc[perc[label_name]
                == label_value], x='Groups', y='Percentage')
    plt.title('Percentage distribution of label for each sensitive group')
    plt.show()


def plot_metrics_curves(df, points, title=''):

    metrics = {'stat_par': 'Statistical Parity', 'zero_one_loss': 'Zero One Loss',
               'disp_imp': 'Disparate Impact', 'acc': 'Accuracy'}
    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    for k, v in metrics.items():
        ax = sns.lineplot(data=df, y=k, x='stop', label=v, )
    for k, v in points.items():
        ax.plot(v['x'], v['y'], v['type'], label=k, markersize=10)
    ax.set(ylabel='Value', xlabel='Stop value')
    ax.lines[0].set_linestyle("--")
    ax.lines[0].set_marker('o')
    #lines[1] is zero_one_loss
    ax.lines[1].set_marker('x')
    ax.lines[1].set_markeredgecolor('orange')
    ax.lines[1].set_linestyle("--")

    ax.lines[2].set_marker('+')
    ax.lines[2].set_markeredgecolor('green')
    ax.lines[2].set_linestyle(":")
    ax.lines[2].set_markevery(0.001)

    ax.lines[3].set_color("black")
    ax.legend(handlelength=5, loc="upper center", bbox_to_anchor=(
        0.5, -0.03), ncol=3, fancybox=True, shadow=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_grid(dfs, ys, iter, types, metrics):
    fig = plt.figure(dpi=60, tight_layout=True)
    fig.set_size_inches(15, 5, forward=True)

    gs = fig.add_gridspec(1, len(dfs))

    ax = np.zeros(3, dtype=object)

    for k, v in dfs.items():
        df = v
        points = ys[k]
        iters = iter[k]
        i = list(dfs.keys()).index(k)
        ax[i] = fig.add_subplot(gs[0, i])
        for key, v in metrics.items():
            ax[i] = sns.lineplot(data=df, y=key, x='stop', label=v, )

        for key, v in points.items():
            ax[i].plot(iters, points[key], types[key],
                       label=key, markersize=10)

        ax[i].set(ylabel='Value', xlabel='Stop value')
        ax[i].set_title(k)

        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax[i].lines[1].set_marker('x')
        ax[i].lines[1].set_markeredgecolor('orange')
        ax[i].lines[1].set_linestyle("--")

        ax[i].lines[2].set_marker('+')
        ax[i].lines[2].set_markeredgecolor('green')
        ax[i].lines[2].set_linestyle(":")
        ax[i].lines[2].set_markevery(0.001)
        ax[i].get_legend().remove()
        ax[i].plot()

    handles, labels = ax[len(dfs)-1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.03), ncol=4, prop={'size': 15}, fancybox=True, shadow=True)
    fig.savefig('img/Grid.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_gridmulti(dfs, ys, iter, types, metrics, name='GridMulti'):
    fig = plt.figure(dpi=60, tight_layout=True)
    fig.set_size_inches(15, 8, forward=True)

    gs = fig.add_gridspec(2, 6)
    ax = np.zeros(5, dtype=object)

    for k, v in dfs.items():

        df = v
        points = ys[k]
        iters = iter[k]
        i = list(dfs.keys()).index(k)

        if(i == 0):
            ax[i] = fig.add_subplot(gs[0, :2])
        elif(i == 1):
            ax[i] = fig.add_subplot(gs[0, 2:4])
        elif(i == 2):
            ax[i] = fig.add_subplot(gs[0, 4:])
        elif(i == 3):
            ax[i] = fig.add_subplot(gs[1, 1:3])
        elif(i == 4):
            ax[i] = fig.add_subplot(gs[1, 3:5])

        for key, v in metrics.items():
            ax[i] = sns.lineplot(data=df, y=key, x='stop', label=v, )

        for key, v in points.items():
            ax[i].plot(iters, points[key], types[key],
                       label=key, markersize=10)

        ax[i].set(ylabel='Value', xlabel='Stop value')
        ax[i].set_title(k)

        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax[i].lines[1].set_marker('x')
        ax[i].lines[1].set_markeredgecolor('orange')
        ax[i].lines[1].set_linestyle("--")

        ax[i].lines[2].set_marker('+')
        ax[i].lines[2].set_markeredgecolor('green')
        ax[i].lines[2].set_linestyle(":")
        ax[i].lines[2].set_markevery(0.001)
        ax[i].get_legend().remove()
        ax[i].plot()

    handles, labels = ax[len(dfs)-1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.03), ncol=4, prop={'size': 15}, fancybox=True, shadow=True)
    fig.savefig(f'img/{name}.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')


def preparepoints(metrics, iters):

    types = {'Stastical Parity (Blackbox)': 'xb',
             'Zero One Loss (Blackbox)': 'xy',
             'Disparate Impact (Blackbox)': 'xg',
             'Accuracy (Blackbox)': 'xr',
             }

    rename = {'Stastical Parity (Blackbox)': 'stat_par',
              'Zero One Loss (Blackbox)': 'zero_one_loss',
              'Disparate Impact (Blackbox)': 'disp_imp',
              'Accuracy (Blackbox)': 'acc'
              }

    points = {}

    for k in types.keys():
        points[k] = {'x': iters, 'y': np.mean(
            metrics[rename[k]]), 'type':  types[k]}

    return points


def unprivpergentage(data, unpriv_group, iters):
    unprivdata = data.copy()
    for k, v in unpriv_group.items():
        unprivdata = unprivdata[(unprivdata[k] == v)]

    xshape, _ = unprivdata.shape

    print('Dataset size:', data.shape[0])
    print('Unprivileged group size:', xshape)
    print('Percentage of unprivileged group:', (xshape/data.shape[0])*100)
    print('Number of iterations:', iters)


def prepareplots(metrics, name):

    df = pd.DataFrame(metrics)
    columnlist = []
    for i in df.columns.values:
        if (i != 'stop'):
            columnlist.append(i)

    df = df.explode(columnlist)

    df.to_csv('ris/'+name+'_eval.csv')

    return df


def gridcomparison(dfs, dfsm, ys, ysm, iter, iterm, types, metrics):

    fig = plt.figure(dpi=60, tight_layout=True)
    fig.set_size_inches(15, 15, forward=True)

    gs = fig.add_gridspec(5, 2)
    ax = np.zeros(10, dtype=object)

    for k, v in dfs.items():
        df = v
        points = ys[k]
        iters = iter[k]
        i = list(dfs.keys()).index(k)

        if(i == 0):
            ax[i] = fig.add_subplot(gs[0, 0])
        elif(i == 1):
            ax[i] = fig.add_subplot(gs[1, 0])
        elif(i == 2):
            ax[i] = fig.add_subplot(gs[2, 0])
        elif(i == 3):
            ax[i] = fig.add_subplot(gs[3, 0])
        elif(i == 4):
            ax[i] = fig.add_subplot(gs[4, 0])

        for key, v in metrics.items():
            ax[i] = sns.lineplot(data=df, y=key, x='stop', label=v, )

        for key, v in points.items():
            ax[i].plot(iters, points[key], types[key],
                       label=key, markersize=10)

        ax[i].set(ylabel='Value', xlabel='Stop value')
        ax[i].set_title(k + " single var")

        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax[i].lines[1].set_marker('x')
        ax[i].lines[1].set_markeredgecolor('orange')
        ax[i].lines[1].set_linestyle("--")

        ax[i].lines[2].set_marker('+')
        ax[i].lines[2].set_markeredgecolor('green')
        ax[i].lines[2].set_linestyle(":")
        ax[i].lines[2].set_markevery(0.001)
        ax[i].get_legend().remove()
        ax[i].plot()

    for k, v in dfsm.items():
        df = v
        points = ysm[k]
        iters = iterm[k]
        i = list(dfsm.keys()).index(k)

        if(i == 0):
            ax[i] = fig.add_subplot(gs[0, 1])
        elif(i == 1):
            ax[i] = fig.add_subplot(gs[1, 1])
        elif(i == 2):
            ax[i] = fig.add_subplot(gs[2, 1])
        elif(i == 3):
            ax[i] = fig.add_subplot(gs[3, 1])
        elif(i == 4):
            ax[i] = fig.add_subplot(gs[4, 1])

        for key, v in metrics.items():
            ax[i] = sns.lineplot(data=df, y=key, x='stop', label=v, )

        for key, v in points.items():
            ax[i].plot(iters, points[key], types[key],
                       label=key, markersize=10)

        ax[i].set(ylabel='Value', xlabel='Stop value')
        ax[i].set_title(k)

        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax[i].lines[1].set_marker('x')
        ax[i].lines[1].set_markeredgecolor('orange')
        ax[i].lines[1].set_linestyle("--")

        ax[i].lines[2].set_marker('+')
        ax[i].lines[2].set_markeredgecolor('green')
        ax[i].lines[2].set_linestyle(":")
        ax[i].lines[2].set_markevery(0.001)
        ax[i].get_legend().remove()
        ax[i].plot()

    handles, labels = ax[len(dfs)-1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.03), ncol=4, prop={'size': 15}, fancybox=True, shadow=True)
    fig.savefig('img/GridMultiSingleVar.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')



def blackboxCVmetrics( data, label, y_true, unpriv_group, pred ):
    #I prepare folds such that each one contains all the unique values of the label feature. Otherwise, MulticlassBalancer won't be able to work.

    from sklearn.model_selection import KFold
    from balancers import MulticlassBalancer

    kf = KFold(n_splits = 10)
    okbool = False
    uniquevalues = data[label].unique().size

    attempt = 0
    while(not okbool):
        folds = []
        pred = pred.sample(frac=1).reset_index(drop=True)
        for train, test in kf.split(y_true):
            if( pred.loc[test, 'y_true'].unique().size >= uniquevalues and pred.loc[test,label].unique().size >= uniquevalues):
                okbool = True
                folds.append(test)

    bbmetrics = []
        
    for fold in folds:
        pb = MulticlassBalancer(y = 'y_true', y_=label, a='combined', data=pred.loc[fold])

        y_adj = pb.adjust(cv = True, summary = False)
        datapred = deepcopy(pred)
        datapred.loc[fold,label] = y_adj
        bbmetrics.append( get_metrics( data, datapred , unpriv_group, label, 1) )

    blackboxmetrics = {}
    for metric in bbmetrics[0]:
        temparr = []
        for i in range(len(bbmetrics)):
            temparr.append( bbmetrics[i].get(metric) )
        
        blackboxmetrics[metric] = ( np.mean(temparr) )

    return blackboxmetrics