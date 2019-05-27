import numpy as np
import os

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC

from sated_nmt import OUTPUT_PATH as SATED_OUTPUT_PATH
from reddit_lm import RESULT_PATH as REDDIT_OUTPUT_PATH
from dialogue import OUTPUT_PATH as CORNELL_OUTPUT_PATH


def histogram_feats(ranks, bins=100, top_words=5000, num_words=5000, relative=False):
    if top_words < num_words:
        if bins == top_words:
            bins += 1
        top_words += 1

    range = (-num_words, top_words) if relative else (0, top_words)
    feats, _ = np.histogram(ranks, bins=bins, normed=False, range=range)
    return feats


def sample_with_ratio(a, b, heldout_ratio=0.5):
    if heldout_ratio == 0.:
        return a
    if heldout_ratio == 1.:
        return b

    if not isinstance(a, list):
        a = a.tolist()
        b = b.tolist()

    l1 = len(a)
    l2 = len(b)
    ratio = float(l2) / (l1 + l2)
    if heldout_ratio > ratio:
        # remove from a
        n = int(l2 / heldout_ratio)
        rest_l1 = n - l2
        return a[:rest_l1] + b
    elif heldout_ratio < ratio:
        # remove from b
        n = int(l1 / (1 - heldout_ratio))
        rest_l2 = n - l1
        return a + b[:rest_l2]
    else:
        return a + b


def load_ranks(save_dir, num_users=5000, cross_domain=False):
    ranks = []
    labels = []
    y = []
    for i in range(num_users):
        save_path = save_dir + 'rank_u{}_y1{}.npz'.format(i, '_cd' if cross_domain else '')
        if os.path.exists(save_path):
            f = np.load(save_path)
            train_rs, train_ls = f['arr_0'], f['arr_1']
            ranks.append(train_rs)
            labels.append(train_ls)
            y.append(1)

        save_path = save_dir + 'rank_u{}_y0{}.npz'.format(i, '_cd' if cross_domain else '')
        if os.path.exists(save_path):
            f = np.load(save_path)
            test_rs, test_ls = f['arr_0'], f['arr_1']
            ranks.append(test_rs)
            labels.append(test_ls)
            y.append(0)

    return ranks, labels, np.asarray(y)


def get_indices_by_labels(sent_labels):
    sent_label_sum = [-np.sum(labels) for labels in sent_labels]
    return np.argsort(sent_label_sum)


def ranks_to_feats(ranks, labels=None, prop=1.0, dim=100, num_words=5000, top_words=5000, shuffle=False,
                   rare=False, relative=False, user_data_ratio=0., heldout_ratio=0., num_users=300):
    if relative or rare:
        assert labels is not None
    X = []

    for i, user_ranks in enumerate(ranks):
        indices = np.arange(len(user_ranks))
        if relative or rare:
            user_labels = labels[i]
            assert len(user_labels) == len(user_ranks)
        else:
            user_labels = None

        r = []

        if 0. < user_data_ratio < 1. and i < num_users:
            l = len(user_ranks)
            for idx in range(l):
                user_ranks[idx] = np.clip(user_ranks[idx], 0, top_words)
                if relative:
                    assert len(user_ranks[idx]) == len(user_labels[idx])
                    user_ranks[idx] = user_ranks[idx] - user_labels[idx]

            train_l = int(l * user_data_ratio)
            train_ranks = user_ranks[:train_l]
            heldout_ranks = user_ranks[train_l:]
            for rank in sample_with_ratio(train_ranks, heldout_ranks, heldout_ratio):
                r.append(rank)
        else:
            if shuffle:
                np.random.seed(None)
                np.random.shuffle(indices)

            if rare:
                indices = get_indices_by_labels(user_labels)

            n = int(len(indices) * prop) + 1 if isinstance(prop, float) else prop
            for idx in indices[:n]:
                user_ranks[idx] = np.clip(user_ranks[idx], 0, top_words)
                if relative:
                    assert len(user_ranks[idx]) == len(user_labels[idx])
                    r.append(user_ranks[idx] - user_labels[idx])
                else:
                    r.append(user_ranks[idx])

        # print i, r
        if isinstance(r[0], int):
            print i
        else:
            r = np.concatenate(r)

        feats = histogram_feats(r, bins=dim, num_words=num_words, top_words=top_words, relative=relative)
        X.append(feats)

    return np.vstack(X)


def user_mi_attack(data_name='reddit', num_exp=5, num_users=5000, dim=100, prop=1.0, user_data_ratio=0.,
                   heldout_ratio=0., num_words=5000, top_words=5000, relative=False, rare=False, norm=True,
                   scale=True, cross_domain=False, rerun=False):

    if data_name == 'reddit':
        result_path = REDDIT_OUTPUT_PATH
    elif data_name == 'sated':
        result_path = SATED_OUTPUT_PATH
    elif data_name == 'dialogs':
        result_path = CORNELL_OUTPUT_PATH
    else:
        raise ValueError(data_name)

    if dim > top_words:
        dim = top_words

    audit_save_path = result_path + 'mi_data_dim{}_prop{}_{}{}.npz'.format(
        dim, prop, num_users, '_cd' if cross_domain else '')

    if not rerun and os.path.exists(audit_save_path):
        f = np.load(audit_save_path)
        X_train, y_train, X_test, y_test = [f['arr_{}'.format(i)] for i in range(4)]
    else:
        save_dir = result_path + 'target_{}{}/'.format(num_users, '_dr' if 0. < user_data_ratio < 1. else '')
        ranks, labels, y_test = load_ranks(save_dir, num_users)
        X_test = ranks_to_feats(ranks, prop=prop, dim=dim, top_words=top_words, user_data_ratio=user_data_ratio,
                                num_words=num_words, labels=labels, rare=rare, relative=relative,
                                heldout_ratio=heldout_ratio)

        X_train, y_train = [], []
        for exp_id in range(num_exp):
            save_dir = result_path + 'shadow_exp{}_{}/'.format(exp_id, num_users)
            ranks, labels, y = load_ranks(save_dir, num_users, cross_domain=cross_domain)
            feats = ranks_to_feats(ranks, prop=prop, dim=dim, top_words=top_words, relative=relative,
                                   num_words=num_words, labels=labels)
            X_train.append(feats)
            y_train.append(y)

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        np.savez(audit_save_path, X_train, y_train, X_test, y_test)

    print X_train.shape, y_train.shape

    if norm:
        normalizer = Normalizer(norm='l2')
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = LinearSVC(verbose=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)

    print classification_report(y_pred=y_pred, y_true=y_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    pres, recs, _, _ = precision_recall_fscore_support(y_test, y_pred)
    pre = pres[1]
    rec = recs[1]

    print 'precision={}, recall={}, acc={}, auc={}'.format(pre, rec, acc, auc)
    return acc, auc, pre, rec
