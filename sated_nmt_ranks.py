import os
import sys
from collections import Counter, defaultdict
from itertools import chain

import keras.backend as K
import numpy as np
import scipy.stats as ss
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

from helper import flatten_data
from data_loader.load_sated import process_texts, process_vocabs, load_texts, load_users, load_sated_data_by_user, \
    SATED_TRAIN_USER, SATED_TRAIN_FR, SATED_TRAIN_ENG, read_europarl_file, EUROPARL_FREN_FR, EUROPARL_FREN_EN
from sated_nmt import build_nmt_model, words_to_indices, MODEL_PATH, OUTPUT_PATH


def load_cross_domain_shadow_user_data(train_users, num_users=100, num_words=10000, num_data_per_user=150, seed=12345):
    src_texts = read_europarl_file(EUROPARL_FREN_EN, num_users * num_data_per_user * 2)
    trg_texts = read_europarl_file(EUROPARL_FREN_FR, num_users * num_data_per_user * 2)

    all_users = np.arange(num_users * 2)
    test_users = np.setdiff1d(all_users, train_users)

    user_src_texts = defaultdict(list)
    user_trg_texts = defaultdict(list)

    test_user_src_texts = defaultdict(list)
    test_user_trg_texts = defaultdict(list)

    for u in train_users:
        user_src_texts[u] = src_texts[u * num_data_per_user: (u + 1) * num_data_per_user]
        user_trg_texts[u] = trg_texts[u * num_data_per_user: (u + 1) * num_data_per_user]

    for u in test_users:
        test_user_src_texts[u] = src_texts[u * num_data_per_user: (u + 1) * num_data_per_user]
        test_user_trg_texts[u] = trg_texts[u * num_data_per_user: (u + 1) * num_data_per_user]

    src_words = []
    trg_words = []
    for u in train_users:
        src_words += list(chain(*user_src_texts[u]))
        trg_words += list(chain(*user_trg_texts[u]))

    src_vocabs = process_vocabs(src_words, num_words)
    trg_vocabs = process_vocabs(trg_words, num_words)

    for u in train_users:
        process_texts(user_src_texts[u], src_vocabs)
        process_texts(user_trg_texts[u], trg_vocabs)

    for u in test_users:
        process_texts(test_user_src_texts[u], src_vocabs)
        process_texts(test_user_trg_texts[u], trg_vocabs)

    src_words = []
    trg_words = []

    for u in train_users:
        src_words += list(chain(*user_src_texts[u]))
        trg_words += list(chain(*user_trg_texts[u]))

    src_vocabs = process_vocabs(src_words, None)
    trg_vocabs = process_vocabs(trg_words, None)

    return user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts,  src_vocabs, trg_vocabs


def load_train_users_heldout_data(train_users, src_vocabs, trg_vocabs, user_data_ratio=0.5):
    src_users = load_users(SATED_TRAIN_USER)
    train_src_texts = load_texts(SATED_TRAIN_ENG)
    train_trg_texts = load_texts(SATED_TRAIN_FR)

    user_src_texts = defaultdict(list)
    user_trg_texts = defaultdict(list)

    for u, s, t in zip(src_users, train_src_texts, train_trg_texts):
        if u in train_users:
            user_src_texts[u].append(s)
            user_trg_texts[u].append(t)

    assert 0. < user_data_ratio < 1.
    # held out some fraction of data for testing
    for u in user_src_texts:
        l = len(user_src_texts[u])
        l = int(l * user_data_ratio)
        user_src_texts[u] = user_src_texts[u][l:]
        user_trg_texts[u] = user_trg_texts[u][l:]

    for u in train_users:
        process_texts(user_src_texts[u], src_vocabs)
        process_texts(user_trg_texts[u], trg_vocabs)

    return user_src_texts, user_trg_texts


def load_shadow_user_data(train_users, num_users=100, num_words=10000, seed=12345):
    src_users = load_users(SATED_TRAIN_USER)
    train_src_texts = load_texts(SATED_TRAIN_ENG)
    train_trg_texts = load_texts(SATED_TRAIN_FR)

    user_counter = Counter(src_users)
    all_users = [tup[0] for tup in user_counter.most_common()]
    np.random.seed(seed)
    np.random.shuffle(all_users)
    np.random.seed(None)

    attacker_users = all_users[num_users * 2: num_users * 4]
    test_users = np.setdiff1d(attacker_users, train_users)
    print len(train_users), len(test_users)

    user_src_texts = defaultdict(list)
    user_trg_texts = defaultdict(list)

    test_user_src_texts = defaultdict(list)
    test_user_trg_texts = defaultdict(list)

    for u, s, t in zip(src_users, train_src_texts, train_trg_texts):
        if u in train_users:
            user_src_texts[u].append(s)
            user_trg_texts[u].append(t)
        if u in test_users:
            test_user_src_texts[u].append(s)
            test_user_trg_texts[u].append(t)

    src_words = []
    trg_words = []
    for u in train_users:
        src_words += list(chain(*user_src_texts[u]))
        trg_words += list(chain(*user_trg_texts[u]))

    src_vocabs = process_vocabs(src_words, num_words)
    trg_vocabs = process_vocabs(trg_words, num_words)

    for u in train_users:
        process_texts(user_src_texts[u], src_vocabs)
        process_texts(user_trg_texts[u], trg_vocabs)

    for u in test_users:
        process_texts(test_user_src_texts[u], src_vocabs)
        process_texts(test_user_trg_texts[u], trg_vocabs)

    src_words = []
    trg_words = []

    for u in train_users:
        src_words += list(chain(*user_src_texts[u]))
        trg_words += list(chain(*user_trg_texts[u]))

    src_vocabs = process_vocabs(src_words, None)
    trg_vocabs = process_vocabs(trg_words, None)

    return user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts,  src_vocabs, trg_vocabs


def rank_lists(lists):
    ranks = np.empty_like(lists)
    for i, l in enumerate(lists):
        ranks[i] = ss.rankdata(l, method='min') - 1
    return ranks


def get_ranks(user_src_data, user_trg_data, pred_fn, save_probs=False):
    indices = np.arange(len(user_src_data))

    ranks = []
    labels = []
    probs = []
    for idx in indices:
        src_text = np.asarray(user_src_data[idx], dtype=np.float32).reshape(1, -1)
        trg_text = np.asarray(user_trg_data[idx], dtype=np.float32)
        trg_input = trg_text[:-1].reshape(1, -1)
        trg_label = trg_text[1:].reshape(1, -1)

        prob = pred_fn([src_text, trg_input, trg_label, 0])[0][0]
        if save_probs:
            probs.append(prob)

        # all_ranks = np.argsort(-prob, axis=-1).argsort(axis=-1)

        all_ranks = rank_lists(-prob)
        sent_ranks = all_ranks[np.arange(len(all_ranks)), trg_label.flatten().astype(int)]

        ranks.append(sent_ranks)
        labels.append(trg_label.flatten())

    if save_probs:
        return probs

    return ranks, labels


def save_users_rank_results(users, user_src_texts, user_trg_texts, src_vocabs, trg_vocabs, prob_fn, save_dir,
                            member_label=1, cross_domain=False, save_probs=False, mask=False, rerun=False):
    for i, u in enumerate(users):
        save_path = save_dir + 'rank_u{}_y{}{}.npz'.format(i, member_label, '_cd' if cross_domain else '')
        prob_path = save_dir + 'prob_u{}_y{}{}.npz'.format(i, member_label, '_cd' if cross_domain else '')

        if os.path.exists(save_path) and not save_probs and not rerun:
            continue

        user_src_data = words_to_indices(user_src_texts[u], src_vocabs, mask=mask)
        user_trg_data = words_to_indices(user_trg_texts[u], trg_vocabs, mask=mask)

        rtn = get_ranks(user_src_data, user_trg_data, prob_fn, save_probs=save_probs)

        if save_probs:
            probs = rtn
            np.savez(prob_path, probs)
        else:
            ranks, labels = rtn[0], rtn[1]
            np.savez(save_path, ranks, labels)

        if (i + 1) % 500 == 0:
            sys.stderr.write('Finishing saving ranks for {} users'.format(i + 1))


def histogram_feats(ranks, bins=100, num_words=5000):
    feats, _ = np.histogram(ranks, bins=bins, normed=False, range=(0, num_words))
    return feats


def get_shadow_ranks(exp_id=0, num_users=200, num_words=5000, mask=False, h=128, emb_h=128, save_probs=False,
                     tied=False, cross_domain=False, rnn_fn='lstm', rerun=False):
    shadow_user_path = 'shadow_users{}_{}_{}_{}.npz'.format(exp_id, rnn_fn, num_users, 'cd' if cross_domain else '')
    shadow_train_users = np.load(MODEL_PATH + shadow_user_path)['arr_0']
    shadow_train_users = list(shadow_train_users)

    print shadow_user_path

    save_dir = OUTPUT_PATH + 'shadow_exp{}_{}/'.format(exp_id, num_users)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if cross_domain:
        user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts, src_vocabs, trg_vocabs \
            = load_cross_domain_shadow_user_data(shadow_train_users, num_users, num_words)
    else:
        user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts, src_vocabs, trg_vocabs \
            = load_shadow_user_data(shadow_train_users, num_users, num_words)
    shadow_test_users = sorted(test_user_src_texts.keys())

    model_path = '{}_shadow_exp{}_{}_{}.h5'.format('europal_nmt' if cross_domain else 'sated_nmt',
                                                   exp_id, rnn_fn, num_users)

    model = build_nmt_model(Vs=num_words, Vt=num_words, mask=mask, drop_p=0., h=h, demb=emb_h, tied=tied, rnn_fn=rnn_fn)
    model.load_weights(MODEL_PATH + model_path)

    src_input_var, trg_input_var = model.inputs
    prediction = model.output
    trg_label_var = K.placeholder((None, None), dtype='float32')

    prediction = K.softmax(prediction)
    prob_fn = K.function([src_input_var, trg_input_var, trg_label_var, K.learning_phase()], [prediction])

    save_users_rank_results(users=shadow_train_users, save_probs=save_probs, rerun=rerun, mask=mask,
                            user_src_texts=user_src_texts, user_trg_texts=user_trg_texts,
                            src_vocabs=src_vocabs, trg_vocabs=trg_vocabs, cross_domain=cross_domain,
                            prob_fn=prob_fn, save_dir=save_dir, member_label=1)
    save_users_rank_results(users=shadow_test_users, save_probs=save_probs, rerun=rerun, mask=mask,
                            user_src_texts=test_user_src_texts, user_trg_texts=test_user_trg_texts,
                            src_vocabs=src_vocabs, trg_vocabs=trg_vocabs, cross_domain=cross_domain,
                            prob_fn=prob_fn, save_dir=save_dir, member_label=0)


def get_target_ranks(num_users=200, num_words=5000, mask=False, h=128, emb_h=128, user_data_ratio=0.,
                     tied=False, save_probs=False):
    user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts, src_vocabs, trg_vocabs \
        = load_sated_data_by_user(num_users, num_words, test_on_user=True, user_data_ratio=user_data_ratio)

    train_users = sorted(user_src_texts.keys())
    test_users = sorted(test_user_src_texts.keys())

    save_dir = OUTPUT_PATH + 'target_{}{}/'.format(num_users, '_dr' if 0. < user_data_ratio < 1. else '')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = 'sated_nmt'.format(num_users)

    if 0. < user_data_ratio < 1.:
        model_path += '_dr{}'.format(user_data_ratio)
        heldout_src_texts, heldout_trg_texts = load_train_users_heldout_data(train_users, src_vocabs, trg_vocabs)
        for u in train_users:
            user_src_texts[u] += heldout_src_texts[u]
            user_trg_texts[u] += heldout_trg_texts[u]

    model = build_nmt_model(Vs=num_words, Vt=num_words, mask=mask, drop_p=0., h=h, demb=emb_h, tied=tied)
    model.load_weights(MODEL_PATH + '{}_{}.h5'.format(model_path, num_users))

    src_input_var, trg_input_var = model.inputs
    prediction = model.output
    trg_label_var = K.placeholder((None, None), dtype='float32')

    prediction = K.softmax(prediction)
    prob_fn = K.function([src_input_var, trg_input_var, trg_label_var, K.learning_phase()], [prediction])

    save_users_rank_results(users=train_users, save_probs=save_probs,
                            user_src_texts=user_src_texts, user_trg_texts=user_trg_texts,
                            src_vocabs=src_vocabs, trg_vocabs=trg_vocabs, cross_domain=False,
                            prob_fn=prob_fn, save_dir=save_dir, member_label=1)
    save_users_rank_results(users=test_users, save_probs=save_probs,
                            user_src_texts=test_user_src_texts, user_trg_texts=test_user_trg_texts,
                            src_vocabs=src_vocabs, trg_vocabs=trg_vocabs, cross_domain=False,
                            prob_fn=prob_fn, save_dir=save_dir, member_label=0)


def ranks_to_feats(ranks, prop=1.0, dim=100, num_words=5000, shuffle=True):
    X = []
    i = 0
    for user_ranks in ranks:
        indices = np.arange(len(user_ranks))
        if shuffle:
            np.random.shuffle(indices)
        n = int(len(indices) * prop)
        r = []
        for idx in indices[:n]:
            r.append(user_ranks[idx])
        r = np.concatenate(r)
        # print i, np.average(r)
        feats = histogram_feats(r, bins=dim, num_words=num_words)
        X.append(feats)
        i += 1
    # quit()
    return np.vstack(X)


def user_mi_attack(num_exp=10, dim=100, prop=1.0, num_words=5000, cross_domain=True):
    f = np.load(OUTPUT_PATH + 'target_user_ranks.npz')
    X_test = ranks_to_feats(f['arr_0'], prop=prop, dim=dim, num_words=num_words)
    y_test = f['arr_1']

    X = []
    y = []
    for exp_id in range(num_exp):
        f = np.load(OUTPUT_PATH + 'shadow_user_ranks_{}{}.npz'.format(exp_id, '_cd' if cross_domain else ''))
        feats = ranks_to_feats(f['arr_0'], prop=prop, dim=dim, num_words=num_words)
        X.append(feats)
        y.append(f['arr_1'])

    X_train = np.vstack(X)
    y_train = np.concatenate(y)

    print X_train.shape, y_train.shape
    normalizer = Normalizer(norm='l1')
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.fit_transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # clf = RandomForestClassifier(n_estimators=20)
    clf = SVC()
    clf.fit(X_train, y_train)

    y_score = clf.decision_function(X_test)  # [:, 1]
    y_pred = clf.predict(X_test)

    print classification_report(y_pred=y_pred, y_true=y_test)
    print 'ACC:', accuracy_score(y_test, y_pred)
    print 'AUC:', roc_auc_score(y_test, y_score)


def test_vocab():
    user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts, src_vocabs, trg_vocabs \
        = load_sated_data_by_user(300, 5000, test_on_user=True, user_data_ratio=0.)
    train_data = []
    test_data = []

    for user in user_trg_texts:
        train_data += user_trg_texts[user]
    train_data = words_to_indices(train_data, trg_vocabs)
    train_data = flatten_data(train_data)

    for user in test_user_trg_texts:
        test_data += test_user_trg_texts[user]
    test_data = words_to_indices(test_data, trg_vocabs)
    test_data = flatten_data(test_data)

    n = float(len(train_data))
    b = np.sum(train_data >= 1000) / n
    print 1 - b, b, n

    n = float(len(test_data))
    b = np.sum(test_data >= 1000) / n
    print 1 - b, b, n


if __name__ == '__main__':
    get_target_ranks(num_users=300, save_probs=False)
    for i in range(10):
        get_shadow_ranks(exp_id=i, num_users=300, cross_domain=False, rerun=True)
