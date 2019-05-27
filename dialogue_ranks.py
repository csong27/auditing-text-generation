import os
from collections import Counter, defaultdict
from itertools import chain

import keras.backend as K
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from dialogue import build_dialogue_model, words_to_indices, MODEL_PATH, OUTPUT_PATH
from helper import flatten_data
from data_loader.load_cornell_movie import load_extracted_cornell_movie, process_texts, process_vocabs, \
    load_cornell_movie_by_user, load_extracted_ubuntu
from sated_nmt_ranks import save_users_rank_results


def load_cross_domain_shadow_user_data(train_users, num_users=100, num_words=5000, num_data_per_user=200):
    src_texts, trg_texts = load_extracted_ubuntu(num_users * num_data_per_user * 2)

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


def load_shadow_user_data(train_users, num_users=100, num_words=10000, min_count=20):
    train_data, dev_data, test_data = load_extracted_cornell_movie(dev_size=5000, test_size=5000)
    train_src_texts, train_trg_texts, src_users, _ = train_data

    user_counter = Counter(src_users)
    all_users = np.asarray([tup[0] for tup in user_counter.most_common() if tup[1] >= min_count])
    print 'Loaded {} users'.format(len(all_users))

    np.random.seed(12345)
    np.random.shuffle(all_users)
    np.random.seed(None)

    attacker_users = all_users[num_users * 2: num_users * 4]
    test_users = np.setdiff1d(attacker_users, train_users)

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


def get_ranks(user_src_data, user_trg_data, pred_fn):
    indices = np.arange(len(user_src_data))

    ranks = []
    for idx in indices:
        src_text = np.asarray(user_src_data[idx], dtype=np.float32).reshape(1, -1)
        trg_text = np.asarray(user_trg_data[idx], dtype=np.float32)
        trg_input = trg_text[:-1].reshape(1, -1)
        trg_label = trg_text[1:].reshape(1, -1)

        prob = pred_fn([src_text, trg_input, trg_label, 0])[0][0]
        sent_ranks = []
        for p, t in zip(prob, trg_label.flatten()):
            t = int(t)
            rank = (-p).argsort().argsort()[t]
            sent_ranks.append(rank)
        ranks.append(sent_ranks)
    return ranks


def get_shadow_ranks(exp_id=0, num_users=200, num_words=5000, mask=False, cross_domain=False, rnn_fn='lstm',
                     h=128, emb_h=128, rerun=False):
    shadow_user_path = 'shadow_users{}_{}_{}_{}.npz'.format(exp_id, rnn_fn, num_users, 'cd' if cross_domain else '')
    shadow_train_users = np.load(MODEL_PATH + shadow_user_path)['arr_0']
    shadow_train_users = list(shadow_train_users)
    print shadow_user_path, shadow_train_users

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

    model_path = '{}_shadow_exp{}_{}_{}.h5'.format('ubuntu_dialog' if cross_domain else 'cornell_movie_dialog',
                                                    exp_id, rnn_fn, num_users)

    model = build_dialogue_model(Vs=num_words, Vt=num_words, mask=mask, drop_p=0., h=h, demb=emb_h, rnn_fn=rnn_fn)
    model.load_weights(MODEL_PATH + model_path)

    src_input_var, trg_input_var = model.inputs
    prediction = model.output
    trg_label_var = K.placeholder((None, None), dtype='float32')

    prediction = K.softmax(prediction)
    prob_fn = K.function([src_input_var, trg_input_var, trg_label_var, K.learning_phase()], [prediction])

    save_users_rank_results(users=shadow_train_users, rerun=rerun,
                            user_src_texts=user_src_texts, user_trg_texts=user_trg_texts,
                            src_vocabs=src_vocabs, trg_vocabs=trg_vocabs, cross_domain=cross_domain,
                            prob_fn=prob_fn, save_dir=save_dir, member_label=1)
    save_users_rank_results(users=shadow_test_users, rerun=rerun,
                            user_src_texts=test_user_src_texts, user_trg_texts=test_user_trg_texts,
                            src_vocabs=src_vocabs, trg_vocabs=trg_vocabs, cross_domain=cross_domain,
                            prob_fn=prob_fn, save_dir=save_dir, member_label=0)


def load_train_users_heldout_data(train_users, src_vocabs, trg_vocabs, user_data_ratio=0.5):
    train_data, dev_data, test_data = load_extracted_cornell_movie(dev_size=5000, test_size=5000)
    train_src_texts, train_trg_texts, src_users, _ = train_data

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


def get_target_ranks(num_users=200, num_words=5000, mask=False, user_data_ratio=0., save_probs=False):
    user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts, src_vocabs, trg_vocabs \
        = load_cornell_movie_by_user(num_users, num_words, test_on_user=True, user_data_ratio=user_data_ratio)

    train_users = sorted(user_src_texts.keys())
    test_users = sorted(test_user_src_texts.keys())

    save_dir = OUTPUT_PATH + 'target_{}{}/'.format(num_users, '_dr' if 0. < user_data_ratio < 1. else '')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = 'cornell_movie_dialog'

    if 0. < user_data_ratio < 1.:
        model_path += '_dr{}'.format(user_data_ratio)
        heldout_src_texts, heldout_trg_texts = load_train_users_heldout_data(train_users, src_vocabs, trg_vocabs)
        for u in train_users:
            user_src_texts[u] += heldout_src_texts[u]
            user_trg_texts[u] += heldout_trg_texts[u]

    model = build_dialogue_model(Vs=num_words, Vt=num_words, mask=mask, drop_p=0.)
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


if __name__ == '__main__':
    get_target_ranks(num_users=300, save_probs=False)
