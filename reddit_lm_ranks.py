import codecs
import os
import sys
from collections import defaultdict

import keras.backend as K
import numpy as np
from nltk.tokenize import word_tokenize

from data_loader.load_reddit import read_top_users, REDDIT_PROCESSED_PATH, build_vocab, read_top_user_comments, \
    remove_puncs
from data_loader.load_wiki import WIKI_TRAIN_PATH, load_wiki_lines
from reddit_lm import words_to_indices, MODEL_PATH, RESULT_PATH, build_lm_model


def load_cross_domain_shadow_user_data(train_users, num_users=100, num_data_per_user=100, num_words=5000):
    train_data = load_wiki_lines(WIKI_TRAIN_PATH, num_lines=2 * num_users * num_data_per_user)

    # l = len(train_data)
    # num_data_per_user = l // (num_users * 2)
    print "Splitting data to {} users, each has {} texts".format(num_users * 2, num_data_per_user)

    all_users = np.arange(num_users * 2)
    train_users = [int(u) for u in train_users]
    test_users = np.setdiff1d(all_users, train_users)
    print len(test_users)

    train_user_comments = defaultdict(list)
    test_user_comments = defaultdict(list)

    all_words = []

    for u in train_users:
        data = train_data[u * num_data_per_user:  (u + 1) * num_data_per_user]
        for words in data:
            all_words += words
        train_user_comments[str(u)] = data

    for u in test_users:
        data = train_data[u * num_data_per_user:  (u + 1) * num_data_per_user]
        test_user_comments[str(u)] = data

    vocabs = build_vocab(all_words, num_words + 1)

    all_words = []
    for user in train_user_comments:
        comments = train_user_comments[user]
        for i in range(len(comments)):
            comment = comments[i]
            for j in range(len(comment)):
                word = comment[j]
                if word not in vocabs:
                    comment[j] = '<unk>'
            all_words += comment

    vocabs = build_vocab(all_words, None)

    for user in test_user_comments:
        comments = test_user_comments[user]
        for i in range(len(comments)):
            comment = comments[i]
            for j in range(len(comment)):
                word = comment[j]
                if word not in vocabs:
                    comment[j] = '<unk>'

    return train_user_comments, test_user_comments, vocabs


def load_shadow_user_data(train_users, num_users=100, num_words=10000):
    all_users = read_top_users(num_users * 4)
    attacker_users = all_users[num_users * 2: num_users * 4]
    test_users = np.setdiff1d(attacker_users, train_users)

    train_user_comments = defaultdict(list)
    test_user_comments = defaultdict(list)
    all_words = []
    for user in train_users:
        filename = os.path.join(REDDIT_PROCESSED_PATH, user)
        with codecs.open(filename, encoding='utf-8') as f:
            for line in f:
                data = line.replace('/url/', '').replace('\n', '').split() + ['<eos>']
                if len(data) == 1:
                    continue

                train_user_comments[user].append(data)
                all_words += data

    for user in test_users:
        filename = os.path.join(REDDIT_PROCESSED_PATH, user)
        with codecs.open(filename, encoding='utf-8') as f:
            for line in f:
                data = line.replace('/url/', '').replace('\n', '').split() + ['<eos>']
                if len(data) == 1:
                    continue

                test_user_comments[user].append(data)

    vocabs = build_vocab(all_words, num_words)

    all_words = []
    for user in train_user_comments:
        comments = train_user_comments[user]
        for i in range(len(comments)):
            comment = comments[i]
            for j in range(len(comment)):
                word = comment[j]
                if word not in vocabs:
                    comment[j] = '<unk>'
            all_words += comment

    vocabs = build_vocab(all_words, None)

    for user in test_user_comments:
        comments = test_user_comments[user]
        for i in range(len(comments)):
            comment = comments[i]
            for j in range(len(comment)):
                word = comment[j]
                if word not in vocabs:
                    comment[j] = '<unk>'

    return train_user_comments, test_user_comments, vocabs


def group_texts_by_len(texts, bs=20):
    # Bucket samples by source sentence length
    buckets = defaultdict(list)
    batches = []
    for t in texts:
        if len(t) < 2:
            continue
        buckets[len(t)].append(t)

    for l, bucket in buckets.items():
        num_batches = int(np.ceil(len(bucket) * 1.0 / bs))
        for i in range(num_batches):
            cur_batch_size = bs if i < num_batches - 1 else len(bucket) - bs * i
            batches.append([bucket[i * bs + j] for j in range(cur_batch_size)])

    return batches


def group_texts_by_maxlen(user_data, maxlen, bs):
    flatten_data = np.asarray([w for t in user_data for w in t]).astype(np.int32)
    if len(flatten_data) - 1 < maxlen:
        maxlen = len(flatten_data) - 1

    n_data = (len(flatten_data) - 1) // maxlen

    inputs_data = flatten_data[:-1][:n_data * maxlen].reshape(n_data, maxlen)
    targets_data = flatten_data[1:][:n_data * maxlen].reshape(n_data, maxlen)

    cumsum = 0
    lengthes = [0]
    for i, t in enumerate(user_data):
        if i == 0:
            cumsum += len(t) - 1
        else:
            cumsum += len(t)
        lengthes.append(cumsum)
        if cumsum >= n_data * maxlen:
            lengthes[-1] = n_data * maxlen
            break

    n_batches = n_data // bs + 1
    batches = []
    for i in range(n_batches):
        if i * bs >= len(inputs_data):
            break
        inputs = inputs_data[i * bs: (i + 1) * bs]
        targets = targets_data[i * bs: (i + 1) * bs]
        batches.append((inputs, targets))

    return batches, lengthes


def get_bigram_probs(texts, pred_fn):
    bs, l = texts.shape
    for i in range(l):
        pass


def get_ranks_labels_by_batch(user_data, pred_fn, maxlen=35, bs=20, save_probs=False, trg_prob_only=False):
    batches, lengthes = group_texts_by_maxlen(user_data, maxlen, bs)
    _ranks = []
    _labels = []
    _probs = []

    for inputs, targets in batches:
        bs, l = targets.shape
        # print inputs.shape, targets.shape
        probs = pred_fn([inputs, targets, 0])[0]
        # all_ranks = np.argsort(-probs, axis=-1).argsort(axis=-1)  # 20 x len x words
        # all_ranks = all_ranks.reshape(bs * l, -1)
        probs = probs.reshape(bs * l, -1)
        if save_probs:
            if trg_prob_only:
                probs = probs[np.arange(bs * l), targets.flatten().astype(int)]
            _probs.append(probs)
        else:
            all_ranks = rank_lists(-probs)
            targets_ranks = all_ranks[np.arange(bs * l), targets.flatten().astype(int)]
            targets_ranks = targets_ranks.reshape(bs, l)
            assert targets.shape == targets_ranks.shape
            # ranks += [r for r in targets_ranks]
            # labels += [t for t in targets]
            _ranks.append(targets_ranks.flatten())
            _labels.append(targets.flatten())

    if save_probs:
        all_probs = np.concatenate(_probs) if trg_prob_only else np.vstack(_probs)
    else:
        all_ranks = np.concatenate(_ranks)
        all_labels = np.concatenate(_labels)
        assert lengthes[-1] == len(all_ranks) == len(all_labels)

    ranks = []
    labels = []
    probs = []

    for b, e in zip(lengthes[:-1], lengthes[1:]):
        if save_probs:
            probs.append(all_probs[b: e])
        else:
            ranks.append(all_ranks[b: e])
            labels.append(all_labels[b: e])

    if save_probs:
        return probs
    else:
        return ranks, labels


def rank_lists(lists):
    # ranks = np.empty_like(lists)
    # for i, l in enumerate(lists):
    #     ranks[i] = ss.rankdata(l, method='min') - 1

    temp = np.argsort(lists, axis=-1)
    ranks = np.empty_like(temp)
    ranks[np.arange(len(temp))[:, None], temp] = np.arange(temp.shape[1])
    return ranks


def save_users_rank_results(users, user_comments, vocabs, prob_fn, save_dir, member_label=1,
                            cross_domain=False, save_probs=False, trg_prob_only=False, rerun=False):
    for i, u in enumerate(users):
        save_path = save_dir + 'rank_u{}_y{}{}.npz'.format(i, member_label, '_cd' if cross_domain else '')
        prob_path = save_dir + 'prob_u{}_y{}{}.npz'.format(i, member_label, '_cd' if cross_domain else '')

        if os.path.exists(save_path) and not save_probs and not rerun:
            continue

        user_data = words_to_indices(user_comments[u], vocabs)
        rtn = get_ranks_labels_by_batch(user_data, prob_fn, save_probs=save_probs, trg_prob_only=trg_prob_only)

        if save_probs:
            probs = rtn
            np.savez(prob_path, probs)
        else:
            ranks, labels = rtn[0], rtn[1]
            np.savez(save_path, ranks, labels)

        if (i + 1) % 500 == 0:
            sys.stderr.write('Finishing saving ranks for {} users\n'.format(i + 1))


def get_shadow_ranks(exp_id=0, num_users=100, num_words=5000, cross_domain=False, h=128, emb_h=256, rerun=False,
                     rnn_fn='lstm'):
    shadow_user_path = 'shadow_users{}_{}_{}_{}.npz'.format(exp_id, rnn_fn, num_users, 'cd' if cross_domain else '')
    shadow_train_users = np.load(MODEL_PATH + shadow_user_path)['arr_0']
    shadow_train_users = list(shadow_train_users)

    print len(shadow_train_users)
    print shadow_user_path

    save_dir = RESULT_PATH + 'shadow_exp{}_{}/'.format(exp_id, num_users)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if cross_domain:
        train_user_comments, test_user_comments, vocabs =\
            load_cross_domain_shadow_user_data(shadow_train_users, num_users=num_users, num_words=num_words)
    else:
        train_user_comments, test_user_comments, vocabs =\
            load_shadow_user_data(shadow_train_users, num_users, num_words)
    shadow_test_users = sorted(test_user_comments.keys())

    if cross_domain:
        model_path = 'wiki_lm_shadow_exp{}_{}_{}.h5'.format(exp_id, rnn_fn, num_users)
    else:
        model_path = 'reddit_lm_shadow_exp{}_{}_{}.h5'.format(exp_id, rnn_fn, num_users)

    model = build_lm_model(V=num_words, drop_p=0., h=h, emb_h=emb_h, rnn_fn=rnn_fn)
    model.load_weights(MODEL_PATH + model_path)

    input_var = K.placeholder((None, None))
    prediction = model(input_var)
    label_var = K.placeholder((None, None), dtype='float32')
    prediction = K.softmax(prediction)
    prob_fn = K.function([input_var, label_var, K.learning_phase()], [prediction])

    save_users_rank_results(shadow_train_users, train_user_comments, cross_domain=cross_domain, rerun=rerun,
                            vocabs=vocabs, prob_fn=prob_fn, save_dir=save_dir, member_label=1)
    save_users_rank_results(shadow_test_users, test_user_comments, cross_domain=cross_domain, rerun=rerun,
                            vocabs=vocabs, prob_fn=prob_fn, save_dir=save_dir, member_label=0)


def get_target_ranks(num_users=100, num_words=5000, h=128, emb_h=256, rerun=False):
    users = read_top_users(num_users * 2)
    train_users = users[:num_users]
    test_users = users[num_users:]
    train_user_comments, vocabs = read_top_user_comments(num_users, num_words, top_users=train_users)
    test_user_comments, _ = read_top_user_comments(num_users, num_words, top_users=test_users, vocabs=vocabs)

    save_dir = RESULT_PATH + 'target_{}/'.format(num_users)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = 'reddit_lm_{}.h5'.format(num_users)
    model = build_lm_model(V=num_words, drop_p=0., h=h, emb_h=emb_h)
    model.load_weights(MODEL_PATH + model_path)

    input_var = K.placeholder((None, None))
    prediction = model(input_var)
    label_var = K.placeholder((None, None), dtype='float32')
    prediction = K.softmax(prediction)
    prob_fn = K.function([input_var, label_var, K.learning_phase()], [prediction])

    save_users_rank_results(train_users, train_user_comments, cross_domain=False, rerun=rerun,
                            vocabs=vocabs, prob_fn=prob_fn, save_dir=save_dir, member_label=1)
    save_users_rank_results(test_users, test_user_comments, cross_domain=False, rerun=rerun,
                            vocabs=vocabs, prob_fn=prob_fn, save_dir=save_dir, member_label=0)


def read_translated_comments(users, vocabs, multi_step=False, trans='yandex'):
    user_comments = defaultdict(list)
    for user in users:
        filename = './translate/{}_{}_{}.txt'.format(user, trans, 'multi' if multi_step else 'two')
        with codecs.open(filename, encoding='utf-8') as f:
            for line in f:
                data = line.replace('\n', '')
                data = word_tokenize(data)
                data = remove_puncs(data)

                if len(data) == 1:
                    print user, data
                    continue

                user_comments[user].append(data + ['<eos>'])

    for user in user_comments:
        comments = user_comments[user]
        for i in range(len(comments)):
            comment = comments[i]
            for j in range(len(comment)):
                word = comment[j]
                if word not in vocabs:
                    comment[j] = '<unk>'

    return user_comments


def get_translated_rank(num_users=100, num_words=5000, h=128, emb_h=128):
    users = read_top_users(num_users * 2, min_count=0)
    train_users = users[:num_users]
    test_users = users[num_users:]

    _, vocabs = read_top_user_comments(num_users, num_words, top_users=train_users)
    train_user_comments = read_translated_comments(train_users, vocabs)
    test_user_comments = read_translated_comments(test_users, vocabs)

    save_dir = RESULT_PATH + 'target_yandex_{}/'.format(num_users)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = 'reddit_lm_{}.h5'.format(num_users)
    model = build_lm_model(V=num_words, drop_p=0., h=h, emb_h=emb_h)
    model.load_weights(MODEL_PATH + model_path)

    input_var = K.placeholder((None, None))
    prediction = model(input_var)
    label_var = K.placeholder((None, None), dtype='float32')
    prediction = K.softmax(prediction)
    prob_fn = K.function([input_var, label_var, K.learning_phase()], [prediction])

    save_users_rank_results(train_users, train_user_comments, cross_domain=False,
                            vocabs=vocabs, prob_fn=prob_fn, save_dir=save_dir, member_label=1)
    save_users_rank_results(test_users, test_user_comments, cross_domain=False,
                            vocabs=vocabs, prob_fn=prob_fn, save_dir=save_dir, member_label=0)
