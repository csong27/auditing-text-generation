import sys

import keras.backend as K
import numpy as np
from keras import Model
from keras.layers import Input, Embedding, CuDNNLSTM, CuDNNGRU, Dropout, Dense
from keras.optimizers import Adam

from data_loader.load_reddit import read_top_user_comments, read_test_comments
from data_loader.load_wiki import load_wiki_by_users, load_wiki_test_data
from helper import DenseTransposeTied, flatten_data, iterate_minibatches, words_to_indices

MODEL_PATH = '/hdd/song/nlp/reddit/model/'
RESULT_PATH = '/hdd/song/nlp/reddit/result/'


def process_test_data(data, vocabs):
    for t in data:
        for i in range(len(t)):
            if t[i] not in vocabs:
                t[i] = '<unk>'


def build_lm_model(emb_h=128, h=128, nh=1, V=5000, maxlen=35, drop_p=0.25, tied=False, rnn_fn='lstm'):
    input_layer = Input((maxlen,))
    emb_layer = Embedding(V, emb_h, mask_zero=False)
    emb_output = emb_layer(input_layer)

    if rnn_fn == 'lstm':
        rnn = CuDNNLSTM
    elif rnn_fn == 'gru':
        rnn = CuDNNGRU
    else:
        raise ValueError(rnn_fn)

    if drop_p > 0.:
        emb_output = Dropout(drop_p)(emb_output)

    lstm_layer = rnn(h, return_sequences=True)(emb_output)
    if drop_p > 0.:
        lstm_layer = Dropout(drop_p)(lstm_layer)

    for _ in range(nh - 1):
        lstm_layer = rnn(h, return_sequences=True)(lstm_layer)
        if drop_p > 0.:
            lstm_layer = Dropout(drop_p)(lstm_layer)

    if tied:
        if emb_h != h:
            raise ValueError('When using the tied flag, nhid must be equal to emsize')
        output = DenseTransposeTied(V, tied_to=emb_layer, activation='linear')(lstm_layer)
    else:
        output = Dense(V, activation='linear')(lstm_layer)
    model = Model(inputs=[input_layer], outputs=[output])
    return model


def train_reddit_lm(num_users=300, num_words=5000, num_epochs=30, maxlen=35, batch_size=20, exp_id=0,
                    h=128, emb_h=256, lr=1e-3, drop_p=0.25, tied=False, nh=1, loo=None, sample_user=False,
                    cross_domain=False, print_every=1000, rnn_fn='lstm'):
    if cross_domain:
        loo = None
        sample_user = True
        user_comments, vocabs = load_wiki_by_users(num_users=num_users, num_words=num_words)
    else:
        user_comments, vocabs = read_top_user_comments(num_users, num_words, sample_user=sample_user)

    train_data = []
    users = sorted(user_comments.keys())

    for i, user in enumerate(users):
        if loo is not None and i == loo:
            print "Leaving {} out".format(i)
            continue
        train_data += user_comments[user]

    train_data = words_to_indices(train_data, vocabs)
    train_data = flatten_data(train_data)

    if cross_domain:
        test_data = load_wiki_test_data()
    else:
        test_data = read_test_comments()

    process_test_data(test_data, vocabs)
    test_data = words_to_indices(test_data, vocabs)
    test_data = flatten_data(test_data)

    n_data = (len(train_data) - 1) // maxlen
    X_train = train_data[:-1][:n_data * maxlen].reshape(-1, maxlen)
    y_train = train_data[1:][:n_data * maxlen].reshape(-1, maxlen)
    print X_train.shape

    n_test_data = (len(test_data) - 1) // maxlen
    X_test = test_data[:-1][:n_test_data * maxlen].reshape(-1, maxlen)
    y_test = test_data[1:][:n_test_data * maxlen].reshape(-1, maxlen)
    print X_test.shape

    model = build_lm_model(emb_h=emb_h, h=h, nh=nh, drop_p=drop_p, V=len(vocabs), tied=tied, maxlen=maxlen,
                           rnn_fn=rnn_fn)

    input_var = K.placeholder((None, maxlen))
    target_var = K.placeholder((None, maxlen))

    prediction = model(input_var)

    loss = K.sparse_categorical_crossentropy(target_var, prediction, from_logits=True)
    loss = K.mean(K.sum(loss, axis=-1))

    optimizer = Adam(lr=lr, clipnorm=5)

    updates = optimizer.get_updates(loss, model.trainable_weights)
    train_fn = K.function([input_var, target_var, K.learning_phase()], [loss], updates=updates)

    pred_fn = K.function([input_var, target_var,  K.learning_phase()], [prediction, loss])

    iteration = 1
    for epoch in range(num_epochs):
        train_batches = 0.
        train_loss = 0.
        train_iters = 0.

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            err = train_fn([inputs, targets, 1])[0]
            train_batches += 1
            train_loss += err
            train_iters += maxlen

            iteration += 1
            if iteration % print_every == 0:
                test_acc = 0.
                test_n = 0.
                test_iters = 0.
                test_loss = 0.
                test_batches = 0.

                for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                    inputs, targets = batch

                    preds, err = pred_fn([inputs, targets, 0])
                    test_loss += err
                    test_iters += maxlen
                    test_batches += 1

                    preds = preds.argmax(axis=-1)
                    test_acc += np.sum(preds.flatten() == targets.flatten())
                    test_n += len(targets.flatten())

                sys.stderr.write("Epoch {}, iteration {}, train loss={:.3f}, train perp={:.3f}, "
                                 "test loss={:.3f}, test perp={:.3f}, "
                                 "test acc={:.3f}%\n".format(epoch, iteration,
                                                             train_loss / train_batches,
                                                             np.exp(train_loss / train_iters),
                                                             test_loss / test_batches,
                                                             np.exp(test_loss / test_iters),
                                                             test_acc / test_n * 100))

    if cross_domain:
        fname = 'wiki_lm{}'.format('' if loo is None else loo)
    else:
        fname = 'reddit_lm{}'.format('' if loo is None else loo)

    if sample_user:
        fname += '_shadow_exp{}_{}'.format(exp_id, rnn_fn)
        np.savez(MODEL_PATH + 'shadow_users{}_{}_{}_{}.npz'.format(exp_id, rnn_fn, num_users,
                                                                   'cd' if cross_domain else ''), users)

    model.save(MODEL_PATH + '{}_{}.h5'.format(fname, num_users))


if __name__ == '__main__':
    train_reddit_lm()
