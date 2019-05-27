from keras import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, CuDNNLSTM, CuDNNGRU
from helper import DenseTransposeTied
from keras.optimizers import Adam

import keras.backend as K
import copy

from collections import defaultdict
from data_loader.load_cornell_movie import load_ubuntu_by_user, load_cornell_movie_by_user
from sated_nmt import beam_search, bleu_score

import pprint
import numpy as np

MODEL_PATH = '/hdd/song/nlp/cornell_movie_dialogs_corpus/model/'
OUTPUT_PATH = '/hdd/song/nlp/cornell_movie_dialogs_corpus/output/'


def group_texts_by_len(src_texts, trg_texts, bs=20):
    print("Bucketing batches")
    # Bucket samples by source sentence length
    buckets = defaultdict(list)
    batches = []
    for src, trg in zip(src_texts, trg_texts):
        buckets[len(src)].append((src, trg))

    for src_len, bucket in buckets.items():
        np.random.shuffle(bucket)
        num_batches = int(np.ceil(len(bucket) * 1.0 / bs))
        for i in range(num_batches):
            cur_batch_size = bs if i < num_batches - 1 else len(bucket) - bs * i
            batches.append(([bucket[i * bs + j][0] for j in range(cur_batch_size)],
                            [bucket[i * bs + j][1] for j in range(cur_batch_size)]))
    return batches


def build_dialogue_model(Vs, Vt, demb=128, h=128, drop_p=0.5, tied=True, mask=True, training=None, rnn_fn='lstm'):
    if rnn_fn == 'lstm':
        rnn = LSTM if mask else CuDNNLSTM
    elif rnn_fn == 'gru':
        rnn = LSTM if mask else CuDNNGRU
    else:
        raise ValueError(rnn_fn)

    # build encoder
    encoder_input = Input((None,), dtype='float32', name='encoder_input')
    if mask:
        encoder_emb_layer = Embedding(Vs + 1, demb, mask_zero=True, name='encoder_emb')
    else:
        encoder_emb_layer = Embedding(Vs, demb, mask_zero=False, name='encoder_emb')

    encoder_emb = encoder_emb_layer(encoder_input)

    if drop_p > 0.:
        encoder_emb = Dropout(drop_p)(encoder_emb, training=training)

    encoder_rnn = rnn(h, return_sequences=True, return_state=True, name='encoder_rnn')
    encoder_rtn = encoder_rnn(encoder_emb)
    # # encoder_outputs, encoder_h, encoder_c = encoder_rnn(encoder_emb)
    # encoder_outputs = encoder_rtn[0]
    encoder_states = encoder_rtn[1:]

    # build decoder
    decoder_input = Input((None,), dtype='float32', name='decoder_input')
    if mask:
        decoder_emb_layer = Embedding(Vt + 1, demb, mask_zero=True, name='decoder_emb')
    else:
        decoder_emb_layer = Embedding(Vt, demb, mask_zero=False, name='decoder_emb')

    decoder_emb = decoder_emb_layer(decoder_input)

    if drop_p > 0.:
        decoder_emb = Dropout(drop_p)(decoder_emb, training=training)

    decoder_rnn = rnn(h, return_sequences=True, name='decoder_rnn')
    decoder_outputs = decoder_rnn(decoder_emb, initial_state=encoder_states)

    if drop_p > 0.:
        decoder_outputs = Dropout(drop_p)(decoder_outputs, training=training)

    if tied:
        final_outputs = DenseTransposeTied(Vt, tied_to=decoder_emb_layer,
                                           activation='linear', name='outputs')(decoder_outputs)
    else:
        final_outputs = Dense(Vt, activation='linear', name='outputs')(decoder_outputs)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[final_outputs])
    return model


def build_inference_decoder(mask=False, demb=128, h=128, Vt=5000, tied=True):
    rnn = LSTM if mask else CuDNNLSTM

    # build decoder
    decoder_input = Input(batch_shape=(None, None), dtype='float32', name='decoder_input')
    encoder_outputs = Input(batch_shape=(None, None, h), dtype='float32', name='encoder_outputs')
    encoder_h = Input(batch_shape=(None, h), dtype='float32', name='encoder_h')
    encoder_c = Input(batch_shape=(None, h), dtype='float32', name='encoder_c')

    if mask:
        decoder_emb_layer = Embedding(Vt + 1, demb, mask_zero=True,
                                      name='decoder_emb')
    else:
        decoder_emb_layer = Embedding(Vt, demb, mask_zero=False,
                                      name='decoder_emb')

    decoder_emb = decoder_emb_layer(decoder_input)

    decoder_rnn = rnn(h, return_sequences=True, name='decoder_rnn')
    decoder_outputs = decoder_rnn(decoder_emb, initial_state=[encoder_h, encoder_c])

    if tied:
        final_outputs = DenseTransposeTied(Vt, name='outputs',
                                           tied_to=decoder_emb_layer, activation='linear')(decoder_outputs)
    else:
        final_outputs = Dense(Vt, activation='linear', name='outputs')(decoder_outputs)

    inputs = [decoder_input, encoder_outputs, encoder_h, encoder_c]
    model = Model(inputs=inputs, outputs=[final_outputs])
    return model


def words_to_indices(data, vocab, mask=True):
    if mask:
        return [[vocab[w] + 1 for w in t] for t in data]
    else:
        return [[vocab[w] for w in t] for t in data]


def pad_texts(texts, eos, mask=True):
    maxlen = max(len(t) for t in texts)
    for t in texts:
        while len(t) < maxlen:
            if mask:
                t.insert(0, 0)
            else:
                t.append(eos)
    return np.asarray(texts, dtype='float32')


def train_cornell_movie(loo=0, num_users=200, num_words=5000, num_epochs=20, sample_user=False, exp_id=0, emb_h=128,
                        lr=0.001, batch_size=32, mask=False, drop_p=0.5, h=128, user_data_ratio=0., cross_domain=False,
                        ablation=False, tied=True, rnn_fn='gru'):
    if cross_domain:
        sample_user = True
        loo = None
        user_src_texts, user_trg_texts, dev_src_texts, dev_trg_texts, test_src_texts, test_trg_texts, \
            src_vocabs, trg_vocabs = load_ubuntu_by_user(num_users, num_words=num_words)
    else:
        user_src_texts, user_trg_texts, dev_src_texts, dev_trg_texts, test_src_texts, test_trg_texts, \
            src_vocabs, trg_vocabs = load_cornell_movie_by_user(num_users, num_words, user_data_ratio=user_data_ratio,
                                                                sample_user=sample_user)
    train_src_texts, train_trg_texts = [], []

    users = sorted(user_src_texts.keys())

    for i, user in enumerate(users):
        if loo is not None and i == loo:
            print "Leave user {} out".format(user)
            continue
        train_src_texts += user_src_texts[user]
        train_trg_texts += user_trg_texts[user]

    train_src_texts = words_to_indices(train_src_texts, src_vocabs, mask=mask)
    train_trg_texts = words_to_indices(train_trg_texts, trg_vocabs, mask=mask)
    dev_src_texts = words_to_indices(dev_src_texts, src_vocabs, mask=mask)
    dev_trg_texts = words_to_indices(dev_trg_texts, trg_vocabs, mask=mask)

    print "Num train data {}, num test data {}".format(len(train_src_texts), len(dev_src_texts))

    Vs = len(src_vocabs)
    Vt = len(trg_vocabs)
    print Vs, Vt

    model = build_dialogue_model(Vs=Vs, Vt=Vt, mask=mask, drop_p=drop_p, demb=emb_h, h=h, tied=tied, rnn_fn=rnn_fn)
    src_input_var, trg_input_var = model.inputs
    prediction = model.output

    trg_label_var = K.placeholder((None, None), dtype='float32')

    loss = K.sparse_categorical_crossentropy(trg_label_var, prediction, from_logits=True)
    loss = K.mean(K.sum(loss, axis=-1))

    optimizer = Adam(lr=lr)

    updates = optimizer.get_updates(loss, model.trainable_weights)
    train_fn = K.function([src_input_var, trg_input_var, trg_label_var, K.learning_phase()], [loss], updates=updates)
    pred_fn = K.function([src_input_var, trg_input_var, trg_label_var, K.learning_phase()], [loss])

    # pad batches to same length
    batches = []
    padded_train_src_texts = copy.deepcopy(train_src_texts)
    padded_train_trg_texts = copy.deepcopy(train_trg_texts)
    for batch in group_texts_by_len(padded_train_src_texts, padded_train_trg_texts, bs=batch_size):
        src_input, trg_input = batch
        src_input = pad_texts(src_input, src_vocabs['<eos>'], mask=mask)
        trg_input = pad_texts(trg_input, trg_vocabs['<eos>'], mask=mask)
        batches.append((src_input, trg_input))

    for epoch in range(num_epochs):
        np.random.shuffle(batches)

        for batch in batches:
            src_input, trg_input = batch
            _ = train_fn([src_input, trg_input[:, :-1], trg_input[:, 1:], 1])[0]

        train_loss, train_it = get_perp(train_src_texts, train_trg_texts, pred_fn, shuffle=True, prop=0.5)
        test_loss, test_it = get_perp(dev_src_texts, dev_trg_texts, pred_fn)

        print "Epoch {}, train loss={:.3f}, train perp={:.3f}, test loss={:.3f}, test perp={:.3f}".format(
            epoch, train_loss / len(train_src_texts) / 0.5,
            np.exp(train_loss / train_it), test_loss / len(dev_src_texts),
            np.exp(test_loss / test_it))

    if cross_domain:
        fname = 'ubuntu_dialog'
    else:
        fname = 'cornell_movie_dialog{}'.format('' if loo is None else loo)

    if ablation:
        fname = 'ablation_' + fname

    if 0. < user_data_ratio < 1.:
        fname += '_dr{}'.format(user_data_ratio)

    if sample_user:
        fname += '_shadow_exp{}_{}'.format(exp_id, rnn_fn)
        np.savez(MODEL_PATH + 'shadow_users{}_{}_{}_{}.npz'.format(exp_id, rnn_fn, num_users,
                                                                   'cd' if cross_domain else ''), users)

    model.save(MODEL_PATH + '{}_{}.h5'.format(fname, num_users))


def get_perp(user_src_data, user_trg_data, pred_fn, prop=1.0, shuffle=False):
    loss = 0.
    iters = 0.

    indices = np.arange(len(user_src_data))
    n = int(prop * len(indices))

    if shuffle:
        np.random.shuffle(indices)

    for idx in indices[:n]:
        src_text = np.asarray(user_src_data[idx], dtype=np.float32).reshape(1, -1)
        trg_text = np.asarray(user_trg_data[idx], dtype=np.float32)
        trg_input = trg_text[:-1].reshape(1, -1)
        trg_label = trg_text[1:].reshape(1, -1)

        err = pred_fn([src_text, trg_input, trg_label, 0])[0]

        loss += err
        iters += trg_label.shape[1]

    return loss, iters


if __name__ == '__main__':
    train_cornell_movie(loo=None, num_users=300, sample_user=False, num_epochs=30, drop_p=0.5, h=128, emb_h=128)
