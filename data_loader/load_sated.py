import numpy as np
from collections import Counter, defaultdict
from itertools import chain


SATED_PATH = '/hdd/song/nlp/sated-release-0.9.0/en-fr/'
SATED_TRAIN_ENG = SATED_PATH + 'train.en'
SATED_TRAIN_FR = SATED_PATH + 'train.fr'
SATED_TRAIN_USER = SATED_PATH + 'train.usr'
SATED_DEV_ENG = SATED_PATH + 'dev.en'
SATED_DEV_FR = SATED_PATH + 'dev.fr'
SATED_DEV_USER = SATED_PATH + 'dev.usr'
SATED_TEST_ENG = SATED_PATH + 'test.en'
SATED_TEST_FR = SATED_PATH + 'test.fr'
SATED_TEST_USER = SATED_PATH + 'test.usr'
EUROPARL_PATH = '/hdd/song/nlp/europarl/'
EUROPARL_DEEN_DE = EUROPARL_PATH + 'europarl.de-en.de.aligned.tok'
EUROPARL_DEEN_EN = EUROPARL_PATH + 'europarl.de-en.en.aligned.tok'
EUROPARL_FREN_FR = EUROPARL_PATH + 'europarl.fr-en.fr.aligned.tok'
EUROPARL_FREN_EN = EUROPARL_PATH + 'europarl.fr-en.en.aligned.tok'


def load_users(p=SATED_TRAIN_USER):
    users = []
    with open(p, 'rb') as f:
        for line in f:
            users.append(line.replace('\n', ''))
    return users


def load_texts(p=SATED_TRAIN_ENG):
    texts = []
    with open(p, 'rb') as f:
        for line in f:
            arr = ['<sos>'] + line.replace('\n', '').split(' ') + ['<eos>']
            words = []
            for w in arr:
                words.append(w)
            texts.append(words)

    return texts


def process_texts(texts, vocabs):
    for t in texts:
        for i, w in enumerate(t):
            if w not in vocabs:
                t[i] = '<unk>'


def process_vocabs(vocabs, num_words=10000):
    counter = Counter(vocabs)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    print('Loaded {} vocabs'.format(len(count_pairs)))

    if num_words is not None:
        count_pairs = count_pairs[:num_words - 1]

    print count_pairs[:50]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, np.arange(len(words))))
    return word_to_id


def load_sated_data(num_words=10000):
    train_src_texts = load_texts(SATED_TRAIN_ENG)
    train_trg_texts = load_texts(SATED_TRAIN_FR)

    dev_src_texts = load_texts(SATED_DEV_ENG)
    dev_trg_texts = load_texts(SATED_DEV_FR)

    test_src_texts = load_texts(SATED_TEST_ENG)
    test_trg_texts = load_texts(SATED_TEST_FR)

    src_words = list(chain(*train_src_texts))
    trg_words = list(chain(*train_trg_texts))

    src_vocabs = process_vocabs(src_words, num_words)
    trg_vocabs = process_vocabs(trg_words, num_words)

    process_texts(train_src_texts, src_vocabs)
    process_texts(train_trg_texts, trg_vocabs)

    process_texts(dev_src_texts, src_vocabs)
    process_texts(dev_trg_texts, trg_vocabs)

    process_texts(test_src_texts, src_vocabs)
    process_texts(test_trg_texts, trg_vocabs)

    src_words = list(chain(*train_src_texts))
    trg_words = list(chain(*train_trg_texts))

    src_vocabs = process_vocabs(src_words, None)
    trg_vocabs = process_vocabs(trg_words, None)

    return train_src_texts, train_trg_texts, dev_src_texts, dev_trg_texts, test_src_texts, test_trg_texts, \
           src_vocabs, trg_vocabs


def load_sated_data_by_user(num_users=100, num_words=10000, test_on_user=False, sample_user=False,
                            seed=12345, user_data_ratio=0.):
    src_users = load_users(SATED_TRAIN_USER)
    train_src_texts = load_texts(SATED_TRAIN_ENG)
    train_trg_texts = load_texts(SATED_TRAIN_FR)

    dev_src_texts = load_texts(SATED_DEV_ENG)
    dev_trg_texts = load_texts(SATED_DEV_FR)

    test_src_texts = load_texts(SATED_TEST_ENG)
    test_trg_texts = load_texts(SATED_TEST_FR)

    user_counter = Counter(src_users)
    all_users = [tup[0] for tup in user_counter.most_common()]
    # print len(all_users)
    np.random.seed(seed)
    np.random.shuffle(all_users)
    np.random.seed(None)

    train_users = set(all_users[:num_users])
    test_users = set(all_users[num_users: num_users * 2])

    if sample_user:
        attacker_users = all_users[num_users * 2: num_users * 4]
        # np.random.seed(None)
        train_users = np.random.choice(attacker_users, size=num_users, replace=False)
        print len(train_users)
        print train_users[:10]

    user_src_texts = defaultdict(list)
    user_trg_texts = defaultdict(list)

    test_user_src_texts = defaultdict(list)
    test_user_trg_texts = defaultdict(list)

    for u, s, t in zip(src_users, train_src_texts, train_trg_texts):
        if u in train_users:
            user_src_texts[u].append(s)
            user_trg_texts[u].append(t)
        if test_on_user and u in test_users:
            test_user_src_texts[u].append(s)
            test_user_trg_texts[u].append(t)

    if 0. < user_data_ratio < 1.:
        # held out some fraction of data for testing
        for u in user_src_texts:
            l = len(user_src_texts[u])
            # print l
            l = int(l * user_data_ratio)
            user_src_texts[u] = user_src_texts[u][:l]
            user_trg_texts[u] = user_trg_texts[u][:l]

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

    if test_on_user:
        for u in test_users:
            process_texts(test_user_src_texts[u], src_vocabs)
            process_texts(test_user_trg_texts[u], trg_vocabs)

    process_texts(dev_src_texts, src_vocabs)
    process_texts(dev_trg_texts, trg_vocabs)

    process_texts(test_src_texts, src_vocabs)
    process_texts(test_trg_texts, trg_vocabs)

    src_words = []
    trg_words = []
    for u in train_users:
        src_words += list(chain(*user_src_texts[u]))
        trg_words += list(chain(*user_trg_texts[u]))

    src_vocabs = process_vocabs(src_words, None)
    trg_vocabs = process_vocabs(trg_words, None)

    if test_on_user:
        return user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts,  src_vocabs, trg_vocabs
    else:
        return user_src_texts, user_trg_texts, dev_src_texts, dev_trg_texts, test_src_texts, test_trg_texts,\
               src_vocabs, trg_vocabs


def read_europarl_file(filename, num_lines=80000):
    texts = []
    with open(filename, 'rb') as f:
        for line in f:
            arr = ['<sos>'] + line.lower().replace('\n', '').split(' ') + ['<eos>']
            texts.append(arr)
            if len(texts) > num_lines:
                break
    return texts


def load_europarl_by_user(num_users=200, num_data_per_user=150, num_words=5000, test_size=5000):
    src_texts = read_europarl_file(EUROPARL_FREN_EN, num_users * num_data_per_user * 2 + test_size)
    trg_texts = read_europarl_file(EUROPARL_FREN_FR, num_users * num_data_per_user * 2 + test_size)

    test_src_texts = src_texts[-test_size:]
    test_trg_texts = trg_texts[-test_size:]
    src_texts = src_texts[:-test_size]
    trg_texts = trg_texts[:-test_size]

    all_users = np.arange(num_users * 2)
    np.random.seed(None)
    train_users = np.random.choice(all_users, size=num_users, replace=False)

    user_src_texts = defaultdict(list)
    user_trg_texts = defaultdict(list)

    for u in train_users:
        user_src_texts[u] = src_texts[u * num_data_per_user: (u + 1) * num_data_per_user]
        user_trg_texts[u] = trg_texts[u * num_data_per_user: (u + 1) * num_data_per_user]

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

    process_texts(test_src_texts, src_vocabs)
    process_texts(test_trg_texts, trg_vocabs)

    src_words = []
    trg_words = []
    for u in train_users:
        src_words += list(chain(*user_src_texts[u]))
        trg_words += list(chain(*user_trg_texts[u]))

    src_vocabs = process_vocabs(src_words, None)
    trg_vocabs = process_vocabs(trg_words, None)

    return user_src_texts, user_trg_texts, test_src_texts, test_trg_texts, test_src_texts, test_trg_texts,\
           src_vocabs, trg_vocabs


if __name__ == '__main__':
    load_sated_data_by_user(num_users=300, sample_user=False)
