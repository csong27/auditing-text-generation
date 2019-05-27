import io
import ast
import numpy as np
import os

from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from itertools import chain
from load_sated import process_vocabs, process_texts


DATA_PATH = '/hdd/song/nlp/cornell_movie_dialogs_corpus/'
MOVIE_LINES_PATH = DATA_PATH + 'movie_lines.txt'
MOVIE_CONVERSATIONS_PATH = DATA_PATH + 'movie_conversations.txt'
SEPARATOR = ' +++$+++ '

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]


def load_lines(filename, fields=MOVIE_LINES_FIELDS):
    lines = {}

    with io.open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.replace('\n', '').split(SEPARATOR)

            # Extract fields
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]

            lines[line_obj['lineID']] = line_obj

    return lines


def load_conversations(lines, filename, fields=MOVIE_CONVERSATIONS_FIELDS):
    conversations = []

    with io.open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.replace('\n', '').split(SEPARATOR)

            # Extract fields
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]

            # Convert string to list (conv_obj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            line_ids = ast.literal_eval(conv_obj["utteranceIDs"])

            # Reassemble lines
            conv_obj["lines"] = []
            for line_id in line_ids:
                conv_obj["lines"].append(lines[line_id])

            conversations.append(conv_obj)

    return conversations


def count_character_lines():
    lines = load_lines(MOVIE_LINES_PATH)
    character_counter = Counter()
    for line_id in lines:
        line = lines[line_id]
        character = line["characterID"]
        character_counter[character] += 1

    print len(character_counter)
    print character_counter.most_common(100)


def save_extracted_cornell_movie():
    lines = load_lines(MOVIE_LINES_PATH)
    conversations = load_conversations(lines, MOVIE_CONVERSATIONS_PATH)
    src_texts = []
    trg_texts = []
    src_chars = []
    trg_chars = []

    for conv_obj in conversations:
        conv_line_objs = conv_obj["lines"]
        n_lines = len(conv_line_objs)
        for i in range(n_lines - 1):
            st = word_tokenize(conv_line_objs[i]["text"])
            tt = word_tokenize(conv_line_objs[i + 1]["text"])

            src_texts.append('\t'.join(st))
            trg_texts.append('\t'.join(tt))

            src_chars.append(conv_line_objs[i]["characterID"])
            trg_chars.append(conv_line_objs[i + 1]["characterID"])

    print len(src_texts), len(trg_texts)

    with io.open(DATA_PATH + 'extracted_src_trg.txt', 'w', encoding='iso-8859-1') as f:
        for st, sc, tt, tc in zip(src_texts, src_chars, trg_texts, trg_chars):
            f.write(SEPARATOR.join([st, sc, tt, tc]) + '\n')


def load_extracted_cornell_movie(dev_size=10000, test_size=10000):
    src_texts = []
    trg_texts = []
    src_chars = []
    trg_chars = []

    with io.open(DATA_PATH + 'extracted_src_trg.txt', 'r', encoding='iso-8859-1') as f:
        for line in f:
            st, sc, tt, tc = line.lower().replace('\n', '').split(SEPARATOR)
            st = ['<sos>'] + st.split('\t') + ['<eos>']
            tt = ['<sos>'] + tt.split('\t') + ['<eos>']

            src_texts.append(st)
            trg_texts.append(tt)

            src_chars.append(sc)
            trg_chars.append(tc)

    src_texts = np.asarray(src_texts)
    trg_texts = np.asarray(trg_texts)
    
    np.random.seed(12345)
    indices = np.arange(len(src_texts))
    train_indices = np.random.choice(indices, len(indices) - dev_size - test_size, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    dev_indices = test_indices[:dev_size]
    test_indices = test_indices[dev_size:]
    
    train_src_texts, dev_src_texts, test_src_texts = \
        src_texts[train_indices], src_texts[dev_indices], src_texts[test_indices]
    train_trg_texts, dev_trg_texts, test_trg_texts = \
        trg_texts[train_indices], trg_texts[dev_indices], trg_texts[test_indices]

    train_src_chars = np.asarray(src_chars)[train_indices]
    train_trg_chars = np.asarray(trg_chars)[train_indices]

    train_data = (train_src_texts, train_trg_texts, train_src_chars, train_trg_chars)
    dev_data = (dev_src_texts, dev_trg_texts)
    test_data = (test_src_texts, test_trg_texts)
    return train_data, dev_data, test_data


def load_cornell_movie(num_words=10000):
    train_data, dev_data, test_data = load_extracted_cornell_movie()
    train_src_texts, train_trg_texts, _, _ = train_data
    dev_src_texts, dev_trg_texts = dev_data
    test_src_texts, test_trg_texts = test_data

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


def load_cornell_movie_by_user(num_users=100, num_words=5000, test_on_user=False, sample_user=False, min_count=20,
                               user_data_ratio=0.):
    train_data, dev_data, test_data = load_extracted_cornell_movie(dev_size=5000, test_size=5000)
    train_src_texts, train_trg_texts, src_users, _ = train_data
    dev_src_texts, dev_trg_texts = dev_data
    test_src_texts, test_trg_texts = test_data

    user_counter = Counter(src_users)
    all_users = np.asarray([tup[0] for tup in user_counter.most_common() if tup[1] >= min_count])
    print 'Loaded {} users'.format(len(all_users))

    np.random.seed(12345)
    np.random.shuffle(all_users)
    np.random.seed(None)

    train_users = set(all_users[:num_users])
    test_users = all_users[num_users:num_users * 2]

    if sample_user:
        attacker_users = all_users[num_users * 2: num_users * 4]
        np.random.seed(None)
        train_users = np.random.choice(attacker_users, size=num_users, replace=False)
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
        return user_src_texts, user_trg_texts, dev_src_texts, dev_trg_texts, test_src_texts, test_trg_texts, \
               src_vocabs, trg_vocabs


UBUNTU_PATH = '/hdd/song/nlp/ubuntu/'


def load_ubuntu_lines(filename):
    lines = []
    with open(filename, 'rb') as f:
        for line in f:
            l = line[line.rindex("\t") + 1:].strip()  # Strip metadata (timestamps, speaker names)
            lines.append(l.lower())
    return lines


def load_raw_ubuntu(max_dir=15):
    conversations = []
    n = 0
    dialogs_path = os.path.join(UBUNTU_PATH, 'dialogs')
    for sub in os.listdir(dialogs_path)[:max_dir]:
        subdir = os.path.join(dialogs_path, sub)
        for filename in os.listdir(subdir):
            filename = os.path.join(subdir, filename)
            if filename.endswith('tsv'):
                lines = load_ubuntu_lines(filename)
                conversations.append(lines)
                n += len(lines)
    return conversations


def preprocess(t):
    words = t.split(' ')
    for i in range(len(words)):
        if 'http' in words[i] or 'www' in words[i]:
            words[i] = '/url/'
    return ' '.join(words).decode("utf8")


def save_extracted_ubuntu():
    src_texts = []
    trg_texts = []

    conversations = load_raw_ubuntu()
    for lines in conversations:
        n_lines = len(lines)
        for i in range(n_lines - 1):
            st = word_tokenize(preprocess(lines[i]))
            tt = word_tokenize(preprocess(lines[i + 1]))

            src_texts.append('\t'.join(st))
            trg_texts.append('\t'.join(tt))

    print len(src_texts), len(trg_texts)

    with open(UBUNTU_PATH + 'extracted_src_trg.txt', 'wb') as f:
        for st, tt in zip(src_texts, trg_texts):
            f.write(SEPARATOR.join([st, tt]).encode('utf-8') + '\n')


def load_extracted_ubuntu(num_lines):
    src_texts = []
    trg_texts = []

    with open(UBUNTU_PATH + 'extracted_src_trg.txt', 'rb') as f:
        for line in f:
            st, tt = line.lower().replace('\n', '').split(SEPARATOR)
            st = ['<sos>'] + st.split('\t') + ['<eos>']
            tt = ['<sos>'] + tt.split('\t') + ['<eos>']

            src_texts.append(st)
            trg_texts.append(tt)
            if len(src_texts) > num_lines:
                break

    return src_texts, trg_texts


def load_ubuntu_by_user(num_users=200, num_words=5000, num_data_per_user=200, test_size=5000):
    src_texts, trg_texts = load_extracted_ubuntu(num_users * num_data_per_user * 2 + test_size)
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


# if __name__ == '__main__':
#     # load_cornell_movie_by_user(num_users=200, sample_user=False, user_data_ratio=0.5)
#     load_ubuntu_by_users()