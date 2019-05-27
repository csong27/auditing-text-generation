from load_reddit import build_vocab
from collections import defaultdict
import numpy as np


WIKI_PATH_DIR = '/hdd/song/nlp/wikitext-103/'
WIKI_TRAIN_PATH = WIKI_PATH_DIR + 'wiki.train.tokens'
WIKI_DEV_PATH = WIKI_PATH_DIR + 'wiki.valid.tokens'
WIKI_TEST_PATH = WIKI_PATH_DIR + 'wiki.test.tokens'


def load_wiki_lines(filename=WIKI_TRAIN_PATH, num_lines=100):
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.replace('\n', '').lower().lstrip().rstrip()
            if not line:
                continue
            arr = line.split(' ')
            if arr[0] == '=':
                continue
            data.append(arr + ['<eos>'])
            if len(data) >= num_lines > 0:
                break
    print num_lines, len(data)
    return data


def load_wiki_test_data():
    dev = load_wiki_lines(WIKI_DEV_PATH, num_lines=0)
    test = load_wiki_lines(WIKI_TEST_PATH, num_lines=0)

    return dev + test


def load_wiki_by_users(num_users=200, num_data_per_user=100, num_words=5000, vocabs=None):
    train_data = load_wiki_lines(num_lines=2 * num_users * num_data_per_user)
    print "Splitting data to {} users, each has {} texts".format(num_users * 2, num_data_per_user)

    all_users = np.arange(num_users * 2)
    np.random.seed(None)
    train_users = np.random.choice(all_users, size=num_users, replace=False)
    print train_users[:10]

    user_comments = defaultdict(list)
    all_words = []

    for u in train_users:
        data = train_data[u * num_data_per_user:  (u + 1) * num_data_per_user]
        for words in data:
            all_words += words
        user_comments[str(u)] = data

    if vocabs is None:
        vocabs = build_vocab(all_words, num_words + 1)

    all_words = []
    for user in user_comments:
        comments = user_comments[user]
        for i in range(len(comments)):
            comment = comments[i]
            for j in range(len(comment)):
                word = comment[j]
                if word not in vocabs:
                    comment[j] = '<unk>'
            all_words += comment

    vocabs = build_vocab(all_words, None)
    return user_comments, vocabs


if __name__ == '__main__':
    load_wiki_by_users()