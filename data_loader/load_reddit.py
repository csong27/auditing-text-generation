import os
import codecs
import numpy as np

from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize


REDDIT_PATH = '/hdd/song/nlp/reddit/'
REDDIT_USER_PATH = REDDIT_PATH + 'shard_by_author/'
REDDIT_PROCESSED_PATH = REDDIT_PATH + 'shard_by_author_processed/'
REDDIT_USER_COUNT_PATH = REDDIT_PATH + 'author_count'
REDDIT_TEST_PATH = REDDIT_PATH + 'test_data.json'
PTB_DATA_DIR = './ptb/simple-examples/data/'


def translate(t):
    t = t.replace(u'\u2018', '\'')
    t = t.replace(u'\u2019', '\'')
    t = t.replace(u'\u201c', '\"')
    t = t.replace(u'\u201d', '\"')
    t = t.replace(u'\u2013', '-')
    t = t.replace(u'\u2014', '-')

    t = t.replace(u'\u2026', '')
    t = t.replace(u'\ufffd', '')
    t = t.replace(u'\ufe0f', '')
    t = t.replace(u'\u035c', '')
    t = t.replace(u'\u0296', '')
    t = t.replace(u'\u270a', '')
    t = t.replace(u'*', '')
    t = t.replace(u'~', '')

    t = t.replace(u'\ufb00', 'ff')

    return t


def preprocess(t):
    words = t.split(' ')
    for i in range(len(words)):
        if 'http' in words[i]:
            words[i] = '/url/'
    return ' '.join(words)


def remove_puncs(words):
    new_words = []
    for w in words:
        flag = False
        for c in w:
            if c.isalnum():
                flag = True
                break
        if flag:
            new_words.append(w)
    return new_words


def write_processed_comments():
    for user in os.listdir(REDDIT_USER_PATH):
        filename = os.path.join(REDDIT_USER_PATH, user)
        cnt = 0
        new_lines = []
        with codecs.open(filename, encoding='utf-8') as f:
            for line in f:
                text = line[1:-2].decode('unicode_escape').lower()
                text = translate(text)
                text = preprocess(text)
                words = word_tokenize(text)
                words = remove_puncs(words)
                if len(words) < 3:
                    continue
                cnt += 1
                new_line = ' '.join(words)
                # print new_line
                new_lines.append(new_line)
        print user, cnt

        with open(REDDIT_PROCESSED_PATH + user, 'wb') as f:
            for line in new_lines:
                f.write(line.encode('utf8') + '\n')
        # quit()


def read_top_users(num_users=100, random=True, min_count=200):
    users = []
    cnts = []
    with codecs.open(REDDIT_USER_COUNT_PATH, encoding='utf-8') as f:
        for line in f:
            user, cnt = line.split('\t')
            cnt = int(cnt)
            if cnt < min_count:
                continue
            users.append(user)
            cnts.append(cnt)

    print len(users), sum(cnts)

    cnts = np.asarray(cnts)
    if random:
        np.random.seed(12345)
        top_indices = np.arange(len(cnts))
        np.random.shuffle(top_indices)
        top_indices = top_indices[:num_users]
        np.random.seed(None)
    else:
        top_indices = np.argsort(-cnts)[:num_users]
    top_users = np.asarray(users)[top_indices]

    print('Loading {} comments for {} users'.format(cnts[top_indices].sum(), num_users))

    return top_users


def read_random_users(num_users=100, num_top_users=100):
    users = []
    cnts = []
    with codecs.open(REDDIT_USER_COUNT_PATH, encoding='utf-8') as f:
        for line in f:
            user, cnt = line.split('\t')
            cnt = int(cnt)
            users.append(user)
            cnts.append(cnt)

    cnts = np.asarray(cnts)
    indices = np.argsort(-cnts)[num_top_users:]
    users = np.asarray(users)[indices]
    random_users = np.random.choice(users, num_users, replace=False)
    return random_users


def build_vocab(data, num_words=20000):
    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    print('Loaded {} vocabs'.format(len(count_pairs)))

    if num_words is not None:
        count_pairs = count_pairs[:num_words - 1]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, np.arange(len(words))))
    return word_to_id


def read_comments_by_user(user, vocabs, max_token=1600):
    filename = os.path.join(REDDIT_PROCESSED_PATH, user)
    user_comments = []
    user_num_tokens = 0
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            data = line.replace('/url/', '').replace('\n', '').split() + ['<eos>']
            if len(data) == 1:
                continue

            user_comments.append(data)
            user_num_tokens += len(data)
            if user_num_tokens >= max_token + 1:
                break

    for i in range(len(user_comments)):
        comment = user_comments[i]
        for j in range(len(comment)):
            word = comment[j]
            if word not in vocabs:
                comment[j] = '<unk>'
    return user_comments


def read_top_user_comments(num_users=200, num_words=5000, max_token=None, vocabs=None, top_users=None,
                           sample_user=False, load_raw=False):
    if sample_user and top_users is None:
        top_users = read_top_users(num_users * 4)[num_users * 2: num_users * 4]
        print len(top_users)

        np.random.seed(None)
        top_users = np.random.choice(top_users, size=num_users, replace=False)
        print len(top_users)
    elif top_users is None:
        top_users = read_top_users(num_users)

    user_comments = defaultdict(list)
    user_num_tokens = Counter()
    all_words = []
    for user in top_users:
        filename = os.path.join(REDDIT_PROCESSED_PATH, user)
        with codecs.open(filename, encoding='utf-8') as f:
            for line in f:
                data = line.replace('/url/', '').replace('\n', '').split() + ['<eos>']

                if len(data) == 1:
                    continue

                user_comments[user].append(data)
                user_num_tokens[user] += len(data)
                all_words += data
                if max_token is not None and user_num_tokens[user] >= max_token + 1:
                    break
    if load_raw:
        return user_comments

    if vocabs is None:
        vocabs = build_vocab(all_words, num_words)

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


def read_test_comments_by_user(num_top_users=100):
    test_users = read_random_users(num_top_users, num_top_users)
    user_comments = defaultdict(list)
    for user in test_users:
        filename = os.path.join(REDDIT_PROCESSED_PATH, user)
        with codecs.open(filename, encoding='utf-8') as f:
            for line in f:
                data = line.replace('/url/', '').replace('\n', '').split() + ['<eos>']
                if len(data) == 1:
                    continue

                user_comments[user].append(data)
    return user_comments


def read_test_comments():
    test_comments = []
    with codecs.open(REDDIT_TEST_PATH, encoding='utf-8') as f:
        for line in f:
            text = line[1:-2].decode('unicode_escape').lower()
            text = translate(text)
            text = preprocess(text)
            text = text.replace('/url/', '').replace('\n', '')
            words = word_tokenize(text)
            words = remove_puncs(words) + ['<eos>']
            if len(words) > 2:
                test_comments.append(words)

    print('Loaded {} test data'.format(len(test_comments)))
    return test_comments


def read_ptb_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(line.decode('utf-8').replace('\n', '<eos>').split())
    return data


def read_ptb_data_by_user(num_users=100, num_words=5000, vocabs=None):
    train_path = os.path.join(PTB_DATA_DIR, 'ptb.train.txt')
    train_data = read_ptb_file(train_path)

    l = len(train_data)
    num_data_per_user = l // (num_users * 2)
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


def read_ptb_test_data():
    test_path = os.path.join(PTB_DATA_DIR, 'ptb.test.txt')
    test_data = read_ptb_file(test_path)
    return test_data


if __name__ == '__main__':
    read_top_user_comments(num_users=5000, num_words=5000, sample_user=True)
