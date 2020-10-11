from __future__ import unicode_literals
import collections
import io
import re
import six
import numpy as np
import progressbar
import json
import os
import pickle
from collections import namedtuple
import torch

from config import get_preprocess_args


split_pattern = re.compile(r'([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')

Special_Seq = namedtuple('Special_Seq', ['PAD', 'EOS', 'UNK', 'BOS'])
Vocab_Pad = Special_Seq(PAD=0, EOS=1, UNK=2, BOS=3)


def split_sentence(s, tok=False):
    if tok:
        s = s.lower()
        s = s.replace('\u2019', "'")
        # s = digit_pattern.sub('0', s)
    words = []
    for word in s.strip().split():
        if tok:
            words.extend(split_pattern.split(word))
        else:
            words.append(word)
    words = [w for w in words if w]
    return words


def open_file(path):
    return io.open(path, encoding='utf-8', errors='ignore')


def count_lines(path):
    with open_file(path) as f:
        return sum([1 for _ in f])


def read_file(path, tok=False, ek=False):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    with open_file(path) as f:
        for line in bar(f, max_value=n_lines):
            tokens = line.strip().split('\t')
            if(not ek):
                label = tokens[0]
                text = ' '.join(tokens[1:])
            else:
                label = -1
                text = ' '.join(tokens)
            words = split_sentence(text, tok)
            yield label, words


def count_words(path, max_vocab_size=40000, tok=False, ek=False):
    counts = collections.Counter()
    for _, words in read_file(path, tok, ek):
        for word in words:
            counts[word] += 1
    vocab = [word for (word, _) in counts.most_common(max_vocab_size)]
    return vocab


def get_label_vocab(path, tok=False):
    vocab_label = set()
    for label, _ in read_file(path, tok):
        vocab_label.add(label)
    return sorted(list(vocab_label))


def make_dataset(path, w2id, tok=False, ek=False):
    labels = []
    dataset = []
    token_count = 0
    unknown_count = 0
    for label, words in read_file(path, tok, ek):
        labels.append(label)
        array = make_array(w2id, words)
        dataset.append(array)
        token_count += array.size
        unknown_count += (array == Vocab_Pad.UNK).sum()
    print('# of tokens: %d' % token_count)
    print('# of unknown: %d (%.2f %%)' % (unknown_count,
                                          100. * unknown_count / token_count))
    return labels, dataset


def make_array(word_id, words):
    ids = [word_id.get(word, Vocab_Pad.UNK) for word in words]
    return np.array(ids, 'i')


if __name__ == "__main__":
    args = get_preprocess_args()

    print(json.dumps(args.__dict__, indent=4))

    # Vocab Construction
    train_path = os.path.join(args.input, args.train_filename)
    valid_path = os.path.join(args.input, args.dev_filename)
    test_path = os.path.join(args.input, args.test_filename)
#     unlabel_path = os.path.join(args.input, args.unlabel_filename)
    ek_path = os.path.join(args.input, args.ek_filename)
    ekt_path = os.path.join(args.input, args.ekt_filename)

    train_word_cntr = count_words(train_path, args.vocab_size, args.tok)
    valid_word_cntr = count_words(valid_path, args.vocab_size, args.tok)
#     unlabel_word_cntr = count_words(unlabel_path, args.vocab_size, args.tok)
    ek_word_cntr = count_words(ek_path, args.vocab_size, args.tok, ek=True)
    ekt_word_cntr = count_words(ekt_path, args.vocab_size, args.tok, ek=True)
    
    all_words = list(set(train_word_cntr + valid_word_cntr + ek_word_cntr))

    # word_cntr = count_words(unlabel_path, args.vocab_size, args.tok)
    #
    # all_words = word_cntr

    vocab = ['<pad>', '<eos>', '<unk>', '<bos>'] + all_words
    w2id = {word: index for index, word in enumerate(vocab)}

    label_list = get_label_vocab(train_path, args.tok)
    label2id = {l: index for index, l in enumerate(label_list)}

#     # Unlabelled Dataset
    
#     labels, data = make_dataset(unlabel_path, w2id, args.tok)
#     print('Original unlabelled data size: %d' % len(data))
#     unlab_r = []
#     unlabel_data = []
#     for i, (l, s) in enumerate(six.moves.zip(labels, data)):
#         if 0 < len(s) < args.max_seq_length:
#             unlabel_data.append((-1, s))
#         else:
#             unlab_r.append(i)
    
#     print('Filtered unlabelled data size: %d' % len(unlabel_data))
#     print('Removed unlabelled data: %d' % len(unlab_r))
    
     # External Knowledge Dataset
    
    labels, data = make_dataset(ek_path, w2id, args.tok, True)
    print('Original ek data size: %d' % len(data))
    ek_data = []
    for i, (l, s) in enumerate(six.moves.zip(labels, data)):
        ek_data.append((-1, s))
    
    print('Filtered ek data size: %d' % len(ek_data))
    
    # External Knowledge test
    
    labels, data = make_dataset(ekt_path, w2id, args.tok, True)
    print('Original ek data size: %d' % len(data))
    ekt_data = []
    for i, (l, s) in enumerate(six.moves.zip(labels, data)):
        ekt_data.append((-1, s))
    
    print('Filtered ek data size: %d' % len(ekt_data))

    # Train Dataset
    labels, data = make_dataset(train_path, w2id, args.tok)
    print('Original training data size: %d' % len(data))
    train_r = []
    train_data = []
    for i, (l, s) in enumerate(six.moves.zip(labels, data)):
        if 0 < len(s) < args.max_seq_length:
            train_data.append((label2id[l], s))
        else:
            train_r.append(i)
    
    print('Removed training data: %d' % len(train_r))
    print('Filtered training data size: %d' % len(train_data))
    
    # Valid Dataset
    labels, data = make_dataset(valid_path, w2id, args.tok)
    valid_data = [(label2id[l], s) for l, s in six.moves.zip(labels, data)
                  if 0 < len(s)]
    
    print('Filtered validation data size: %d' % len(valid_data))

    # Test Dataset
    labels, data = make_dataset(test_path, w2id, args.tok)
    test_r = []
    test_data = []
    for i, (l, s) in enumerate(six.moves.zip(labels, data)):
        if 0 < len(s):
            test_data.append((label2id[l], s))
        else:
            test_r.append(i)
    
    print('Filtered testing data size: %d' % len(test_data))
    
    # Addn Dataset
    print('Preparing addn data...')
    addn_data_tweets = np.load(args.addndata + "/train/tweet_features.npy")
    addn_data_users = np.load(args.addndata + "/train/user_features.npy")

    addn_data = np.concatenate((addn_data_tweets, addn_data_users), axis=1)
    
    print('Train Addn')
#     for i in train_r:
#         print("\t", i)
    addn_data_1 = np.delete(addn_data, train_r, 0)
        
    addn_data = torch.from_numpy(addn_data_1)
    
    print('Filtered addn train data:', addn_data.shape)
    
    print(ek_data[:5])
    
    ek_data_1 = np.delete(np.array(ek_data, dtype=object), train_r, 0)
        
    ek_data = ek_data_1.tolist()
    
    print('Filtered ek train data:', len(ek_data))

    addn_data_tweets_t = np.load(args.addndata + "/test/tweet_features.npy")
    addn_data_users_t = np.load(args.addndata + "/test/user_features.npy")

    addn_data_t = np.concatenate((addn_data_tweets_t, addn_data_users_t), axis=1)
    
    print('Test Addn')
#     for i in test_r:
#         print("\t", i)
    addn_data_t_1 = np.delete(addn_data_t, test_r, 0)
        
    addn_data_t = torch.from_numpy(addn_data_t_1)
    
    print('Filtered addn testing data:', addn_data_t.shape)
    
    print(ekt_data[:5])
    
    ekt_data_1 = np.delete(np.array(ekt_data, dtype=object), test_r, 0)
        
    ekt_data = ekt_data_1.tolist()
    
    print('Filtered ek test data:', len(ekt_data))

#     addn_data_unlab_tweets = np.load(args.addndata + "/unlab/tweet_features.npy")
#     addn_data_unlab_users = np.load(args.addndata + "/unlab/user_features.npy")

#     addn_data_unlab = np.concatenate((addn_data_unlab_tweets, addn_data_unlab_users), axis=1)
    
#     print('Unlab Addn')
# #     for i in unlab_r:
# #         print("\t", i)
#     addn_data_unlab_1 = np.delete(addn_data_unlab, unlab_r, 0)
    
#     addn_data_unlab = torch.from_numpy(addn_data_unlab_1)
    
#     print('Filtered addn unlabelled data:', addn_data_unlab.shape)

    # Display corpus statistics
    print("Vocab: {}".format(len(vocab)))

    id2w = {i: w for w, i in w2id.items()}
    id2label = {i: l for l, i in label2id.items()}

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save the dataset as pytorch serialized files
#     torch.save(unlabel_data,
#                os.path.join(args.output, args.save_data + '.unlabel.pth'))
    torch.save(train_data,
               os.path.join(args.output, args.save_data + '.train.pth'))
    torch.save(ek_data,
               os.path.join(args.output, args.save_data + '.external.pth'))
    torch.save(ekt_data,
               os.path.join(args.output, args.save_data + '.externalT.pth'))
    torch.save(valid_data,
               os.path.join(args.output, args.save_data + '.valid.pth'))
    torch.save(test_data,
               os.path.join(args.output, args.save_data + '.test.pth'))
#     torch.save(addn_data_unlab,
#                os.path.join(args.output, args.save_data + '.unlabel_addn.pth'))
    torch.save(addn_data,
               os.path.join(args.output, args.save_data + '.train_addn.pth'))
    torch.save(addn_data_t,
               os.path.join(args.output, args.save_data + '.valid_addn.pth'))
    torch.save(addn_data_t,
               os.path.join(args.output, args.save_data + '.test_addn.pth'))

    # Save the word vocab
    with open(os.path.join(args.output, args.save_data + '.vocab.pickle'), 'wb') as f:
        pickle.dump(id2w, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the label vocab
    with open(os.path.join(args.output, args.save_data + '.label.pickle'),
              'wb') as f:
        pickle.dump(id2label, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Done')