from collections import Counter
import numpy as np
import pickle
import os

# parameters
vocab_size = 100
num_sentences = 100

# data path names
data_path = os.path.join(os.path.abspath(os.path.curdir), "FrenchEnglishData")
train_path = os.path.join(data_path, "training") #475 files per language
test_path = os.path.join(data_path, "test") # 1 file per language

def load_data(data_path, num_sentences=100):
    # get file_names
    file_names = os.listdir(data_path)

    english_sentences = []
    french_sentences = []

    # collecting sentences
    for i in range(len(file_names)):
        f = open(os.path.join(train_path, file_names[i]), 'r', encoding = "ISO-8859-1")
        text = f.read()
        f.close()
        sentences = text.split("\n")
        if len(sentences[-1]) == 0:
            del sentences[-1]
        words = []
        for sentence in sentences:
            sent_words = sentence.split(" ")
            if len(sent_words[-1]) == 0:
                del sent_words[-1]
            sent_words.append("<EOS>")
            sent_words = ["<SOS>"] + sent_words
            if i % 2 == 0:
                english_sentences.append(sent_words)
            else:
                french_sentences.append(sent_words)
            if len(french_sentences) > num_sentences:
                break

    return (english_sentences, french_sentences)

def build_vocab(sentences, language, vocab_size=50000):
    word_counter = Counter()
    for sentence in sentences:
        for word in sentence:
            word_counter[word] += 1
    ls = word_counter.most_common(vocab_size-1)
    vocab = {w[0]: index + 1 for (index, w) in enumerate(ls)} # leave 0 to UNK
    vocab["<UNK>"] = 0
    if language == "english":
        pickle.dump(vocab, open("english_vocabulary_dict.p", "wb"))
    else:
        pickle.dump(vocab, open("french_vocabulary_dict.p", "wb"))
    return vocab

def vectorize_data(english_sentences, french_sentences, train=True):
    if train:
        english_vocab = build_vocab(english_sentences, "english")
        french_vocab = build_vocab(french_sentences, "french")
    else:
        english_vocab = pickle.load(open("english_vocabulary_dict.p", "rb"))
        french_vocab = pickle.load(open("french_vocabulary_dict.p", "rb"))

    english_indices = []
    french_indices = []
    for sentence in english_sentences:
        word_seq = [english_vocab[w] if w in english_vocab else 0 for w in sentence]
        english_indices.append(word_seq)
    for sentence in french_sentences:
        word_seq = [french_vocab[w] if w in french_vocab else 0 for w in sentence]
        french_indices.append(word_seq)

    return (english_indices, french_indices)

def batch_iter(data, num_epochs=1, batch_size=32, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# create batches
english_sentences, french_sentences = load_data(train_path)
english_indices, french_indices = vectorize_data(english_sentences, french_sentences)
train_data = list(zip(english_indices, french_indices))
batches = batch_iter(train_data)

for batch in batches:
    english = batch[:,0]
    french = batch[:,1]

    print(english)
    print(french)
