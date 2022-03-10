import re
import string
import pandas as pd
from paths import *

stop_words = [ 'that','would','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
              "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
              'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which',
              'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'been',
              'being', 'have', 'has', 'had', 'having', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
              'if',
              'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'against',
              'between',
              'into', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'in',
              'out',
              'on', 'under', 'again', 'when', 'where',
              'why',
              'how', 'all', 'any', 'both', 'each', 'few', 'more', 'other', 'such', 'no', 'nor',
              'not',
              'own', 'than', 'too', 'very', 's', 't', 'will', 'don', "don't",
              'should',
              "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
              "couldn't",
              'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
              "isn't",
              'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
              "shouldn't",
              'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

train_file = pd.read_json(train_path, lines=True)
dev_file = pd.read_json(dev_path, lines=True)
# To track the target word because there are too many duplicated instances
uniqueWord = 'bukelimehicbiryerdeyok'

def get_lemma(file): return list(file["lemma"])


def get_ranges(file): return [((s1, e1), (s2, e2)) for s1, e1, s2, e2 in
                              zip(file["start1"], file["end1"], file["start2"], file["end2"])]


ranges = get_ranges(dev_file)
lemmas = get_lemma(dev_file)


# Capitalized versions of the stop words to extend to the stop_words
caps = []
for i in stop_words:
    caps.append(i.capitalize())

stop_words.extend(caps)

# This is combined with preprocessing the entire text
class Vocabulary:
    def __init__(self):

        self.PAD_token = 0
        self.SEP_token = 1
        self.UNK_token = 2
        self.word2index = {}
        self.word2count = {self.PAD_token: 0, self.SEP_token: 0, self.UNK_token: 0}
        self.index2word = {self.PAD_token: "[pad]", self.SEP_token: "[sep]", self.UNK_token: "[unk]"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0
        self.preprocessed_sentences = []
        self.tokenized_sentences = []
        self.new_targets = []

    # Remove punctuations and words that include numbers, ( OVER WORDS )
    def preprocess_steps(self, word):

        word = word.encode("ascii", "ignore").decode()
        word = word.lower()
        word = re.sub('\[.*?\]', '', word)
        punks = string.punctuation
        punks = punks.replace('-', '')
        word = re.sub('[%s]' % re.escape(punks), '', word)
        word = re.sub('\w*\d\w*', '', word)
        return word

    def add_word(self, word):

        word = self.preprocess_steps(word)

        # Ensure uniqueness, ensure to not add if in extended stop_words list
        if word:
            if word not in self.word2index and word not in stop_words:

                # First entry of word into vocabulary
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                # Word exists; increase word count
                if word in stop_words:
                    self.word2count[self.UNK_token] += 1
                else:
                    self.word2count[word] += 1

    def add_sentence(self, sentence, index, which):


        rx = ranges[index][which]
        lm = lemmas[index]
        sentence = sentence[:rx[0]] + uniqueWord + sentence[rx[1]:]



        # lm_extracts = list(filter(lambda x: lm in x, sentence.split()))
        # extract_indexes = sentence.split().index(lm_extracts[0])
        # self.new_targets.append(extract_indexes)
        # Add the preprocessed sentences to the pp sentences list in order to save them to a json in the end

        self.preprocessed_sentences.append(sentence)

        sentence_len = 0

        for word in sentence.split():
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    def tokenize_sentences(self):
        patterns = [r'\'',
                    r'\"',
                    r'\.',
                    r'<br \/>',
                    r',',
                    r'\(',
                    r'\)',
                    r'\!',
                    r'\?',
                    r'\;',
                    r'\:',
                    r'\s+']
        replacements = [' \'  ',
                        '',
                        ' . ',
                        ' ',
                        ' , ',
                        ' ( ',
                        ' ) ',
                        ' ! ',
                        ' ? ',
                        ' ',
                        ' ',
                        ' ']
        patterns_dict = list((re.compile(p), r) for p, r in zip(patterns, replacements))
        tok_sent = []
        # Tokenizes each sentence
        for sentence in self.preprocessed_sentences:
            for pattern_re, replaced_str in patterns_dict:
                sentence = pattern_re.sub(replaced_str, sentence)
                words = sentence.split()
                for i in words:
                    if i not in stop_words:
                        tok_sent.append(self.preprocess_steps(i))
                    else:
                        pass
                return tok_sent
