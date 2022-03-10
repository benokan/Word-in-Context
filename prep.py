from Vocabulary import *
import pandas as pd

train_file = pd.read_json(train_path, lines=True)

dev_file = pd.read_json(dev_path, lines=True)

def get_sentences(file): return [(s1, s2) for s1, s2 in zip(file["sentence1"], file["sentence2"])]


def get_ranges(file): return [((s1, e1), (s2, e2)) for s1, e1, s2, e2 in
                              zip(file["start1"], file["end1"], file["start2"], file["end2"])]


def get_labels(file): return list(file["label"])


def get_lemma(file): return list(file["lemma"])


ranges_dev = get_ranges(dev_file)
lemmas_dev = get_lemma(dev_file)
s_dev = get_sentences(dev_file)

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



# Tokenizes each sentence
def tokenize(sentence):
    sentence = sentence.lower()
    for pattern_re, replaced_str in patterns_dict:
        sentence = pattern_re.sub(replaced_str, sentence)

    return sentence.split()


sep = "[sep]"


# Combines into the form :  [[s1 + <SEP> + s2],[s1 + <SEP> + s2], ..... ]
def combined_sentences(sentences):
    combined = []
    for i1, i2 in sentences:
        temp = i1
        temp += " " + sep + " "
        temp += i2
        combined.append(temp)

    return combined


def tokenized_combo_sentences(combo_sent):
    x = []
    for i in combo_sent:
        x.append(tokenize(i))
    return x


vocabulary = Vocabulary()


for e, (i1, i2) in enumerate(s_dev):
    vocabulary.add_sentence(i1, e, 0)
    vocabulary.add_sentence(i2, e, 1)

# print(vocabulary.num_words)

# print(vocabulary.word2index)
# all_tokenized_sentences = []



# TODO: Track the changing indexes of the target words here
#
def tokenize_sentences(preproc_sentence: string):
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

    # print(preproc_sentence.split()[vocabulary.new_targets[0]])
    #
    # print(vocabulary.new_targets[0])

    for pattern_re, replaced_str in patterns_dict:
        sentence = pattern_re.sub(replaced_str, preproc_sentence)
        words = sentence.split()

        for i in words:
            if i not in stop_words:
                tok_sent.append(vocabulary.preprocess_steps(i))

            else:
                pass

        return tok_sent


ats = []
target_indices = []


# list and filter stuff is to remove empty strings that are happening due to some messy data cleaning steps


for i in vocabulary.preprocessed_sentences:
    ats.append(list(filter(None, tokenize_sentences(i))))
# New position of the target word
# print(ats[-1].index(lemmas[0]))
# print(ats[-1])
#

grouped_sentences = [ats[n:n+2] for n in range(0, len(ats), 2)]


for k,i in enumerate(grouped_sentences):
    # I was looking for substrings before uniqueWord idea...
    # Result doesn't change but implementation is unnecessarily complex this way
    try:
        extracts1 = list(filter(lambda x: uniqueWord.lower() in x, i[0]))
        extracts2 = list(filter(lambda x: uniqueWord.lower() in x, i[1]))

        target_indices.append([i[0].index(extracts1[0]),i[1].index(extracts2[0])])
    except:
        print(lemmas_dev[k])
        print(i[0])
        print(extracts1[0])


# Swap original lemma with the unique word
for x,sentence in enumerate(grouped_sentences):
    sentence[0][target_indices[x][0]] = lemmas_dev[x]
    sentence[1][target_indices[x][1]] = lemmas_dev[x]




# Written 8K data points grouped by 2 sentences
import csv





# with open('dev_sentences.txt', 'w') as f:
#     for item in grouped_sentences:
#         f.write("%s\n" % item)
#
# with open('dev_targets.txt', 'w') as f:
#     for item in target_indices:
#         f.write("%s\n" % item)

labels = get_labels(dev_file)

with open('dev_labels.txt', 'w') as f:
    for item in labels:
        f.write("%s\n" % item)

