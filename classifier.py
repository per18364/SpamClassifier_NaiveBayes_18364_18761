import pandas as pd
import re


data = pd.read_csv('entrenamiento.txt', sep='\t',
                   header=None, names=['label', 'sms'])
# print(data.shape)
# data.head()
# data['label'].value_counts(normalize=True)
# data = data.replace(['ham', 'spam'], [0, 1])

# Randomize dataset
data_random = data.sample(frac=1, random_state=1)

# Splitting 80% Training - 20% Testing
training_test_index = round(len(data_random) * 0.8)

training_set = data_random[:training_test_index].reset_index(drop=True)
test_set = data_random[training_test_index:].reset_index(drop=True)
# print(training_set.shape)
# print(test_set.shape)
# training_set['label'].value_counts(normalize=True)
# test_set['label'].value_counts(normalize=True)

# Cleaning
training_set['sms'] = training_set['sms'].str.replace('\W', ' ')
training_set['sms'] = training_set['sms'].str.lower()
# training_set.head(3)

# Vocabulario
training_set['sms'] = training_set['sms'].str.split()

vocabulary = []
for sms in training_set['sms']:
    for word in sms:
        vocabulary.append(word)

vocabulary = list(set(vocabulary))

# Training Final
word_counts_per_sms = {unique_word: [
    0] * len(training_set['sms']) for unique_word in vocabulary}
for index, sms in enumerate(training_set['sms']):
    for word in sms:
        word_counts_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_sms)
# word_counts.head()

training_set_clean = pd.concat([training_set, word_counts], axis=1)
# training_set_clean.head()

# Calculos
spam_messages = training_set_clean[training_set_clean['label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['label'] == 'ham']

p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

n_words_per_spam_message = spam_messages['sms'].apply(len)
n_spam = n_words_per_spam_message.sum()

n_words_per_ham_message = ham_messages['sms'].apply(len)
n_ham = n_words_per_ham_message.sum()

n_vocabulary = len(vocabulary)
alpha = 1

# Parametros
parameters_spam = {unique_word: 0 for unique_word in vocabulary}
parameters_ham = {unique_word: 0 for unique_word in vocabulary}

for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()
    p_word_given_spam = (n_word_given_spam + alpha) / \
        (n_spam + alpha * n_vocabulary)
    parameters_spam[word] = p_word_given_spam

    n_word_given_ham = ham_messages[word].sum()
    p_word_given_ham = (n_word_given_ham + alpha) / \
        (n_ham + alpha * n_vocabulary)
    parameters_ham[word] = p_word_given_ham


def classify(message):
    message = re.sub('\W', ' ', message)
    message = message.lower().split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]

    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal probabilities, have a human classify this!')


classify('WINNER!! This is the secret code to unlock the money: C3421.')
