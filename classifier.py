import pandas as pd
import re
import random


data = pd.read_csv("entrenamiento.txt", sep="\t",
                   header=None, names=["label", "sms"])

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

train_data["sms"] = train_data["sms"].apply(
    lambda x: re.sub("[^a-zA-Z\s]", "", x).lower())

vocabulary = set()
for sms in train_data["sms"]:
    words = sms.split()
    vocabulary.update(words)

word_counts = {word: [0, 0] for word in vocabulary}
for i, sms in train_data.iterrows():
    words = sms["sms"].split()
    for word in words:
        word_counts[word][0 if sms["label"] == "ham" else 1] += 1

p_spam = len(train_data[train_data["label"] == "spam"]) / len(train_data)
p_ham = 1 - p_spam

n_words_spam = sum([count[1] for count in word_counts.values()])
n_words_ham = sum([count[0] for count in word_counts.values()])

parameters_spam = {word: (count[1] + 1) / (n_words_spam + len(vocabulary))
                   for word, count in word_counts.items()}
parameters_ham = {word: (count[0] + 1) / (n_words_ham + len(vocabulary))
                  for word, count in word_counts.items()}


test_data["sms"] = test_data["sms"].apply(
    lambda x: re.sub("[^a-zA-Z\s]", "", x).lower())


def predict(sms):
    words = sms.split()
    p_spam_given_sms = p_spam
    p_ham_given_sms = p_ham
    for word in words:
        if word in parameters_spam:
            p_spam_given_sms *= parameters_spam[word]
        if word in parameters_ham:
            p_ham_given_sms *= parameters_ham[word]
    if p_spam_given_sms > p_ham_given_sms:
        return "spam"
    else:
        return "ham"


train_data["predicted_label"] = train_data["sms"].apply(predict)
test_data["predicted_label"] = test_data["sms"].apply(predict)


accuracy_train = (train_data["label"] == train_data["predicted_label"]).mean()
print("Training accuracy:", accuracy_train)


accuracy = (test_data["label"] == test_data["predicted_label"]).mean()
print("Test Accuracy:", accuracy)


def classify_message(message):
    # Clean the message
    message = re.sub("[^a-zA-Z\s]", "", message).lower()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    for word in message.split():
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]

    if p_spam_given_message > p_ham_given_message:
        return "spam"
    else:
        return "ham"


while True:
    message = input("Enter a message to classify (or 'quit' to exit): ")
    if message == "quit":
        break
    else:
        label = classify_message(message)
        print("Predicted label:", label)
