import string
from tools import *
import spacy
import nltk
from nltk.corpus import sentiwordnet as swn
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_list
nltk.download('wordnet')
nltk.download('sentiwordnet')
nlp = spacy.load("en_core_web_md")

folder = "review_polarity/txt_sentoken/"
pos_dict = read_in(folder + "pos/")
neg_dict = read_in(folder + "neg/")
categories = ["Positive", "Negative"]

rows = []
rows.append(["Category", "Avg_Len(Review)", "Avg_Len(Sent)", "Vocabulary Size", "Diversity"])
stats = {}
stats["Positive"] = statistics(pos_dict)
stats["Negative"] = statistics(neg_dict)
for cat in categories:
    rows.append([cat, f"{stats.get(cat)[0]:.6f}",
                 f"{stats.get(cat)[1]:.6f}",
                 f"{stats.get(cat)[2]:.6f}",
                 f"{stats.get(cat)[3]:.6f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))

pos_wordlist = pos_dict.values()
neg_wordlist = neg_dict.values()

print(str(len(vocab_difference(pos_wordlist, neg_wordlist))) + " unique words in positive reviews only")
print(str(len(vocab_difference(neg_wordlist, pos_wordlist))) + " unique words in negative reviews only")
print(lemmatize(pos_dict.get(next(iter(pos_dict))), "on")[:200])
print(lemmatize(pos_dict.get(next(iter(pos_dict))), "off")[:200])

pos_docs = spacy_preprocess_reviews(pos_dict)
neg_docs = spacy_preprocess_reviews(neg_dict)

categories = ["Positive", "Negative"]
rows = []
rows.append(["Category", "Avg_Len(Review)", "Vocabulary Size", "Diversity"])
stats = {}
stats["Positive"] = statistics_lem(pos_docs)
stats["Negative"] = statistics_lem(neg_docs)
for cat in categories:
    rows.append([cat, f"{stats.get(cat)[0]:.6f}",
                 f"{stats.get(cat)[1]:.6f}",
                 f"{stats.get(cat)[2]:.6f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))

print(str(len(vocab_lem_difference(pos_docs, neg_docs))) + " unique lemmas in positive reviews only")
print(str(len(vocab_lem_difference(neg_docs, pos_docs))) + " unique lemmas in negative reviews only")

categories = ["Positive", "Negative"]
rows = []
rows.append(["Category", "Unique adj's", "Unique adv's"])
stats = {}
stats["Positive"] = (len(vocab_pos_difference(pos_docs, neg_docs, "ADJ")),
                     len(vocab_pos_difference(pos_docs, neg_docs, "ADV")))
stats["Negative"] = (len(vocab_pos_difference(neg_docs, pos_docs, "ADJ")),
                     len(vocab_pos_difference(neg_docs, pos_docs, "ADV")))
for cat in categories:
    rows.append([cat, f"{stats.get(cat)[0]:.6f}",
                f"{stats.get(cat)[1]:.6f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))

adj_90 = collect_wordlist("sentiment_words/adjectives/1990.tsv")
print(adj_90.get("cool"))
print(len(adj_90))
adj_00 = collect_wordlist("sentiment_words/adjectives/2000.tsv")
print(adj_00.get("cool"))
print(len(adj_00))
all_90 = collect_wordlist("sentiment_words/frequent_words/1990.tsv")
print(len(all_90))
all_00 = collect_wordlist("sentiment_words/frequent_words/2000.tsv")
print(len(all_00))
movie_words = collect_wordlist("sentiment_words/subreddits/movies.tsv")
print(len(movie_words))

categories = ["Adj_90", "Adj_00", "All_90", "All_00", "Movies"]
rows = []
rows.append(["List", "Acc(positive)", "Acc(negative)", "Acc(all)"])
accs = {}
accs["Adj_90"] = get_accuracy(pos_docs, neg_docs, adj_90)
accs["Adj_00"] = get_accuracy(pos_docs, neg_docs, adj_00)
accs["All_90"] = get_accuracy(pos_docs, neg_docs, all_90)
accs["All_00"] = get_accuracy(pos_docs, neg_docs, all_00)
accs["Movies"] = get_accuracy(pos_docs, neg_docs, movie_words)
for cat in categories:
    rows.append([cat, f"{accs.get(cat)[0]:.6f}",
                 f"{accs.get(cat)[1]:.6f}",
                 f"{accs.get(cat)[2]:.6f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))

categories = ["Adj_90", "Adj_00", "All_90", "All_00", "Movies"]
rows = []
rows.append(["List", "Occurs(pos)", "Occurs(neg)"])
occs = {}
occs["Adj_90"] = occurrences(pos_docs, adj_90), occurrences(neg_docs, adj_90)
occs["Adj_00"] = occurrences(pos_docs, adj_00), occurrences(neg_docs, adj_00)
occs["All_90"] = occurrences(pos_docs, all_90), occurrences(neg_docs, all_90)
occs["All_00"] = occurrences(pos_docs, all_00), occurrences(neg_docs, all_00)
occs["Movies"] = occurrences(pos_docs, movie_words), occurrences(neg_docs, movie_words)
for cat in categories:
    rows.append([cat, f"{occs.get(cat)[0]:.6f}",
                f"{occs.get(cat)[1]:.6f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))

print(list(swn.senti_synsets('joy')))
print(list(swn.senti_synsets('trouble')))

joy1 = swn.senti_synset('joy.n.01')
joy2 = swn.senti_synset('joy.n.02')

trouble1 = swn.senti_synset('trouble.n.03')
trouble2 = swn.senti_synset('trouble.n.04')

categories = ["Joy1", "Joy2", "Trouble1", "Trouble2"]
rows = []
rows.append(["List", "Positive score", "Negative Score"])
accs = {}
accs["Joy1"] = [joy1.pos_score(), joy1.neg_score()]
accs["Joy2"] = [joy2.pos_score(), joy2.neg_score()]
accs["Trouble1"] = [trouble1.pos_score(), trouble1.neg_score()]
accs["Trouble2"] = [trouble2.pos_score(), trouble2.neg_score()]
for cat in categories:
    rows.append([cat, f"{accs.get(cat)[0]:.3f}",
                f"{accs.get(cat)[1]:.3f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))

synsets = swn.senti_synsets('terrific', 'a')
for synset in synsets:
    print("pos: +" + str(synset.pos_score()) + " neg: -" + str(synset.neg_score()))

accuracies = get_swn_accuracy(pos_docs, neg_docs)

rows = []
rows.append(["List", "Acc(positive)", "Acc(negative)", "Acc(all)"])
rows.append(["SentiWordNet", f"{accuracies[0]:.6f}",
                f"{accuracies[1]:.6f}",
                f"{accuracies[2]:.6f}"])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))


punctuation_list = [punct for punct in string.punctuation]
texts, labels = prepare_data(pos_docs, neg_docs, list(stopwords_list) + punctuation_list)

print(len(texts), len(labels))
print(texts[0])

train_data, train_targets, test_data, test_targets = split(texts, labels, 0.8)

print(len(train_data))  # 1600?
print(len(train_targets))  # 1600?
print(len(test_data))  # 400?
print(len(test_targets))  # 400?
print(train_targets[:10])  # print out the targets for the first 10 training reviews
print(test_targets[:10])  # print out the targets for the first 10 test reviews

count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(train_data)
# print(train_counts.shape)
#
# print(train_counts[:11])
#
# print(count_vect.inverse_transform(train_data))

transformer = Binarizer()
train_bin = transformer.fit_transform(train_counts)
print(train_bin.shape)
print(train_bin[0])

clf = MultinomialNB().fit(train_counts, train_targets)
test_counts = count_vect.transform(test_data)
predicted = clf.predict(test_counts)

for text, label in list(zip(test_data, predicted))[:10]:
    if label==1:
        print('%r => %s' % (text[:100], "pos"))
    else:
        print('%r => %s' % (text[:100], "neg"))

text_clf = Pipeline([('vect', CountVectorizer(min_df=10, max_df=0.5)),
                     ('binarizer', Binarizer()), # include this for detecting presence-absence of features
                     ('clf', MultinomialNB())
                    ])

text_clf.fit(train_data, train_targets)
print(text_clf)
predicted = text_clf.predict(test_data)

print("\nConfusion matrix:")
print(metrics.confusion_matrix(test_targets, predicted))
print(metrics.classification_report(test_targets, predicted))

scores = cross_val_score(text_clf, texts, labels, cv=10)
print(scores)
print("Accuracy: " + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, texts, labels, cv=10)
print("\nConfusion matrix:")
print(metrics.confusion_matrix(labels, predicted))
print(metrics.classification_report(labels, predicted))


text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                     ('binarizer', Binarizer()), #presence-absence of features
                     ('clf', MultinomialNB())
                    ])

text_clf.fit(train_data, train_targets)
print(text_clf)
predicted = text_clf.predict(test_data)

print("\nConfusion matrix:")
print(metrics.confusion_matrix(test_targets, predicted))
print(metrics.classification_report(test_targets, predicted))

scores = cross_val_score(text_clf, texts, labels, cv=10)
print(scores)
print("Accuracy: " + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, texts, labels, cv=10)
print("\nConfusion matrix:")
print(metrics.confusion_matrix(labels, predicted))
print(metrics.classification_report(labels, predicted))

#texts_incl, labels_incl = prepare_data(pos_docs, neg_docs, set(list(adj_90.keys()) + list(adj_00.keys())))
texts_incl, labels_incl = prepare_data_incl(pos_docs, neg_docs,
                                       set(list(adj_90.keys()) + list(adj_00.keys()) +
                                       list(all_90.keys()) + list(all_00.keys()) + list(movie_words.keys())))

print(len(texts_incl), len(labels_incl))
print(texts_incl[0])

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     #options: ngram_range=(1, 2) or min_df=10, max_df=0.5
                     ('binarizer', Binarizer()), #presence-absence of features
                     ('clf', MultinomialNB()),
                    ])

print(text_clf)


scores = cross_val_score(text_clf, texts_incl, labels_incl, cv=10)
print(scores)
print("Accuracy: " + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, texts_incl, labels_incl, cv=10)
print("\nConfusion matrix:")
print(metrics.confusion_matrix(labels_incl, predicted))
print(metrics.classification_report(labels_incl, predicted))

texts_postag, labels_postag = prepare_data_postag(pos_docs, neg_docs, ["ADJ", "ADV"])

print(len(texts_postag), len(labels_postag))
print(texts_postag[0])

scores = cross_val_score(text_clf, texts_postag, labels_postag, cv=10)
print(scores)
print("Accuracy: " + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, texts_postag, labels_postag, cv=10)
print("\nConfusion matrix:")
print(metrics.confusion_matrix(labels_postag, predicted))
print(metrics.classification_report(labels_postag, predicted))

texts_neg, labels_neg = prepare_data_neg(pos_docs, neg_docs, punctuation_list)

print(len(texts_neg), len(labels_neg))
print(texts_neg[0])

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     #options: ngram_range=(1, 2) or min_df=10, max_df=0.5
                     ('binarizer', Binarizer()), #presence-absence of features
                     ('clf', MultinomialNB()),
                    ])

scores = cross_val_score(text_clf, texts_neg, labels_neg, cv=10)
print(scores)
print("Accuracy: " + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, texts_neg, labels_neg, cv=10)
print("\nConfusion matrix:")
print(metrics.confusion_matrix(labels_neg, predicted))
print(metrics.classification_report(labels_neg, predicted))
