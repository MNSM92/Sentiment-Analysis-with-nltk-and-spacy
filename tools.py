import os, codecs
import spacy
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.util import mark_negation
nlp = spacy.load("en_core_web_md")
import random


def read_in(folder):
    files = os.listdir(folder)
    a_dict = {}
    for a_file in sorted(files):
        if not a_file.startswith("."):
            with codecs.open(folder + a_file, encoding='ISO-8859-1', errors ='ignore') as f:
                file_id = a_file.split(".")[0].strip()
                a_dict[file_id] = f.read()
            f.close()
    return a_dict


def tokenize(text):
    text.replace("\n", " ")
    return text.split()


def statistics(a_dict):
    length = 0
    sent_length = 0
    num_sents = 0
    vocab = []
    for review in a_dict.values():
        length += len(tokenize(review))
        sents = review.split("\n")
        num_sents += len(sents)
        for sent in sents:
            sent_length += len(tokenize(sent))
        vocab += tokenize(review)
    avg_length = float(length) / len(a_dict)
    avg_sent_length = float(sent_length) / num_sents
    vocab_size = len(set(vocab))
    diversity = float(length) / float(vocab_size)
    return avg_length, avg_sent_length, vocab_size, diversity

def vocab_difference(list1, list2):
    vocab1 = []
    vocab2 = []
    for rev in list1:
        vocab1 += tokenize(rev)
    for rev in list2:
        vocab2 += tokenize(rev)
    return sorted(list(set(vocab1) - set(vocab2)))

def lemmatize(sentence, switch):
    text = nlp(sentence.replace("\n", " "))
    if switch=="on":
        lemmas = [text[i].lemma_ for i in range(len(text))]
        return lemmas
    else:
        tokens = [text[i] for i in range(len(text))]
        return tokens


def spacy_preprocess_reviews(source):
    source_docs = {}
    index = 0
    for review_id in source.keys():
        #to speed processing up, you can disable "ner" – Named Entity Recognition module of spaCy
        source_docs[review_id] = nlp(source.get(review_id).replace("\n", ""), disable=["ner"])
        if index>0 and (index%200)==0:
            print(str(index) + " reviews processed")
        index += 1
    print("Dataset processed")
    return source_docs


def statistics_lem(source_docs):
    length = 0
    vocab = []
    for review_id in source_docs.keys():
        review_doc = source_docs.get(review_id)
        lemmas = []
        for token in review_doc:
            lemmas.append(token.lemma_)
        length += len(lemmas)
        vocab += lemmas
    avg_length = float(length) / len(source_docs)
    vocab_size = len(set(vocab))
    diversity = float(length) / float(vocab_size)
    return avg_length, vocab_size, diversity

def vocab_lem_difference(source_docs1, source_docs2):
    vocab1 = []
    vocab2 = []
    for rev_id in source_docs1.keys():
        rev = source_docs1.get(rev_id)
        for token in rev:
            vocab1.append(token.lemma_)
    for rev_id in source_docs2.keys():
        rev = source_docs2.get(rev_id)
        for token in rev:
            vocab2.append(token.lemma_)
    return sorted(list(set(vocab1) - set(vocab2)))


def vocab_pos_difference(source_docs1, source_docs2, pos):
    vocab1 = []
    vocab2 = []
    for rev_id in source_docs1.keys():
        rev = source_docs1.get(rev_id)
        for token in rev:
            if token.pos_==pos:
                vocab1.append(token.text)
    for rev_id in source_docs2.keys():
        rev = source_docs2.get(rev_id)
        for token in rev:
            if token.pos_==pos:
                vocab2.append(token.text)
    return sorted(list(set(vocab1) - set(vocab2)))


def collect_wordlist(input_file):
    word_dict = {}
    with codecs.open(input_file, encoding='ISO-8859-1', errors ='ignore') as f:
        for a_line in f.readlines():
            cols = a_line.split("\t")
            if len(cols)>2:
                word = cols[0].strip()
                score = float(cols[1].strip())
                word_dict[word] = score
    f.close()
    return word_dict


def bin_decisions(a_dict, label, sent_dict):
    decisions = []
    for rev_id in a_dict.keys():
        score = 0
        for token in a_dict.get(rev_id):
            if token.text in sent_dict.keys():
                if sent_dict.get(token.text) < 0:
                    score -= 1
                else:
                    score += 1
        if score < 0:
            decisions.append((-1, label))
        else:
            decisions.append((1, label))
    return decisions


def weighted_decisions(a_dict, label, sent_dict):
    decisions = []
    for rev_id in a_dict.keys():
        score = 0
        for token in a_dict.get(rev_id):
            if token.text in sent_dict.keys():
                score += sent_dict.get(token.text)
        if score < 0:
            decisions.append((-1, label))
        else:
            decisions.append((1, label))
    return decisions


def get_accuracy(pos_docs, neg_docs, sent_dict):
    decisions_pos = bin_decisions(pos_docs, 1, sent_dict)
    decisions_neg = bin_decisions(neg_docs, -1, sent_dict)
    decisions_all = decisions_pos + decisions_neg
    lists = [decisions_pos, decisions_neg, decisions_all]
    accuracies = []
    for i in range(0, len(lists)):
        match = 0
        for item in lists[i]:
            if item[0] == item[1]:
                match += 1
        accuracies.append(float(match) / float(len(lists[i])))
    return accuracies

def occurrences(a_dict, sent_dict):
    occur = []
    for rev_id in a_dict.keys():
        for token in a_dict.get(rev_id):
            if token.text in sent_dict.keys():
                occur.append(token.text)
    return len(set(occur))


def convert_tags(pos_tag):
    if pos_tag.startswith("JJ"):
        return wn.ADJ
    elif pos_tag.startswith("NN"):
        return wn.NOUN
    elif pos_tag.startswith("RB"):
        return wn.ADV
    elif pos_tag.startswith("VB") or pos_tag.startswith("MD"):
        return wn.VERB
    return None


def swn_decisions(a_dict, label):
    decisions = []
    for rev_id in a_dict.keys():
        score = 0
        neg_count = 0
        pos_count = 0
        for token in a_dict.get(rev_id):
            wn_tag = convert_tags(token.tag_)
            if wn_tag in (wn.ADJ, wn.ADV, wn.NOUN, wn.VERB):
                synsets = list(swn.senti_synsets(token.lemma_, pos=wn_tag))
                if len(synsets) > 0:
                    temp_score = 0.0
                    for synset in synsets:
                        temp_score += synset.pos_score() - synset.neg_score()
                    score += temp_score / len(synsets)
        if score < 0:
            decisions.append((-1, label))
        else:
            decisions.append((1, label))
    return decisions


def get_swn_accuracy(pos_docs, neg_docs):
    decisions_pos = swn_decisions(pos_docs, 1)
    decisions_neg = swn_decisions(neg_docs, -1)
    decisions_all = decisions_pos + decisions_neg
    lists = [decisions_pos, decisions_neg, decisions_all]
    accuracies = []
    for i in range(0, len(lists)):
        match = 0
        for item in lists[i]:
            if item[0]==item[1]:
                match += 1
        accuracies.append(float(match)/float(len(lists[i])))
    return accuracies

def text_filter(a_dict, label, exclude_lists):
    data = []
    for rev_id in a_dict.keys():
        tokens = []
        for token in a_dict.get(rev_id):
            if not token.text in exclude_lists:
                tokens.append(token.text)
                #tokens.append(token.lemma_) # for the use of lemmas instead of word tokens
        data.append((' '.join(tokens), label))
    return data

def prepare_data(pos_docs, neg_docs, exclude_lists):
    data = text_filter(pos_docs, 1, exclude_lists)
    data += text_filter(neg_docs, -1, exclude_lists)
    random.seed(42)
    random.shuffle(data)
    texts = []
    labels = []
    for item in data:
        texts.append(item[0])
        labels.append(item[1])
    return texts, labels

def split(texts, labels, proportion):
    train_data = []
    train_targets = []
    test_data = []
    test_targets = []
    for i in range(0, len(texts)):
        if i < proportion * len(texts):
            train_data.append(texts[i])
            train_targets.append(labels[i])
        else:
            test_data.append(texts[i])
            test_targets.append(labels[i])
    return train_data, train_targets, test_data, test_targets



def text_filter_incl(a_dict, label, include_lists):
    data = []
    for rev_id in a_dict.keys():
        tokens = []
        for token in a_dict.get(rev_id):
            if token.text in include_lists:
                tokens.append(token.text)
        data.append((' '.join(tokens), label))
    return data

def prepare_data_incl(pos_docs, neg_docs, include_lists):
    data = text_filter_incl(pos_docs, 1, include_lists)
    data += text_filter_incl(neg_docs, -1, include_lists)
    random.shuffle(data)
    texts = []
    labels = []
    for item in data:
        texts.append(item[0])
        labels.append(item[1])
    return texts, labels

def text_filter_postag(a_dict, label, pos_lists):
    data = []
    for rev_id in a_dict.keys():
        tokens = []
        for token in a_dict.get(rev_id):
            if token.pos_ in pos_lists:
                tokens.append(token.text)
        data.append((' '.join(tokens), label))
    return data

def prepare_data_postag(pos_docs, neg_docs, pos_lists):
    data = text_filter_postag(pos_docs, 1, pos_lists)
    data += text_filter_postag(neg_docs, -1, pos_lists)
    random.shuffle(data)
    texts = []
    labels = []
    for item in data:
        texts.append(item[0])
        labels.append(item[1])
    return texts, labels


def text_filter_neg(a_dict, label, exclude_lists):
    data = []
    for rev_id in a_dict.keys():
        tokens = []
        for sent in a_dict.get(rev_id).sents:
            neg_tokens = mark_negation(sent.text.split())
            #print (neg_tokens)
            for token in neg_tokens:
                if not token in exclude_lists:
                    tokens.append(token)
        data.append((' '.join(tokens), label))
    return data

def prepare_data_neg(pos_docs, neg_docs, exclude_lists):
    data = text_filter_neg(pos_docs, 1, exclude_lists)
    data += text_filter_neg(neg_docs, -1, exclude_lists)
    random.seed(42)
    random.shuffle(data)
    texts = []
    labels = []
    for item in data:
        texts.append(item[0])
        labels.append(item[1])
    return texts, labels