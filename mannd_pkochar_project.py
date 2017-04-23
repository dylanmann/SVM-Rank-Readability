import subprocess
import xml.etree.ElementTree as ET
import os
from collections import Counter
# from pprint import pprint
from statistics import median, mean
import math
from nltk import sent_tokenize, word_tokenize
from numpy import array, concatenate
import numpy as np
import re

import pyphen

# ================= Preprocessing ===================


def get_all_files(directory):
    if directory == "xml/test":
        return sorted([os.path.join(directory, fn) for fn in next(os.walk(directory))[2]], key=lambda x: int(re.search(r'\d+', x).group()))
    else:
        return sorted([os.path.join(directory, fn) for fn in next(os.walk(directory))[2]])


def load_file_excerpts(filename):
    return [line for line in open(filename, "r")]


# This calls the bash script that makes the file list and calls
# both of the core nlp commands
def get_xml_files():
    command = "run_corenlp.sh"
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()


def flatten(listoflists):
    return [elem for l in listoflists for elem in l]

def convert_to_files():
    with open("/home1/c/cis530/project/data/project_train.txt", 'r') as inFile:
        i = 0
        for line in inFile:
            i += 1
            with open('test_excerpt_' + str(i), "w") as out:
                out.write(line)

    with open("/home1/c/cis530/project/data/project_test.txt", 'r') as inFile:
        i = 0
        for line in inFile:
            i += 1
            with open('train_excerpt_' + str(i), "w") as out:
                out.write(line)

# ================== Unigram Model ==============


class UnigramModel:
    # freqmodel is a filename with word,count on each line
    def __init__(self, freqmodel):
        self.counter = Counter()
        with open(freqmodel) as file:
            for line in file:
                tup = line.rsplit(',', 1)
                self.counter[tup[0]] = int(tup[1])
        self.n = sum(self.counter.values())

    # target_word is a str
    # returns float of the base two log prob of given word
    def logprob(self, target_word):
        return math.log(self.counter[target_word] / self.n, 2)



#  ================== Various Feature Calculations ==================

mannd_pkochar_dic = pyphen.Pyphen(lang='en_US')


def map_word_features(xml_filename, pos_tag_list):
    tags = Counter()
    with open(xml_filename, "r") as file:
        element = ET.parse(file)
        POS_xpath = "./document/sentences/sentence/tokens/token/POS"
        [element.findall(token_xpath)]

    # total counts
    n = sum(tags.values())

    # avoid div by 0
    if (n == 0):
        print("0 sucks")
        return [0 for dep in pos_tag_list]

    # return frequency of each tag in the file
    return [tags[tag] / n for tag in pos_tag_list]


def calculate_word_features(words):
    counter = Counter(words)
    word_lengths = [len(k) for k, v in counter.items() for x in range(0, v)]
    median_word = median(word_lengths)
    average_word = mean(word_lengths)
    avg_sentence_length = mean([len(s) for s in sent_tokenize(excerpt)])
    type_token_ratio = len(counter) / sum(counter.values())
    syllables = [len(mannd_pkochar_dic.inserted(w).split('-')) for w in words]
    prop_few_syll = sum([0 if s < 5 else 1 for s in syllables]) / len(words)
    prop_many_syll = 1 - prop_few_syll

    return [median_word, average_word, prop_few_syll, prop_many_syll,
            avg_sentence_length, type_token_ratio]

# ================== POS =====================


# returns all unique pos tags from all documents in directory
def extract_pos_tags(xml_dir):
    tags = set()
    for filename in get_all_files(xml_dir):
        with open(filename, "r") as file:
            element = ET.parse(file)
            POS_xpath = "./document/sentences/sentence/tokens/token/POS"
            for i in element.findall(POS_xpath):
                tags.add(i.text)

    return sorted(list(tags))


def map_pos_tags(xml_filename, pos_tag_list):
    tags = Counter()
    with open(xml_filename, "r") as file:
        element = ET.parse(file)
        POS_xpath = "./document/sentences/sentence/tokens/token/POS"
        for i in element.findall(POS_xpath):
            tags[i.text] += 1

    # total counts
    n = sum(tags.values())

    # avoid div by 0
    if (n == 0):
        print("0 sucks")
        return [0 for dep in pos_tag_list]

    # return frequency of each tag in the file
    return [tags[tag] / n for tag in pos_tag_list]


def map_universal_tags(feat_vector, pos_tag_list, ptb_map, universal_tag_list):
    freq = Counter()
    for i in range(len(feat_vector)):
        freq[ptb_map[pos_tag_list[i]]] += feat_vector[i]

    return [freq[tag] for tag in universal_tag_list]


def get_google_map():
    with open("en-ptb.map", "r") as file:
        rows = [line.split('\t') for line in file]
        return {row[0]: row[1] for row in rows}


# ============================ Named Entity Tags =======================


# returns all unique NER tags from all documents in directory
def extract_ner_tags(xml_dir):
    tags = set()
    xpath = "./document/sentences/sentence/tokens/token/NER"
    for filename in get_all_files(xml_dir):
        with open(filename, "r") as file:
            element = ET.parse(file)

            for i in element.findall(xpath):
                tags.add(i.text)

    return sorted(list(tags))


def map_named_entity_tags(xml_filename, entity_list):
    tags = Counter()

    xpath = "./document/sentences/sentence/tokens/token/NER"
    with open(xml_filename, "r") as file:
        element = ET.parse(file)
        for i in element.findall(xpath):
            tags[i.text] += 1

    n = sum(tags.values())

    if (n == 0):
        print("0 sucks")
        return [0 for dep in entity_list]

    return [tags[tag] / n for tag in entity_list]

# ============================ Dependency Parsing =======================


def extract_dependencies(xml_dir):
    deps = set()
    xpath = "./document/sentences/sentence/basic-dependencies/dep"
    for filename in get_all_files(xml_dir):
        with open(filename, "r") as file:
            element = ET.parse(file)
            for e in element.findall(xpath):
                deps.add(e.get("type"))

    return sorted(list(deps))


def map_dependencies(xml_filename, dep_list):
    deps = Counter()
    xpath = "./document/sentences/sentence/basic-dependencies/dep"
    with open(xml_filename, "r") as file:
        element = ET.parse(file)
        for e in element.findall(xpath):
            deps[e.get("type")] += 1

    n = sum(deps.values())

    if (n == 0):
        print("0 sucks")
        return [0 for dep in dep_list]

    return [deps[dep] / n for dep in dep_list]


# =========== Syntax Tree Parsing ==============

def pop_stack(stack, rules):
    last = stack.pop()
    child = last.split("_")[0]
    if len(stack) == 0:
        rules.add(last)
        return
    parent = stack.pop()
    parent = parent + "_" + child
    stack.append(parent)
    if len(last.split("_")) > 1:
        rules.add(last)


def parse_file(filename):
    rules = set()
    xpath = "./document/sentences/sentence/parse"
    with open(filename, "r") as file:
        element = ET.parse(file)
        stack = []
        for i in element.findall(xpath):
            for elem in i.text.split():
                if elem[0] == "(":
                    stack.append(elem[1:])
                while elem[-1] == ")":
                    elem = elem[:-1]
                    pop_stack(stack, rules)
        if len(stack) > 0:
            print("stack not empty")
    return rules


def extract_prod_rules(xml_dir):
    rules = set()
    for filename in get_all_files(xml_dir):
        rules.update(parse_file(filename))
    return rules


def map_prod_rules(xml_filename, all_rules):
    rules = set(parse_file(xml_filename))
    return [1 if elem in rules else 0 for elem in all_rules]

# =============== Brown Clustering =======================


def generate_word_cluster_mapping(path):
    with open(path, "r") as file:
        return {line.split("\t")[1]: line.split("\t")[0] for line in file}


def generate_word_cluster_codes(path):
    codes = list(set(generate_word_cluster_mapping(path).values()))
    codes.append("8888")
    return sorted(codes, key=lambda x: int(x))


def map_brown_clusters(xml_file_path, cluster_code_list, word_cluster_mapping):
    clusters = Counter()
    n = 0

    xpath = "./document/sentences/sentence/tokens/token/word"
    with open(xml_file_path, "r") as file:
        element = ET.parse(file)
        for i in element.findall(xpath):
            n += 1
            clusters[word_cluster_mapping.get(i.text, "8888")] += 1

    return [clusters[c] / n for c in cluster_code_list]

# ======================= Creating Features ========================


def createPOSFeat(xml_dir, pos_tag_list):
    files = get_all_files(xml_dir)
    feats = [map_pos_tags(f, pos_tag_list) for f in files]
    return array(feats)


def createUniversalPOSFeat(pos_vecs, p_tags, g_map, uni_tags):
    feats = [map_universal_tags(v, p_tags, g_map, uni_tags) for v in pos_vecs]
    return array(feats)


def createNERFeat(xml_dir, entity_list):
    files = get_all_files(xml_dir)
    return array([map_named_entity_tags(file, entity_list) for file in files])


def createDependencyFeat(xml_dir, dep_list):
    files = get_all_files(xml_dir)
    return array([map_dependencies(f, dep_list) for f in files])


def createSyntaticProductionFeat(xml_dir, all_rules):
    files = get_all_files(xml_dir)
    return array([map_prod_rules(f, all_rules) for f in files])


def createBrownClusterFeat(xml_dir, cluster_codes, wc_map):
    files = get_all_files(xml_dir)
    feats = [map_brown_clusters(f, cluster_codes, wc_map) for f in files]
    return array(feats)

# ======================= Part 9.3 ========================


def make_X(xml_dir):
    brown_file = "brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt"
    pos_tags = open("output/hw3_4-1.txt", "r").read().splitlines()
    entity_list = open("output/hw3_5-1.txt", "r").read().splitlines()
    dep_list = open("output/hw3_6-1.txt", "r").read().splitlines()
    all_rules = open("output/hw3_7-1.txt", "r").read().splitlines()
    cluster_codes = generate_word_cluster_codes(brown_file)
    wc_map = generate_word_cluster_mapping(brown_file)
    g_map = get_google_map()
    uni_tags = sorted(g_map.values())

    pos_vecs = createPOSFeat(xml_dir, pos_tags)

    print("making magic happen for " + xml_dir)
    return concatenate((
        pos_vecs,
        createUniversalPOSFeat(pos_vecs, pos_tags, g_map, uni_tags),
        createNERFeat(xml_dir, entity_list),
        createDependencyFeat(xml_dir, dep_list),
        createSyntaticProductionFeat(xml_dir, all_rules),
        createBrownClusterFeat(xml_dir, cluster_codes, wc_map)), 1)


def make_classifier_run():
    try:
        X_test = np.load("x_test.npy")
    except:
        X_test = make_X("xml/test")
        np.save("x_test", X_test)

    try:
        X_train = np.load("x_train.npy")
    except:
        X_train = make_X("xml/train")
        np.save("x_train", X_train)

    y = array([1 if f[10] == 'g' else 0 for f in get_all_files("xml/train")])

    print("running classifier")
    run_classifier(X_train, y, X_test, "m.txt")


def run_classifier(X_train, y_train, X_test, predicted_labels_file):
    clf = SVC(verbose=True)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    with open(predicted_labels_file, "w") as file:
        for p in predictions:
            file.write(str(p) + "\n")


# ======================= Cross Validation and Actual Stuff =====================

def cross_validation(X_train, y_train):
    clf = SVC(verbose=True)
    clf.fit(X_train, y_train)
    return cross_val_score(clf, X_train, y_train, cv=5)


def cross_validation_nn(X_train, y_train):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),
                        random_state=1)
    clf.fit(X_train, y_train)
    return cross_val_score(clf, X_train, y_train, cv=5)


def experiment_X(xml_dir):
    brown_file = "brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt"
    pos_tags = open("output/hw3_4-1.txt", "r").read().splitlines()
    entity_list = open("output/hw3_5-1.txt", "r").read().splitlines()
    dep_list = open("output/hw3_6-1.txt", "r").read().splitlines()
    all_rules = open("output/hw3_7-1.txt", "r").read().splitlines()
    cluster_codes = generate_word_cluster_codes(brown_file)
    wc_map = generate_word_cluster_mapping(brown_file)
    g_map = get_google_map()
    uni_tags = sorted(g_map.values())

    pos_vecs = createPOSFeat(xml_dir, pos_tags)

    print("making magic happen for " + xml_dir)
    return concatenate((
        createDependencyFeat(xml_dir, dep_list),
        createSyntaticProductionFeat(xml_dir, all_rules),
        createBrownClusterFeat(xml_dir, cluster_codes, wc_map)), 1)


def part_10():
    print("testing")
    try:
        X_test = np.load("x_test.npy")
    except:
        X_test = experiment_X("xml/test")
        np.save("x_test", X_test)

    print("training")
    try:
        X_train = np.load("x_train.npy")
    except:
        X_train = experiment_X("xml/train")
        np.save("x_train", X_train)

    y = array([1 if f[10] == 'g' else 0 for f in get_all_files("xml/train")])

    print("running classifier")

    scores = cross_validation_nn(X_train, y)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# if __name__ == "__main__":
#     pass
