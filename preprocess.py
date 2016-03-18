#!/usr/bin/env python

"""Text Classification Preprocessing
"""

import re
import sys
import codecs
import pandas
import argparse
import yaml

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def parse_arg(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='dataset name')
    return parser.parse_args(argv[1:])

if __name__ == '__main__':
    args = parse_arg(sys.argv)
    dataset = args.dataset
    dfs = []
    with open('corpus.yaml') as f:
        corpus = yaml.load(f)
        corpus_dir = corpus['dir']

    for split, filename in corpus[dataset].items():
        filename = corpus_dir+'/'+filename
        if not filename:
            continue
        labels = []
        sentences = []
        with open(filename) as f:
            for line in f:
                div = line.index(' ')
                sentences.append(clean_str(line[div+1:]))
                labels.append(line[:div])
        dfs.append(pandas.DataFrame({'sentence':sentences, 'label':labels, 'split':split}))

    filename = args.dataset + '.pkl'
    pandas.concat(dfs).to_pickle(filename)
