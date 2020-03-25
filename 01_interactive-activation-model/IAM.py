#!/usr/bin/env python
# coding: utf-8

import pandas as pd


# # Load four letter words 

# Kucera & Francis Word Pool downloaded from http://memory.psych.upenn.edu/files/wordpools/kfpool.txt
kf_corpus = pd.read_csv('kfpool.txt', header=None, sep=' ', names=['word', 'frequency'])

kf_corpus


# Let's see how many four letter words there are in the corpus.

(kf_corpus.word.str.len() == 4).sum()


# That is more than 1179 words reported in McClelland & Rumelhart, 1981. Probably they filtered by frequency.

for i in range(7):
    word_count = len(kf_corpus[(kf_corpus.word.str.len() == 4) & (kf_corpus.frequency > i)])
    print(f'There are {word_count} four letter words with frequency larger than {i}')


# Frequency threshold of 4 yields the number closest to 1179.

four_letter_words = kf_corpus[(kf_corpus.word.str.len() == 4) & (kf_corpus.frequency > 4)]
len(four_letter_words)

