#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np
from matplotlib import pyplot as plt


from iam import IAM, features_binary, alphabet, corpus


iam = IAM()


# # Present the ambiguous input

# We need to present features consistent with both 'WORK' and 'WORR'

# Features that are both in K and R

rk_common_features = [min(K_f, K_r) for (K_f, K_r) in zip(features_binary['K'], features_binary['R'])]
rk_common_features


# All features

ambiguous_input = np.array([
    features_binary['W'],
    features_binary['O'],
    features_binary['R'],
    rk_common_features
])
ambiguous_input


def present_ambiguous_input():
    iam.feature_layer.activations = ambiguous_input.astype(float)


# # Test

def get_letter_activation(position, letter):
    return iam.letter_layer.activations[position, alphabet.index(letter)]


def get_word_activation(word):
    word_index = corpus.word.tolist().index(word.lower())
    return iam.word_layer.activations[word_index]


def take_snapshot():
    for letter, activation_list in letter_activations_history.items():
        activation_list.append(get_letter_activation(position=3, letter=letter))
        
    for word, activation_list in word_activations_history.items():
        activation_list.append(get_word_activation(word))


letter_activations_history = dict(K=[], R = [], D = [])
word_activations_history = dict(WORK=[], WORD=[], WEAK=[], WEAR=[])

iam.reset_nodes()
present_ambiguous_input()

take_snapshot()
for _ in range(40):
    iam.run_cycle()
    take_snapshot()


plt.figure(figsize=(10, 6))
plt.plot(np.array(list(letter_activations_history.values())).T)
plt.legend(list(letter_activations_history.keys()), loc='upper left')
plt.grid()
plt.yticks(np.arange(-0.2, 1.1, 0.1));


# ![](work-word_letter-activation-plots.png)

# - In our simulation, "D" gets uninhibited, in the article, "D" stayst at -0.2
# - In our simulation, "R" gets a bit activated (~0.1) and then decays towards 0, in the article, "R" grows steadily towards ~0.35

plt.figure(figsize=(10, 6))
plt.plot(np.array(list(word_activations_history.values())).T)
plt.legend(list(word_activations_history.keys()), loc='upper left')
plt.grid()
plt.yticks(np.arange(-0.2, 1.1, 0.1));


# ![](work-word_word-activation-plots.png)

# - "WORD" get less activate at the peak (~0.03) than in the article (~0.1).
# - "WEAK" and "WEAR" plateau later in our simulation.
