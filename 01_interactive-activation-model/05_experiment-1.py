#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt


from iam.added_absence_detectors import IAM, corpus


iam = IAM()


# # Bright target/patterned mask

def present_mask_at(iam, position):
    patterned_mask = np.hstack((np.ones((1, 9)), np.zeros((1, 14 - 9))))
    iam.feature_layer.activations[position, :] = patterned_mask
    iam.absence_detector_layer.activations[position, :] = 1 - patterned_mask


def present_mask(iam):
    for position in range(4):
        present_mask_at(iam, position)


def present_nothing_at(iam, position):
    iam.feature_layer.activations[position, :] = np.zeros(14)
    iam.absence_detector_layer.activations[position, :] = np.zeros(14)


def present_letter(iam, word, position):
    iam.present_word(word)
    for j in range(4):
        if j == position:
            continue
        present_nothing_at(iam, j)


def run_bright_trial(iam, word, letter_position=None):
    letter_activations_history = dict(K=[], R = [], D = [], E = [])
    word_activations_history = dict(READ=[], DEAL=[])
    active_words = set()
    
    def take_snapshot():
        for letter, activation_list in letter_activations_history.items():
            activation_list.append(iam.get_letter_activation(position=1, letter=letter))
        
        for word, activation_list in word_activations_history.items():
            activation_list.append(iam.get_word_activation(word))
            
        active_words.update(corpus[iam.word_layer.activations > 0].word.tolist())
    
    n_cycles = 40
    n_stim_cycles = 15
    iam.reset_nodes()
    take_snapshot()
    
    for i in range(n_cycles):
        if i < n_stim_cycles:
            if letter_position is None:
                iam.present_word(word)
            else:
                present_letter(iam, word, letter_position)
        else:
            present_mask(iam)

        iam.run_cycle()
        take_snapshot()
    
    return letter_activations_history, word_activations_history, active_words


# ## Word condition 

letter_activations_history, word_activations_history, active_words = run_bright_trial(iam, 'READ', letter_position=None)


plt.figure(figsize=(10, 6))
plt.plot(np.array(list(letter_activations_history.values())).T)
plt.legend(list(letter_activations_history.keys()), loc='upper left')
plt.grid()
plt.yticks(np.arange(-0.2, 1.1, 0.1));


plt.figure(figsize=(10, 6))
plt.plot(np.array(list(word_activations_history.values())).T)
plt.legend(list(word_activations_history.keys()), loc='upper left')
plt.grid()
plt.yticks(np.arange(-0.2, 1.1, 0.1));


print(active_words)


# ## Letter with number signs

letter_activations_history, word_activations_history, active_words = run_bright_trial(iam, 'READ', letter_position=1)


plt.figure(figsize=(10, 6))
plt.plot(np.array(list(letter_activations_history.values())).T)
plt.legend(list(letter_activations_history.keys()), loc='upper left')
plt.grid()
plt.yticks(np.arange(-0.2, 1.1, 0.1));


plt.figure(figsize=(10, 6))
plt.plot(np.array(list(word_activations_history.values())).T)
plt.legend(list(word_activations_history.keys()), loc='upper left')
plt.grid()
plt.yticks(np.arange(-0.2, 1.1, 0.1));


print(active_words)


# The reason "E" got activate more strongly here than in the article is that more word were activated. In turn, more word were activated because in our model, the resting state is zero.

# ## Letter with number signs - negative resting state of words

iam_negative_resting_state = IAM()
iam_negative_resting_state.word_layer.resting_activation = -0.4


letter_activations_history, word_activations_history, active_words = run_bright_trial(iam_negative_resting_state, 'READ', letter_position=1)


plt.figure(figsize=(10, 6))
plt.plot(np.array(list(letter_activations_history.values())).T)
plt.legend(list(letter_activations_history.keys()), loc='upper left')
plt.grid()
plt.yticks(np.arange(-0.2, 1.1, 0.1));


plt.figure(figsize=(10, 6))
plt.plot(np.array(list(word_activations_history.values())).T)
plt.legend(list(word_activations_history.keys()), loc='upper left')
plt.grid()
plt.yticks(np.arange(-0.2, 1.1, 0.1));


print(active_words)


# We got the right pattern of activation of "E" but we had to set the resting state to -0.4. In the article, the resting state is between -0.05 and 0.
