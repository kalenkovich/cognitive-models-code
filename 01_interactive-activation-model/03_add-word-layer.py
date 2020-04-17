#!/usr/bin/env python
# coding: utf-8

import numpy as np


from iam import IAM, Layer, load_corpus, alphabet, Connection


# Load the words

corpus = load_corpus()
print(corpus[:5])

n_words = len(corpus)
print(n_words)


# Letter-word excitation
# Letter-word inhibition
# Word-word inhibition
# Word-letter excitation
# .07
# .04
# .21
# .30

class IAMWithWords1(IAM):
    def __init__(self):
        super().__init__()
        
        self.word_layer = Layer(
            shape=(1188, ),
            resting_activation=0,  # it will be a random number between -0.05 and 0
            minimum_activation=self.m,
            maximum_activation=self.M,
            decay_rate=self.theta
        )
        self._layers.append(self.word_layer)
        
        # Letter-to-word connections
        is_excitatory = np.array([[[word[position].lower() == letter.lower()
                                    for word in corpus.word]
                                   for letter in alphabet]
                                  for position in range(4)])
        is_excitatory = np.reshape(is_excitatory, newshape=(-1, is_excitatory.shape[-1]))
        # For one letter
        letter_to_word_excitatory = 0.07
        letter_to_word_inhibitory = 0.04
        letter_to_word_weights = np.where(
            is_excitatory,
            letter_to_word_excitatory,
            - letter_to_word_inhibitory
        )        
        letter_to_word_connection = Connection(
            layer_from=self.letter_layer,
            layer_to=self.word_layer,
            weights=letter_to_word_weights
        )


iam_with_words_1 = IAMWithWords1()
iam_with_words_1.present_word('WORK')


iam_with_words_1.run_cycle()
iam_with_words_1.print_active_letters()
(corpus
 .assign(activation=iam_with_words_1.word_layer.activations)
 .sort_values(by='activation', ascending=False)
)


iam_with_words_1.run_cycle()
iam_with_words_1.print_active_letters()
(corpus
 .assign(activation=iam_with_words_1.word_layer.activations)
 .sort_values(by='activation', ascending=False)
)


# Why is "cork" more activated than "word"?
