#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.linalg import block_diag


feature_numbers = {
    'A': [0, 1, 2, 3, 4, 6, 8],
    'B': [2, 3, 4, 5, 7, 8, 9],
    'C': [0, 1, 2, 5],
    'D': [2, 3, 4, 5, 7, 9],
    'E': [0, 1, 2, 5, 6],
    'F': [0, 1, 2, 6],
    'G': [0, 1, 2, 4, 5, 8],
    'H': [0, 1, 3, 4, 6, 8],
    'I': [2, 5, 7, 9],
    'J': [0, 3, 4, 5],
    'K': [0, 1, 6, 11, 12],
    'L': [0, 1, 5],
    'M': [0, 1, 3, 4, 10, 11],
    'N': [0, 1, 3, 4, 10, 12],
    'O': [0, 1, 2, 3, 4, 5],
    'P': [0, 1, 2, 3, 6, 8],
    'Q': [0, 1, 2, 3, 4, 5, 12],
    'R': [0, 1, 2, 3, 6, 8, 12],
    'S': [1, 2, 4, 5, 6, 8],
    'T': [2, 7, 9],
    'U': [0, 1, 3, 4, 5],
    'V': [0, 1, 11, 13],
    'W': [0, 1, 3, 4, 12, 13],
    'X': [10, 11, 12, 13],
    'Y': [9, 10, 11],
    'Z': [2, 5, 11, 13]
}

feature_count = 14
features_binary = {
    letter: [1 if i in feature_list else 0 for i in range(feature_count)]
    for letter, feature_list in feature_numbers.items()}
letter_count = len(list(feature_numbers.keys()))
alphabet = sorted(feature_numbers.keys())


class Connection(object):
    def __init__(self, layer_from, layer_to, weights):
        self.layer_from = layer_from
        self.layer_to = layer_to
        
        # This way weights can be a constant
        weights_shape = (layer_from.size, layer_to.size)
        self.weights = np.ones(weights_shape) * weights
       
        layer_to.add_connection(self)
    
    def calculate_net_input(self):
        activations_from = self.layer_from.activations
        return (
            (activations_from.ravel() * (activations_from.ravel() > 0) @ self.weights)
                .reshape(self.layer_to.shape))


class Layer(object):
    def __init__(self, 
                 shape, 
                 resting_activation,
                 minimum_activation,
                 maximum_activation,
                 decay_rate):
        self.shape = shape
        self.resting_activation = resting_activation
        self.minimum_activation = minimum_activation
        self.maximum_activation = maximum_activation
        self.decay_rate = decay_rate
        self.connections = []
        
        self._activation_delta = None
        
        self.reset()
        
    @property
    def size(self):
        return self.activations.size
        
    def reset(self):
        self.activations = np.ones(self.shape) * self.resting_activation
        
    def calculate_decay(self):
        return (self.activations - self.resting_activation) * self.decay_rate
    
    def calculate_neighbours_effect(self):
        if not self.connections:
            return 0
        
        net_input = sum(
            connection.calculate_net_input()
            for connection in self.connections)
        
        return np.where(
            net_input > 0,
            net_input * (self.maximum_activation - self.activations),
            net_input * (self.activations - self.minimum_activation)
        )
    
    def calculate_activation_delta(self):
        self._activation_delta = - self.calculate_decay() + self.calculate_neighbours_effect()
    
    def update_activations(self):        
        self.activations += self._activation_delta
        self._activation_delta = None
        
    def add_connection(self, connection: Connection):
        self.connections.append(connection)


class FeatureLayer(Layer):
    def present_word(self, word):
        """Show a word to the model"""
        features_present = np.array([features_binary[letter] for letter in word])
        # Set features present in the word to the maximum activation
        self.activations = self.maximum_activation * features_present


class LetterLayer(Layer):
    def print_active_letters(self):
        for i in range(self.shape[0]):
            active_letters = np.array(alphabet)[self.activations[i] > 0]
            print(f'letter {i+1}: {active_letters}')


class IAM(object):
    
    def __init__(self):
        # Parameters
        self.position_count = 4  # number of letters
        
        # Layers
        M = 1.0  # maximum activation
        m = -0.2  # minimum activation
        theta = 0.07  # decay rate
        
        self.feature_layer = FeatureLayer(
            shape=(self.position_count, feature_count),
            resting_activation=0,
            minimum_activation=m,
            maximum_activation=M,
            decay_rate=theta)
        
        self.letter_layer = LetterLayer(
            shape=(self.position_count, letter_count),
            resting_activation=0,
            minimum_activation=m,
            maximum_activation=M,
            decay_rate=theta)
        
        # Connections
        letter_to_letter_connection = Connection(
            layer_from=self.letter_layer,
            layer_to=self.letter_layer,
            weights=0
        )
        
        # Each feature excites letters that contain it and inhibits those that don't
        is_excitatory = np.array([features_binary[letter] 
                                  for letter 
                                  in sorted(features_binary.keys())]).T
        # For one letter
        feature_to_letter_excitatory = 0.005
        feature_to_letter_inhibitory = 0.15
        feature_to_letter_weights_1 = np.where(
            is_excitatory,
            feature_to_letter_excitatory,
            - feature_to_letter_inhibitory
        )
        # For all letters
        feature_to_letter_weights = block_diag(
            *[feature_to_letter_weights_1 for _ in range(4)])
        feature_to_letter_connection = Connection(
            layer_from=self.feature_layer,
            layer_to=self.letter_layer,
            weights=feature_to_letter_weights
        )
        
    @property
    def layers(self):
        return (self.feature_layer, self.letter_layer)
    
    def reset_nodes(self):
        for layer in self.layers:
            layer.reset()
        
    def present_word(self, word: str):
        """Show a word to the model"""
        self.feature_layer.present_word(word)
    
    def run_cycle(self):        
        for layer in self.layers:
            layer.calculate_activation_delta()
        for layer in self.layers:
            layer.update_activations()
        
    def run_n_cycles(self, n: int):
        for _ in range(n):
            self.run_cycle()
            
    def print_active_letters(self):
        self.letter_layer.print_active_letters()


iam = IAM()
iam.present_word('WORK')
iam.run_cycle()
iam.print_active_letters()


iam.reset_nodes()
iam.present_word('WQRK')
iam.run_cycle()
iam.print_active_letters()


# We've copied the code to the `iam` module. Let's test the version imported from there.

from iam import IAM as IAMTest


iam_test = IAMTest()
iam_test.present_word('WORK')
iam_test.run_cycle()
iam_test.print_active_letters()


iam_test.reset_nodes()
iam_test.present_word('WQRK')
iam_test.run_cycle()
iam_test.print_active_letters()

