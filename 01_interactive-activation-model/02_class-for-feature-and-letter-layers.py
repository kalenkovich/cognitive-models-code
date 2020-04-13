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


class IAM(object):
    
    def __init__(self):
        # Parameters
        self.M = 1.0  # maximum activation
        self.m = -0.2  # minimum activation
        self.theta = 0.07  # decay rate
        self.r_feature = 0  # baseline activation of the feature nodes
        self.r_letter = 0  # baseline activation of the letter nodes
        self.position_count = 4  # number of letters
        self.letter_to_feature_excitatory = 0.005
        self.letter_to_feature_inhibitory = 0.15
        
        # Nodes
        self.initialize_nodes()
        
        # Weights
        self.letter_to_letter_weights = np.zeros((self.letter_nodes.size, 
                                                  self.letter_nodes.size))
        is_excitatory = np.array([features_binary[letter] 
                                  for letter 
                                  in sorted(features_binary.keys())]).T
        # For one position
        feature_to_letter_weights_1 = np.where(
            is_excitatory,
            self.letter_to_feature_excitatory,
            - self.letter_to_feature_inhibitory
        )
        # For all positions
        self.feature_to_letter_weights = block_diag(
            *[feature_to_letter_weights_1 for _ in range(4)])
        
    def initialize_nodes(self):
        self.feature_nodes = np.ones((self.position_count, feature_count)) * self.r_feature
        self.letter_nodes = np.ones((self.position_count, letter_count)) * self.r_letter
        
    def present_word(self, word: str):
        """Show a word to the model"""
        features_present = np.array([features_binary[letter] for letter in word])
        # Set features present in the word to the maximum activation
        self.feature_nodes = self.M * features_present
    
    def calculate_decay(self):
        feature_decay = (self.feature_nodes - self.r_feature) * self.theta
        letter_decay = (self.letter_nodes - self.r_letter) * self.theta
        return feature_decay, letter_decay
    
    @staticmethod
    def calculate_layer_to_layer_input(layer_A, layer_B, weights):
        """
        Calculates net input from layer_A to layer_B with weights connecting them.
        This function is necessary because we prefer to keep nodes in 4 x N arrays instead of long vectors
        """
        # Only the active (activation > 0) nodes get to send signals.
        # Inhibitory connections have negative weights in this implementation
        return (layer_A.ravel() * (layer_A.ravel() > 0) @ weights).reshape(layer_B.shape)
    
    def calculate_neighbours_effect(self):
        # There are no connections to the feature level except for the visual input
        feature_neighbours_effect = np.zeros(self.feature_nodes.shape)

        # Equation 1
        # Only the active (activation > 0) nodes get to send signals.

        net_input = (
            self.calculate_layer_to_layer_input(
                self.feature_nodes, 
                self.letter_nodes, 
                self.feature_to_letter_weights)
            + self.calculate_layer_to_layer_input(
                self.letter_nodes, 
                self.letter_nodes, 
                self.letter_to_letter_weights))

        # Equation 2 and 3
        letter_neighbours_effect = np.where(
            net_input > 0,
            net_input * (self.M - self.letter_nodes),
            net_input * (self.letter_nodes - self.m)
        )

        return feature_neighbours_effect, letter_neighbours_effect
    
    def run_cycle(self):        
        feature_decay, letter_decay = self.calculate_decay()
        feature_neighbours_effect, letter_neighbours_effect = self.calculate_neighbours_effect()
        
        self.feature_nodes += - feature_decay + feature_neighbours_effect
        self.letter_nodes += - letter_decay + letter_neighbours_effect
        
    def run_n_cycles(self, n: int):
        for _ in range(n):
            self.run_cycle()
            
    def print_active_letters(self):
        for i in range(4):
            active_letters = np.array(alphabet)[iam.letter_nodes[i] > 0]
            print(f'letter {i+1}: {active_letters}')


iam = IAM()
iam.present_word('WORK')
iam.run_cycle()
iam.print_active_letters()


iam.initialize_nodes()
iam.present_word('WQRK')
iam.run_cycle()
iam.print_active_letters()

