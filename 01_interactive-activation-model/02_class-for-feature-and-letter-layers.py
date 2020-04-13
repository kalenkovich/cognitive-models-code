#!/usr/bin/env python
# coding: utf-8

import numpy as np


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


class IAM(object):
    
    def __init__(self):
        self.M = 1.0  # maximum activation
        self.m = -0.2  # minimum activation
        self.theta = 0.07  # decay rate
        self.r_feature = 0  # baseline activation
        self.position_count = 4  # number of letters
        
        self.feature_nodes = np.zeros((self.position_count, feature_count))
        
    def present_word(self, word: str):
        """Show a word to the model"""
        features_present = np.array([features_binary[letter] for letter in word])
        # Set features present in the word to the maximum activation
        self.feature_nodes = self.M * features_present
        
    def run_cycle(self):        
        decay = (self.feature_nodes - self.r_feature) * self.theta
        self.feature_nodes = self.feature_nodes - decay


iam = IAM()
iam.present_word('WORK')

