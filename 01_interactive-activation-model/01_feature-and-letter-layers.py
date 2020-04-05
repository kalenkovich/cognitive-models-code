#!/usr/bin/env python
# coding: utf-8

import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import pyplot as plt
from matplotlib import animation as anim, rc
import numpy as np
from scipy.linalg import block_diag


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


# # Encode letters as feature bundles 

# In the original model, the input letters came from a simplified font in which each letter is composed of a number of simplified line strokes (*features*) as in the following image:
# 
# ![Font from Rumelhar & Siple, 1974](rumelhart-siple-font.jpg)

# Here are all the features numbered from 0 to 13:
# 
# ![Numbered line features](line_features.png)

# A list of letters specifying which features it is composed of.

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


# Let's draw all the letter to check that we got everything right. First, we need the coordinates of all features.

feature_coordinates = {
    0: [(-1, -1), (-1, 0)],
    1: [(-1, 0), (-1, 1)],
    2: [(-1, 1), (1, 1)],
    3: [(1, 1), (1, 0)],
    4: [(1, 0), (1, -1)],
    5: [(1, -1), (-1, -1)],
    6: [(0, 0), (-1, 0)],
    7  : [(0, 0), (0, 1)],
    8 : [(0, 0), (1, 0)],
    9 : [(0, 0), (0, -1)],
    10 : [(0, 0), (-1, 1)],
    11 : [(0, 0), (1, 1)],
    12 : [(0 ,0), (1, -1)],
    13 : [(0, 0), (-1, -1)]
}


# Function that draws one letter:

def draw_letter_features(feature_list, feature_coordinates, axes, color='k'):
    axes.grid()
    axes.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    
    # Remove ticks and labels from the axes
    axes.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
   
    for line_nmbr in feature_list:
        line_coords = feature_coordinates[line_nmbr]
        x_values = [line_coords[0][0], line_coords[1][0]]
        y_values = [line_coords[0][1], line_coords[1][1]]
        axes.plot(x_values, y_values, color = color, linewidth = 4)


fig = plt.figure()
axes = fig.gca()
draw_letter_features(feature_list=feature_numbers['A'], 
            feature_coordinates=feature_coordinates, 
            axes=axes)


# Now, let's draw all of them.

def draw_letter_features_all(letter_list, feature_numbers, feature_coordinates):
    N_letters = len(letter_list)
    fig, axs = plt.subplots(3, 9)
    for axes, lttr in zip(axs.flatten()[:N_letters], letter_list):
        plt.sca(axes)
        draw_letter_features(feature_numbers[lttr], feature_coordinates, axes)
    
    # Clear the last empty axes
    axs.flatten()[-1].axis('off');
    
draw_letter_features_all(sorted(feature_numbers.keys())[:26], feature_numbers, feature_coordinates)


# ![Font from Rumelhar & Siple, 1974](rumelhart-siple-font.jpg)

# Looks correct.

# # Feature perception

M = 1.0  # max activation
m = -0.2  # min activation
theta = 0.07  # decay rate
r_feature = 0  # resting state activation


feature_count = len(list(feature_coordinates.keys()))
position_count = 4
feature_nodes = np.zeros((position_count, feature_count))


# We'll have to represent the letters as a list of binary feature flags.

features_binary = {
    letter: [1 if i in feature_list else 0 for i in range(feature_count)]
    for letter, feature_list in feature_numbers.items()}

features_binary


def present_word(word: str):
    """
    Activates features corresponding to the letters in the word
    """
    global feature_nodes
    features_present = np.array([features_binary[letter] for letter in word])
    feature_nodes = M * features_present


def f():
    global feature_nodes
    decay = (feature_nodes - r_feature) * theta
    feature_nodes = feature_nodes - decay


def draw_features(fig_axs=None, title=None):
    if fig_axs is None:
        fig, axs = plt.subplots(ncols=4, nrows=1)
    else:
        fig, axs = fig_axs
    
    fig.suptitle(title)
    
    for pos_features, axes in zip(feature_nodes, axs):
        a = max(pos_features)  # activation level
        color = (1 - a, 1 - a, 1 - a)
        draw_letter_features(feature_list=np.nonzero(pos_features)[0], 
                    feature_coordinates=feature_coordinates, 
                    axes=axes,
                    color=color)


np.set_printoptions(precision=2, floatmode='fixed')
present_word('WORK')
print(feature_nodes)

fig, axs = plt.subplots(ncols=4, nrows=1)
for t in range(10):
    f()
    draw_features(fig_axs=(fig, axs), title=f'\nafter {t + 1} cycles:')
    fig.canvas.draw()


# # Letter layer

letter_count = len(list(feature_numbers.keys()))
position_count = 4
r_letter = 0
letter_nodes = np.ones((position_count, letter_count)) * r_letter


def run_cycle():
    global feature_nodes, letter_nodes
    
    feature_decay, letter_decay = calculate_decay()
    feature_neighbours_effect, letter_neighbours_effect = calculate_neighbours_effect()
    
    feature_nodes += - feature_decay + feature_neighbours_effect
    letter_nodes += - letter_decay + letter_neighbours_effect


def calculate_decay():
    feature_decay = (feature_nodes - r_feature) * theta
    letter_decay = (letter_nodes - r_letter) * theta
    return feature_decay, letter_decay


def calculate_layer_to_layer_input(layer_A, layer_B, weights):
    """
    Calculates net input from layer_A to layer_B with weights connecting them.
    This function is necessary because we prefer to keep nodes in 4 x N arrays instead of long vectors
    """
    # Only the active (activation > 0) nodes get to send signals.
    # Inhibitory connections have negative weights in this implementation
    return (layer_A.ravel() * (layer_A.ravel() > 0) @ weights).reshape(layer_B.shape)


def calculate_neighbours_effect():
    # There are no connections to the feature level except for the visual input
    feature_neighbours_effect = np.zeros(feature_nodes.shape)
    
    # Equation 1
    # Only the active (activation > 0) nodes get to send signals.
    
    # This won't work! feature_nodes and letter_nodes would need to be vectors for this
    net_input = (calculate_layer_to_layer_input(feature_nodes, letter_nodes, feature_to_letter_weights)
               + calculate_layer_to_layer_input(letter_nodes, letter_nodes, letter_to_letter_weights))
    
    # Equation 2 and 3
    letter_neighbours_effect = np.where(
        net_input > 0,
        net_input * (M - letter_nodes),
        net_input * (letter_nodes - m)
    )
    
    return feature_neighbours_effect, letter_neighbours_effect


# Letter-to-letter weights were set to zero so this one is easy.

letter_to_letter_weights = np.zeros((letter_nodes.size, letter_nodes.size))


# Each feature excites all the letter that contain it and inhibits all the others.

letter_to_feature_excitatory = 0.005
letter_to_feature_inhibitory = 0.15


# Let's first build a binary array from features to letters in one position:

is_excitatory = np.array([features_binary[letter] for letter in sorted(features_binary.keys())]).T

is_excitatory


feature_to_letter_weights_1 = np.where(
    is_excitatory,
    letter_to_feature_excitatory,
    - letter_to_feature_inhibitory
)

feature_to_letter_weights_1


# Now we just have to duplicate this array for each position. Since the weights are zeros across the four positions, we need a block-diagonal matrix made of four `feature_to_letter_weights_1`s along the diagonal.

feature_to_letter_weights = block_diag(*[feature_to_letter_weights_1 for _ in range(4)])

feature_to_letter_weights.shape


# Each non-grey rectangle corresponds to weights from 14 features in a position to the 26 letters in the same position.
# All the other weights are zero.
# NB: Excitatory weights are close to 0, so they are grey as well.

fig = plt.figure()
axes = fig.gca()
axes.matshow(feature_to_letter_weights, cmap='Set1')


alphabet = sorted(features_binary.keys())


# Run for a bit and save letter activations.

def present_word_and_run(word, cycle_count):
    global letter_nodes
    present_word(word)
    # reset letter nodes
    letter_nodes = np.ones((position_count, letter_count)) * r_letter

    # "1 + " to save the initial state
    history_shape = [1 + cycle_count] + list(letter_nodes.shape)
    letters_history = np.empty(history_shape)

    letters_history[0] = letter_nodes
    for t in range(cycle_count):
        run_cycle()
        letters_history[t + 1] = letter_nodes
        
    return letters_history


cycle_count = 20
letters_history = present_word_and_run(word='WORK', cycle_count=cycle_count)


# Plot

def setup_letter_plot(test=False):
    fig = plt.figure(figsize=(6, 6))
    axes = fig.gca()
    
    axes.set_xlim((0, 4))
    axes.set_ylim((0, 26))
    axes.invert_yaxis()
    
    text_objects = [[axes.text(y=(i + 0.8), x=(pos + 0.2), s='A' if test else '', size=10)
                     for i  in range(len(alphabet))]
                    for pos in range(4)]
    
    return fig, axes, text_objects


setup_letter_plot(test=True);


def update_letter_plot(text_objects, t, axes=None, title=None):
    if not axes:
        fig = plt.figure(figsize=(4, 10))
        axes = fig.gca()

    axes.set_title(title)
    
    for pos, (letter_nodes_, text_objects_) in enumerate(zip(letters_history[t], text_objects)):
        for i, (letter, activation, text_object) in enumerate(zip(alphabet, letter_nodes_, text_objects_)):        
            text_object.set_text(letter)
            color = tuple([min(1 - activation, 1)] * 3)
            text_object.set_color(color)
            
    return (text_object for row in text_objects for text_object in row)


get_ipython().run_cell_magic('capture', '', "fig, axes, text_objects = setup_letter_plot();\n\nline_ani = anim.FuncAnimation(fig, \n                              lambda t: update_letter_plot(text_objects, t, axes=axes, title=f'\\nafter {t + 1} cycles:'), \n                              frames=range(0, cycle_count + 1), blit=True, repeat=True)\nrc('animation', html='jshtml')")


line_ani


# "O" and "Q" are equally activated becase all of the "O" features are present in "Q" as well.
# "Q" is not inhibited because only features that are not present in "Q" could do that.
# Let's check that "WQRK" would not activate "O" because "Q" does have a feature absent in "O".

cycle_count = 20
letters_history = present_word_and_run(word='WQRK', cycle_count=cycle_count)


get_ipython().run_cell_magic('capture', '', "fig, axes, text_objects = setup_letter_plot();\n\nline_ani = anim.FuncAnimation(fig, \n                              lambda t: update_letter_plot(text_objects, t, axes=axes, title=f'\\nafter {t + 1} cycles:'), \n                              frames=range(0, cycle_count + 1), blit=True, repeat=True)\nrc('animation', html='jshtml')")


line_ani


# "O" was inhibited by the tail of "Q".
