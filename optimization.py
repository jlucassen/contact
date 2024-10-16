import numpy as np
import scipy.optimize as opt
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
#test

def life(probabilities, lifetimes, maxes, beta=10):
    max_p = np.max(np.multiply(maxes, probabilities))
    return max_p + (1 - max_p) * (1 + np.dot(lifetimes, probabilities.T))

def make_objective(lifetimes, maxes):
    def objective(probabilities):
        return -life(probabilities, lifetimes, maxes)
    return objective

def optimize_probabilities(lifetimes, maxes):
    objective = make_objective(lifetimes, maxes)
    probability_constraints = [{'type':'eq', 'fun':lambda p : np.sum(p) - 1}]
    probability_bounds = [(0, 1) for _ in range(lifetimes.shape[0])]
    init = np.random.rand(lifetimes.shape[0])
    init = init / np.sum(init)
    result = opt.minimize(objective, init, bounds=probability_bounds, constraints=probability_constraints)
    return result.x, -result.fun

def analyze_vocabulary(path, slice=slice(0, -1)):
    prefix_trie = {} # trie of prefixes to valid words
    valid_words = set() # set of valid words
    with open(path) as word_file:
        for word in word_file.readlines()[slice]:
            position = prefix_trie
            for letter in word.strip():
                if letter not in position:
                    position[letter] ={}
                position = position[letter]
            valid_words.add(word.strip())
            position['$'] = {}
    return prefix_trie

def optimize_conditional_distribution(prefix_trie):
    bfs_queue = deque([(prefix_trie, '')])
    nodes_by_prefix = {} # map from prefix to node in prefix_trie
    while bfs_queue:
        current_node, prefix = bfs_queue.popleft()
        nodes_by_prefix[prefix] = current_node
        bfs_queue.extend((child, prefix+letter) for letter, child in current_node.items() if isinstance(child, dict))
    word_trie_blt = [x for x in sorted(nodes_by_prefix.items(), key=lambda x: len(x[0]), reverse=True)] # (prefix, node) pairs in backward level traversal order

    prefix_data = {} # storing info about each prefix, including conditional probabilities
    for prefix, node in tqdm(word_trie_blt):
        if len(node) == 0:
            continue
        if '$' in node:
            prefix_data[prefix+'$'] = {'lifetime': 1, 'max': 1, 'probs': np.array([1])}
        child_prefixes = sorted([prefix+letter for letter in node.keys()], reverse=True) # this creates alphabetic bias??
        child_lifetimes = np.array([prefix_data[child]['lifetime'] for child in child_prefixes])
        child_maxes = np.array([prefix_data[child]['max'] for child in child_prefixes])
        probs, prefix_lifetime = optimize_probabilities(child_lifetimes, child_maxes)
        # print(child_prefixes, child_lifetimes, child_maxes, probs, prefix_lifetime)
        prefix_data[prefix] = {'lifetime': prefix_lifetime,
                               'max': np.max(np.multiply(child_maxes, probs)),
                               'probs': probs,
                               'children': child_prefixes}
    return prefix_data

def flatten_distribution(prefix_trie, prefix_data):
    probs_by_word = {}
    bfs_queue = deque([(prefix_trie, '', 1)])
    while bfs_queue:
        current_node, prefix, prob_above = bfs_queue.popleft()
        if prefix.endswith('$'):
            probs_by_word[prefix] = prob_above*prefix_data[prefix]['probs'][-1] # 0 when sort is not reversed, -1 when reversed
        bfs_queue.extend((child, prefix+letter, prob_above*prefix_data[prefix]['probs'][i]) for (i, (letter, child)) in enumerate(current_node.items()) if isinstance(child, dict))
    return probs_by_word

prefix_trie = analyze_vocabulary('words_alpha.txt', slice=slice(0, -1))
# print(prefix_trie)
prefix_data = optimize_conditional_distribution(prefix_trie)
# print(prefix_data)
dist = flatten_distribution(prefix_trie, prefix_data)

[print(x) for x in sorted(dist.items(), key=lambda x: -x[1])[:10]]
# [print(x) for x in sorted(dist.items(), key=lambda x: -x[1])[-10:]]
# print(sum(dist.values()))
# print(sum([sum(prefix_data[prefix]['probs'] == 0) for prefix in prefix_data]))

# def test_word_trajectory(prefix_data, word):
#     for i, prefix in list(enumerate([word[:i] for i in range(len(word))]))[:-1]: # skip last one, no continuation
#         print(prefix)
#         transition = word[i]
#         print(transition)
#         trans_child = [p[-1:] for p in prefix_data[prefix]['children']].index(transition)
#         print(prefix_data[prefix]['children'][trans_child])
#         print(prefix_data[prefix]['probs'].shape)
#         print(prefix_data[prefix]['probs'][trans_child])
#         print('---\n\n')

# test_word_trajectory(prefix_data, 'antidisestablishmentarianism')

# plt.hist(retries, bins=100)
# plt.show()



## zeroes
# optimization methods
# retries
# maxiter
# log probs
# averaging
# logsumexp

## Alphabetic bias
# confirm it still exists
# ????