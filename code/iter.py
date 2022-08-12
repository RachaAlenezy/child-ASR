import itertools
import pprint
import pickle
# def generate_groups(lst, n):
#     if not lst:
#         yield []
#     else:
#         for group in (((lst[0],) + xs) for xs in itertools.combinations(lst[1:], n-1)):
#             for groups in generate_groups([x for x in lst if x not in group], n):
#                 yield [group] + groups
#
#
# # all = ['alef', 'ba', 'ta', 'tha', 'jeem', '7a', '5a', 'dal', '4al', 'ra', 'za', 'sen', 'shen', '9ad', 'dhad', '6a', 'dha', '3en', 'ghen', 'fa', '8af', 'kaf', 'lam', 'meem', 'non', 'ha', 'wa', 'ya']
# all = ['alef', 'ba', 'ta', 'tha', 'jeem', '7a', '5a', 'dal', '4al', 'ra', 'za', 'sen', 'shen', '9ad', 'dhad', '6a']
#
#
# l = list(generate_groups(all, 4))
# #
# # with open("combs7.txt", "wb") as fp:   #Pickling
# #     pickle.dump(l, fp)
# #
#
# pprint.pprint(l)

# print(len(l))


import random

n = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

pprint.pprint()
# [['B', 'H', 'G'], ['D', 'A', 'C'], ['E', 'F', 'I'], ['J', 'K']]
