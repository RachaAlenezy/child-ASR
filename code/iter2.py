import itertools
import pprint
import pickle
def generate_groups(lst, n, i):
    print("generating, ", i)
    if not lst:
        yield []
    else:
        for group in (((lst[0],) + xs) for xs in itertools.combinations(lst[1:], n-1)):
            for groups in generate_groups([x for x in lst if x not in group], n, i+1):
                yield [group] + groups


all = ['1', '2', '3', '4']
# all = ['alef', 'ba', 'ta', 'tha', 'jeem', '7a', '5a', 'dal', '4al', 'ra', 'za', 'sen', 'shen', '9ad', 'dhad', '6a']
l = list(generate_groups(all, 2, 0))
#
# with open("combs7.txt", "wb") as fp:   #Pickling
#     pickle.dump(l, fp)


pprint.pprint(l)

print(len(l))
