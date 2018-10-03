import random
import re
from functools import partial

four_uniform_randoms = [random.random() for _ in range(4)]
# [0.8444218515250481, # random.random() produces numbers
# 0.7579544029403025, # uniformly between 0 and 1
# 0.420571580830845, # it's the random function we'll use
# 0.25891675029296335] # most often

print(four_uniform_randoms)

random.seed(10)  # set the seed to 10
print(random.random())  # 0.57140259469
random.seed(10)  # reset the seed to 10
print(random.random())  # 0.57140259469 again

random.randrange(10)  # choose randomly from range(10) = [0, 1, ..., 9]
random.randrange(3, 6)  # choose randomly from range(3, 6) = [3, 4, 5]

up_to_ten = list(range(10))
random.shuffle(up_to_ten)
print(up_to_ten)  # [2, 5, 1, 9, 7, 3, 8, 6, 4, 0]

my_best_friend = random.choice(["Alice", "Bob", "Charlie"])  # "Bob" for me

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6) # [16, 36, 10, 6, 25, 9]  # without duplication

four_with_replacement = [random.choice(range(10)) for _ in range(4)] # [9, 4, 4, 2], allow duplication

# regular expression
print(all([ # all of these are true, because
not re.match("a", "cat"), # * 'cat' doesn't start with 'a'
re.search("a", "cat"), # * 'cat' has an 'a' in it
not re.search("c", "dog"), # * 'dog' doesn't have a 'c' in it
3 == len(re.split("[ab]", "carbs")), # * split on a or b to ['c','r','s']
"R-D-" == re.sub("[0-9]", "-", "R2D2") # * replace digits with dashes
])) # prints True


# OOP
# by convention, we give classes PascalCase names
class Set:

    # these are the member functions
    # every one takes a first parameter "self" (another convention)
    # that refers to the particular Set object being used

    def __init__(self, values=None):
        """This is the constructor.
        It gets called when you create a new Set.
        You would use it like
        s1 = Set() # empty set
        s2 = Set([1,2,2,3]) # initialize with values"""

        self.dict = {}  # each instance of Set has its own dict property
                        # which is what we'll use to track memberships

        if values is not None:
            for value in values:
                self.add(value)

    def __repr__(self):
        """this is the string representation of a Set object
        if you type it at the Python prompt or pass it to str()"""
        return "Set: " + str(self.dict.keys())

    # we'll represent membership by being a key in self.dict with value True
    def add(self, value):
        self.dict[value] = True

    # value is in the Set if it's a key in the dictionary
    def contains(self, value):
        return value in self.dict

    def remove(self, value):
        del self.dict[value]


s = Set([1,2,3])
s.add(4)
print(s.contains(4))  # True
s.remove(3)
print(s.contains(3))  # False


def exp(base, power):
    return base ** power


def two_to_the(power):
    return exp(2, power)


two_to_the = partial(exp, 2) # is now a function of one variable
print(two_to_the(3)) # 8

square_of = partial(exp, power=2)
print(square_of(3)) # 9

def double(x):
    return 2 * x


xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs]  # [2, 4, 6, 8]
twice_xs = map(double, xs)  # same as above
list_doubler = partial(map, double)  # *function* that doubles a list
twice_xs = list_doubler(xs)  # again [2, 4, 6, 8]


def multiply(x, y): return x * y


products = map(multiply, [1, 2], [4, 5])  # [1 * 4, 2 * 5] = [4, 10]


def is_even(x):
    """True if x is even, False if x is odd"""
    return x % 2 == 0


x_evens = [x for x in xs if is_even(x)] # [2, 4]
x_evens = filter(is_even, xs) # same as above
list_evener = partial(filter, is_even) # *function* that filters a list
x_evens = list_evener(xs) # again [2, 4]

# reduce combines the first two elements of
# a list, then that result with the third, that
# result with the fourth, and so on, producing
# a single result:
x_product = reduce(multiply, xs) # = 1 * 2 * 3 * 4 = 24
list_product = partial(reduce, multiply) # *function* that reduces a list
x_product = list_product(xs) # again = 24

# enumerate
for i, document in enumerate(documents):
    do_something(i, document)

for i, _ in enumerate(documents): # just want index
    do_something(i) # Pythonic


# zip
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
# If the lists are different lengths, zip stops as soon as the first list ends.
zip(list1, list2) # is [('a', 1), ('b', 2), ('c', 3)]

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs) # unzip, argument unpacking


def add(a, b): return a + b


add(1, 2) # returns 3
add([1, 2]) # TypeError!
add(*[1, 2]) # returns 3


def doubler(f):
    def g(x):
        return 2 * f(x)
    return g

def f1(x):
    return x + 1


g = doubler(f1)
print(g(3)) # 8 (== ( 3 + 1) * 2)
print(g(-1)) # 0 (== (-1 + 1) * 2)


def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)


magic(1, 2, key="word", key2="word2")
# prints
# unnamed args: (1, 2)
# keyword args: {'key2': 'word2', 'key': 'word'}


def other_way_magic(x, y, z):
    return x + y + z


x_y_list = [1, 2]
z_dict = { "z" : 3 }
print(other_way_magic(*x_y_list, **z_dict)) # 6


def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        """whatever arguments g is supplied, pass them through to f"""
        return 2 * f(*args, **kwargs)
    return g


def f2(x, y):
    return x + y

g = doubler_correct(f2)
print(g(1, 2)) # 6
