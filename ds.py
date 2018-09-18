
# list
integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_list = [integer_list, heterogeneous_list, []]

list_length = len(integer_list)  # equals 3
list_sum = sum(integer_list)  # equals 6

x = range(10)  # is the list [0, 1, ..., 9]
print(x)
# In python3 range is a generator object - it does not return a list
x = list(range(10))
zero = x[0]  # equals 0, lists are 0-indexed
one = x[1]  # equals 1
nine = x[-1]  # equals 9, 'Pythonic' for last element
eight = x[-2]  # equals 8, 'Pythonic' for next-to-last element
x[0] = -1  # now x is [-1, 1, 2, 3, ..., 9]

# slice
first_three = x[:3]  # [-1, 1, 2]
three_to_end = x[3:]  # [3, 4, ..., 9]
one_to_four = x[1:5]  # [1, 2, 3, 4]
last_three = x[-3:]  # [7, 8, 9]
without_first_and_last = x[1:-1]  # [1, 2, ..., 8]
copy_of_x = x[:]  # [-1, 1, 2, ..., 9]

# check list membership
1 in [1, 2, 3]  # True
0 in [1, 2, 3]  # False

# list concatenation
x = [1, 2, 3]
x.extend([4, 5, 6])  # x is now [1,2,3,4,5,6]

x = [1, 2, 3]
y = x + [4, 5, 6]  # y is [1, 2, 3, 4, 5, 6]; x is unchanged

x = [1, 2, 3]
x.append(0)  # x is now [1, 2, 3, 0]
y = x[-1]  # equals 0
z = len(x)  # equals 4

# unpack the list
x, y = [1, 2]  # now x is 1, y is 2

_, y = [1, 2]  # now y == 2, didn't care about the first element, throw away


# tuples
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3  # my_list is now [1, 3]

try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")


def sum_and_product(x, y):
    return (x + y),(x * y)


sp = sum_and_product(2, 3)  # equals (5, 6)
s, p = sum_and_product(5, 10)  # s is 15, p is 50

# multiple assignment
x, y = 1, 2  # now x is 1, y is 2
x, y = y, x  # Pythonic way to swap variables; now x is 2, y is 1


# dictionary
empty_dict = {}  # Pythonic
empty_dict2 = dict()  # less Pythonic
grades = {"Joel": 80, "Tim": 95}  # dictionary literal

joels_grade = grades["Joel"]  # equals 80

try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate!")

joel_has_grade = "Joel" in grades  # True
kate_has_grade = "Kate" in grades  # False

# get method that returns a default value
joels_grade = grades.get("Joel", 0)  # equals 80
kates_grade = grades.get("Kate", 0)  # equals 0
no_ones_grade = grades.get("No One")  # default default is None

grades["Tim"] = 99  # replaces the old value
grades["Kate"] = 100  # adds a third entry
num_students = len(grades)  # equals 3

tweet = {
"user": "joelgrus",
"text": "Data Science is Awesome",
"retweet_count": 100,
"hashtags": ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}

tweet_keys = tweet.keys()  # list of keys
tweet_values = tweet.values()  # list of values
tweet_items = tweet.items()  # list of (key, value) tuples

"user" in tweet_keys  # True, but uses a slow list in
"user" in tweet  # more Pythonic, uses faster dict in
"joelgrus" in tweet_values  # True

# word count
word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1

word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1


# A defaultdict is like a regular dictionary,
# except that when you try to look up a key it
# doesn’t contain, it first adds a value for
# it using a zero-argument function you provided
# when you created it.

from collections import defaultdict

word_counts = defaultdict(int)  # int() produces 0
for word in document:
    word_counts[word] += 1

dd_list = defaultdict(list) # list() produces an empty list
dd_list[2].append(1) # now dd_list contains {2: [1]}
dd_dict = defaultdict(dict) # dict() produces an empty dict
dd_dict["Joel"]["City"] = "Seattle" # { "Joel" : { "City" : Seattle"}}
dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1 # now dd_pair contains {2: [0,1]}

# A Counter turns a sequence of values into a
# defaultdict(int)-like object mapping keys
# to counts
from collections import Counter
c = Counter([0, 1, 2, 0])  # c is (basically) { 0 : 2, 1 : 1, 2 : 1 }

word_counts = Counter(document)

# print the 10 most common words and their counts
for word, count in word_counts.most_common(10):
    print(word, count)

# set
s = set()
s.add(1)  # s is now { 1 }
s.add(2)  # s is now { 1, 2 }
s.add(2)  # s is still { 1, 2 }
x = len(s)  # equals 2
y = 2 in s  # equals True
z = 3 in s  # equals False

stopwords_list = ["a","an","at"] + hundreds_of_other_words + ["yet", "you"]
"zip" in stopwords_list  # False, but have to check every element

stopwords_set = set(stopwords_list)
"zip" in stopwords_set  # very fast to check

item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)  # 6
item_set = set(item_list)  # {1, 2, 3}
num_distinct_items = len(item_set)  # 3
distinct_item_list = list(item_set)  # [1, 2, 3]
