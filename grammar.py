
if 1 > 2:
    message = "if only 1 were greater than two…"
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"

x = 0
while x < 10:
    print(x, "is less than 10")
    x += 1

for x in range(10):
    print(x, "is less than 10")

parity = "even" if x % 2 == 0 else "odd"


for x in range(10):
    if x == 3:
        continue # go immediately to the next iteration
    if x == 5:
        break # quit the loop entirely
    print(x)

one_is_less_than_two = 1 < 2  # equals True
true_equals_false = True == False  # equals False

x = None
print(x == None)  # prints True, but is not Pythonic
print(x is None)  # prints True, and is Pythonic

s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = ""

# and returns its second value when the first
# is “truthy,” the first value when it’s not
first_char = s and s[0]

safe_x = x or 0

# all function, which takes a list and returns
# True precisely when every element is truthy
all([True, 1, { 3 }])  # True
all([True, 1, {}])  # False, {} is falsy
any([True, 1, {}])  # True, True is truthy

# any function, which returns True when at least
# one element is truthy
all([])  # True, no falsy elements in the list
any([])  # False, no truthy elements in the list

# sort
x = [4,1,2,3]
y = sorted(x)  # is [1,2,3,4], x is unchanged
x.sort()  # now x is [1,2,3,4]

# sort the list by absolute value from largest to smallest
x = sorted([-4,1,-2,3], key=abs, reverse=True) # is [-4,3,-2,1]

# sort the words and counts from highest count to lowest
# wc = sorted(word_counts.items(), key=lambda(word, count): count, reverse=True)

even_numbers = [x for x in range(5) if x % 2 == 0]  # [0, 2, 4]
squares = [x * x for x in range(5)]  # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers]  # [0, 4, 16]

square_dict = { x : x * x for x in range(5) }  # { 0:0, 1:1, 2:4, 3:9, 4:16 }
square_set = { x * x for x in [1, -1] }  # { 1 }

zeroes = [0 for _ in even_numbers] # has the same length as even_numbers

pairs = [(x, y) for x in range(10) for y in range(10)] # 100 pairs (0,0) (0,1) ... (9,8), (9,9)

increasing_pairs = [(x, y) # only pairs with x < y,
for x in range(10) # range(lo, hi) equals
for y in range(x + 1, 10)] # [lo, lo + 1, ..., hi - 1]


def lazy_range(n):
    """a lazy version of range"""
    i = 0
    while i < n:
        yield i
        i += 1

for i in lazy_range(10):
    do_something_with(i)

lazy_evens_below_20 = (i for i in lazy_range(20) if i % 2 == 0)
