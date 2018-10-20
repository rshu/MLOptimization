from functools import partial

def add(a, b):
    return a + b

add_2 = partial(add, b=2)

print(add_2(3))
