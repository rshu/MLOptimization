import random

with open('/usr/share/dict/words', 'rt') as f:
    words = f.readlines()

# If no argument is passed, it removes trailing spaces.
words = [w.rstrip() for w in words]

for w in random.sample(words, 5):
    print(w)
