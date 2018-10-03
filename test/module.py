from __future__ import division

print(5/2)
print(5//2)  # integer division
match = 10

print(match)


def my_print(message="my default message"):
    print(message)


my_print("hello")
my_print()

tab_string = "\t"  # represents the tab character
print(len(tab_string))

not_tab_string = r"\t"  # raw strings, represents the characters '\' and 't'
print(len(not_tab_string))

multi_line_string = """This is the first line.
and this is the second line
and this is the third line"""

try:
    print(0/0)
except ZeroDivisionError:
    print("cannot divide by zero")

