import multiprocessing as mp
import random
import string

random.seed(123)

# define an output queue
output = mp.Queue()


# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits)
                       for i in range(length))
    output.put(rand_str)


# # define a example function
# def rand_string(length, pos, output):
#     """ Generates a random string of numbers, lower- and uppercase chars. """
#     rand_str = ''.join(random.choice(
#                         string.ascii_lowercase
#                         + string.ascii_uppercase
#                         + string.digits)
#                    for i in range(length))
#     output.put((pos, rand_str))


# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(5, output)) for x in range(4)]

# # Setup a list of processes that we want to run
# processes = [mp.Process(target=rand_string, args=(5, x, output)) for x in range(4)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
# the order in which the processes finished determines the order of our results
results = [output.get() for p in processes]

# to retrieve the results in order
# results.sort()
# results = [r[1] for r in results]

print(results)
