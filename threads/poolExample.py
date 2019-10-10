import multiprocessing as mp


def cube(x):
    return x ** 3


# The Pool.map and Pool.apply will lock the main program until all processes are finished
# which is quite useful if we want to obtain results in a particular order for certain applications.
pool = mp.Pool(processes=4)
results = [pool.apply(cube, args=(x,)) for x in range(1, 7)]
print(results)

pool = mp.Pool(processes=4)
results = pool.map(cube, range(1,7))
print(results)

# the async variants will submit all processes at once
# and retrieve the results as soon as they are finished
pool = mp.Pool(processes=4)
results = [pool.apply_async(cube, args=(x,)) for x in range(1,7)]
output = [p.get() for p in results]
print(output)