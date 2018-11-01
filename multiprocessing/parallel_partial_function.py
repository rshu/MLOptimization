from functools import partial
from time import sleep
import multiprocessing as mp

def add(a, b):
    sleep(2)
    print("{} + {} = {}".format(a,b,a+b))
    return a + b

if __name__ == "__main__":
    num_cpu = mp.cpu_count()
    args = list(range(4 * num_cpu))
    print(args, "\n")

    add_5 = partial(add, b=5)

    p = mp.Pool(num_cpu)
    res = p.map_async(add_5, args, chunksize=4)
    res.get()
