from time import sleep
import multiprocessing as mp
import random
def func(arg):
    sleep(5)
    print("Hello, world! {}".format(arg))


if __name__ == "__main__":
    num_cpu = mp.cpu_count()
    args = list(range(4 * num_cpu)) # 16 processes
    print(args, "\n")
    p = mp.Pool(num_cpu)
    res = p.map_async(func, args, chunksize=4).get()
