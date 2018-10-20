import multiprocessing as mp

num_cpu = mp.cpu_count()

print("Number of cpu cores: {}".format(num_cpu))
