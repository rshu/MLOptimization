from scipy.optimize import rosen, differential_evolution
from joblib import Parallel, delayed, cpu_count
import time

bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]


def objfunc(params):
    for it in range(1000000):
        it ** 2
    return rosen(params)


def pareval(listcoords):
    listresults = Parallel(n_jobs=cpu_count())(delayed(objfunc)(i) for i in listcoords)
    return listresults


def parallel_run():
    result = differential_evolution(pareval, bounds, parallel=True, maxiter=10, polish=False)
    print(result.x, result.fun)


def serial_run():
    result = differential_evolution(objfunc, bounds, maxiter=10, polish=False)
    print(result.x, result.fun)


start_time = time.time()
serial_run()
print("Serial run took %s seconds using 1 core " % (time.time() - start_time))

start_time = time.time()
parallel_run()
print("Parallel run took %s seconds using %s cores" % ((time.time() - start_time), cpu_count()))
