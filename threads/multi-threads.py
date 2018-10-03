import threading
import time
import logging

# https://pymotw.com/2/threading/index.html#module-threading
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] (%(threadName)-10s) %(message)s', )


def worker(num):
    """thread worker function"""
    print('Worker: %s' % num)
    return


def smart_worker():
    # print(threading.currentThread().getName(), 'Starting')
    # time.sleep(1)
    # print('')
    # print(threading.currentThread().getName(), 'Exiting')
    logging.debug('Starting')
    time.sleep(1)
    logging.debug('Exiting')


def my_service():
    # print(threading.currentThread().getName(), 'Starting')
    # time.sleep(3)
    # print('')
    # print(threading.currentThread().getName(), 'Exiting')
    logging.debug('Starting')
    time.sleep(3)
    logging.debug('Exiting')


threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

t = threading.Thread(name='my_service', target=my_service)
w = threading.Thread(name='smart_worker', target=smart_worker)
w2 = threading.Thread(target=smart_worker)  # use default thread name

w.start()
w2.start()
t.start()
