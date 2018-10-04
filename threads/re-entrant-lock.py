import threading

lock = threading.Lock()

print('First try: ', lock.acquire())
print('Second try: ', lock.acquire(0))


rlock = threading.RLock()

print('First try: ', rlock.acquire())
print('Second try: ', rlock.acquire(0))
