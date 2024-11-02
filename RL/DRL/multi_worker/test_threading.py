# %%
# no thread
import time
start = time.perf_counter()

def do_something(s):
    print(f"Slee {s} second(s)...")
    time.sleep(s)
    print("Done")
    
do_something(1.5)
do_something(1.5)

finish = time.perf_counter()

print(f"Finished all work in {round(finish-start,3)} s")

# %%
# 2 threads
import threading
start = time.perf_counter()

t1 = threading.Thread(target=do_something,args=[1.5])
t2 = threading.Thread(target=do_something,args=[1.5])
t1.start()
t2.start()

print("Waiting")
print("Am i the main thread?")

t1.join()
t2.join()

finish = time.perf_counter()
print(f"Finished all work in {round(finish-start,3)} s")
# %%
# 10 at 1 time
start = time.perf_counter()
threads = []
for _ in range(10):
    t = threading.Thread(target=do_something,args=[1.5])
    t.start()
    threads.append(t)

for thread in threads:
    thread.join()
finish = time.perf_counter()
print(f"Finished all work in {round(finish-start,3)} s")
# %%
# Execute
import concurrent

start = time.perf_counter()
def return_something(s):
    print(f"Slee {s} second(s)...")
    time.sleep(s)
    return "Done"

with concurrent.futures.ThreadPoolExecutor() as executor:
    f1 = executor.submit(return_something,1)
    print(f1.result())

finish = time.perf_counter()
print(f"Finished all work in {round(finish-start,3)} s")  

# %%
import concurrent

start = time.perf_counter()
def return_something(s,name):
    print(f"{name} Sleep {s} second(s)...")
    time.sleep(s)
    return f"{name} Done"

with concurrent.futures.ThreadPoolExecutor() as executor:
    seconds = [5,4,3,2,1]
    threads = [executor.submit(return_something,s,index) for index,s in enumerate(seconds)]
    
    for f in concurrent.futures.as_completed(threads):
        print(f.result())
    
finish = time.perf_counter()
print(f"Finished all work in {round(finish-start,3)} s")  

# %%

import concurrent

start = time.perf_counter()
def return_something(s,name):
    print(f"{name} Sleep {s} second(s)...")
    time.sleep(s)
    return f"{name} Done"

with concurrent.futures.ThreadPoolExecutor() as executor:
    seconds = [5,4,3,2,1]
    names = [i for i in range(len(seconds))]
    results = executor.map(return_something,seconds,names)
    
    for result in results:
        print(result)
    
finish = time.perf_counter()
print(f"Finished all work in {round(finish-start,3)} s")  
# %%
"""
def thread_delay(thread_name, delay):
    count = 0
    print(thread_name, 'start:', time.time())
    while count < 3:
        time.sleep(delay)
        count += 1
        print(thread_name, 'doning',count, time.time())
    print(thread_name, 'done:', time.time())
# %%
t1 = threading.Thread(target=thread_delay, args=('t1', 1))
t2 = threading.Thread(target=thread_delay, args=('t2', 3))
t1.start()
t2.start()
# %%
t1.join()
t2.join()"""
