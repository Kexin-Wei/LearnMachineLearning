# %%
# 1 process
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
# 2 processes
import multiprocessing
start = time.perf_counter()

def do_something(s):
    print(f"Slee {s} second(s)...")
    time.sleep(s)
    print("Done")
    
p1 = multiprocessing.Process(target=do_something,args=[1.5])
p2 = multiprocessing.Process(target=do_something,args=[1.5])

p1.start()
p2.start()

p1.join()
p2.join()

finish = time.perf_counter()

print(f"Finished all work in {round(finish-start,3)} s")
# %%
# 10 processes
import multiprocessing
start = time.perf_counter()

def do_something(s):
    print(f"Slee {s} second(s)...")
    time.sleep(s)
    print("Done")
    
processes = []
for _ in range(10):
    p = multiprocessing.Process(target=do_something,args=[1.5])
    processes.append(p)
    p.start()

for p in processes:
    p.join()

finish = time.perf_counter()

print(f"Finished all work in {round(finish-start,3)} s")
# %%
import concurrent
import time

start = time.perf_counter()
def return_something(s,name):
    print(f"{name} Sleep {s} second(s)...")
    time.sleep(s)
    return f"{name} Done"


f_list = []
for i  in range(10):
    print(f"Now ep :{i}")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        seconds = [5,4,3,2,1]
        processes = [executor.submit(return_something,s,index) for index,s in enumerate(seconds)]
        
        for f in concurrent.futures.as_completed(processes):
            print(f.result())
            f_list.append(f.result())
    
finish = time.perf_counter()
print(f"Finished all work in {round(finish-start,3)} s")  
# %%
import concurrent

start = time.perf_counter()
def return_something(s,name):
    print(f"{name} Sleep {s} second(s)...")
    time.sleep(s)
    return f"{name} Done"

with concurrent.futures.ProcessPoolExecutor() as executor:
    seconds = [5,4,3,2,1]
    names   = [i for i in range(len(seconds))]
    results = executor.map(return_something,seconds,names)
    
    for result in results:
        print(result)
    
finish = time.perf_counter()
print(f"Finished all work in {round(finish-start,3)} s")  
# %%
