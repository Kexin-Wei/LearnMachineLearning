import concurrent.futures
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