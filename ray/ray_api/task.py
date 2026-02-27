import ray
import time

database = ["1", "2", "3", "4", "5", "6", "7", "8"]

def retrive(item):
    time.sleep(item / 10.)
    return item, database[item]

def print_func_runtime(data, start_time):
    print(f"total time: {time.time() - start_time:.2f}")
    print(*data, sep="\n")

@ray.remote
def retrive_task(item):
    return retrive(item)

if __name__ == "__main__":
    start_time = time.time()
    retrived_data = [retrive_task.remote(i) for i in range(len(database))]
    print_func_runtime(ray.get(retrived_data), start_time)