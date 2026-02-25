import ray
import time

database = ["1", "2", "3", "4", "5", "6", "7", "8"]

def retrive(item):
    time.sleep(item / 10.)
    return item, database[item]

@ray.remote
def retrive_task(item):
    return retrive(item)

if __name__ == "__main__":
    retrived_data = [retrive_task.remote(i) for i in range(len(database))]
    print(f"retrived_data: {ray.get(retrived_data)}")