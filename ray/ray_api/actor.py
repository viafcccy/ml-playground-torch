import time
import ray
import task

db_obj_ref = ray.put(task.database)

@ray.remote # remote 修饰类实现 actor
class DataTracker:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1

    def count(self):
        return self.count

@ray.remote
def retrive_data(item, db, tracker):
    time.sleep(item / 10.)
    tracker.increment.remote()
    return item, db[item]

if __name__ == "__main__":
   tracker = DataTracker.remote()
   data_ref = [retrive_data.remote(i, ray.get(db_obj_ref), tracker) for i in range(len(task.database))]
   data = ray.get(data_ref)

   print(data)
   print(ray.get(tracker.count.remote()))