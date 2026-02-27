import ray
import subprocess
from collections.abc import Generator

"""
数据流:

map: 对每个元素施加同一个操作，"一对一变换"——输入一个元素，输出一个（或多个）结果，元素之间互不影响。正因为互不影响，所以天然可以并行。
Map Worker 0 → [桶0, 桶1, 桶2]
Map Worker 1 → [桶0, 桶1, 桶2]    
Map Worker 2 → [桶0, 桶1, 桶2]

reduce: 把一组元素聚合成一个值，"多对一聚合"——把一堆中间结果合并成最终结果。
Reduce Worker 0 ← 所有 map worker 的桶0
Reduce Worker 1 ← 所有 map worker 的桶1
Reduce Worker 2 ← 所有 map worker 的桶2
"""


# map 原子操作
def map_func(doc: str) -> Generator[tuple[str, int], None, None]:
    for word in doc.lower().split():
        yield word, 1


# map + shuffle，分布式并行
@ray.remote
def apply_map_func(
    doc_list: list[str], num_partitions=3
) -> list[list[tuple[str, int]]]:
    worker_map_list = [list() for _ in range(num_partitions)]
    for doc in doc_list:
        for res in map_func(doc):
            hash_key = res[0][0]  # 取首字符
            worker_index = ord(hash_key) % num_partitions
            worker_map_list[worker_index].append(res)
    return worker_map_list


# reduce
@ray.remote
def apply_reduce_func(*map_res: list[tuple[str, int]]) -> dict[str, int]:
    reduce_dict: dict[str, int] = dict()
    for one_partition in map_res:
        for key, value in one_partition:
            if key not in reduce_dict:
                reduce_dict[key] = 0
            reduce_dict[key] += value
    return reduce_dict


if __name__ == "__main__":
    zen_article = subprocess.check_output(["python", "-c", "import this"]).decode(
        "utf-8"
    )
    zen_article_split_list = zen_article.split()

    # 分块
    partition_num = 3
    zen_article_split_list_length = len(zen_article_split_list)
    chunk_list = [
        zen_article_split_list[
            (i * zen_article_split_list_length // partition_num) : (i + 1)
            * zen_article_split_list_length
            // partition_num
        ]
        for i in range(partition_num)
    ]

    # map
    map_res = [
        apply_map_func.options(num_returns=partition_num).remote(
            chunk_list[i], partition_num
        )
        for i in range(partition_num)
    ]
    print("---map---")
    print(f"map_res: {map_res}\n")
    print(f"ray.get(map_res[0]: {ray.get(map_res[0])}\n")  # 每个 worker 的返回
    print(f"ray.get(map_res[0][0]: {ray.get(map_res[0][0])}\n")
    print("---map end---")

    # reduce
    output = []
    for i in range(partition_num):
        output.append(
            apply_reduce_func.remote(*[partition[i] for partition in map_res])
        )

    # 汇总结果
    results = ray.get(output)
    final = {}
    for d in results:
        final.update(d)
    print(final)
