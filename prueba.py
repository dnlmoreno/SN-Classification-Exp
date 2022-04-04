import multiprocessing as mp
import time
import math

def make_calculation_one(numbers, return_dict):
    results_a = []
    for number in numbers:
        results_a.append(math.sqrt(number ** 3))
    return_dict[0] = results_a

    
def make_calculation_two(numbers, return_dict):
    results_b = []
    for number in numbers:
        results_b.append(math.sqrt(number ** 4))
    return_dict[1] = results_b

def make_calculation_three(numbers, return_dict):
    results_c = []
    for number in numbers:
        results_c.append(math.sqrt(number ** 5))
    return_dict[2] = results_c

import copy

if __name__ == '__main__':
    manager = mp.Manager()
    return_dict = manager.dict()

    #process_manager = mp.Manager()
    #shared_queue = process_manager.Queue()

    number_list = list(range(10))

    p1 = mp.Process(target=make_calculation_one, args=(number_list,return_dict,))
    p2 = mp.Process(target=make_calculation_two, args=(number_list,return_dict,))
    p3 = mp.Process(target=make_calculation_three, args=(number_list,return_dict,))

    start = time.time()
    p1.start()
    p2.start()
    p3.start()
    end = time.time()

    print(f"Tiempo process: {end-start}")

    p1.join()
    p2.join()
    p3.join()

    print(return_dict)
    print(return_dict.values())

    temp_1 = return_dict.copy()

    start = time.time()
    make_calculation_one(number_list, return_dict)
    make_calculation_two(number_list, return_dict)
    make_calculation_three(number_list, return_dict)
    end = time.time()

    print(f"Tiempo normal: {end-start}\n")

    print("Mismo resultado?")
    print(temp_1[0] == return_dict[0])
    print(temp_1[1] == return_dict[1])
    print(temp_1[2] == return_dict[2])