import multiprocessing as mp
import time
import math

import preprocess

def function(key_types, path_data, path_save, bands_name, num_augments, normalize, reproduce_paper):
    preprocess.CharnockPreprocess(key_types, path_data, path_save, bands_name, num_augments, normalize, reproduce_paper)

if __name__ == '__main__':
    # Creación de los datos del ZTF para todos los puntos de la banda g y r
    key_types = {'SNIa': 0, 'nonSNIa': 1}
    #path_data = 'data/ztf/lc_from_first_mjd_forced_fot.pkl'
    path_data = 'ztf_toy.pkl'
    path_save = 'data'
    bands_name = ['g', 'r']
    num_augments = 5
    normalize = False
    reproduce_paper = False

    manager = mp.Manager()
    return_dict = manager.dict()

    p1 = mp.Process(target=function, args=(key_types, path_data, path_save, bands_name, num_augments, normalize, reproduce_paper,))
    p2 = mp.Process(target=function, args=(key_types, path_data, path_save, bands_name, num_augments, normalize, reproduce_paper,))
    p3 = mp.Process(target=function, args=(key_types, path_data, path_save, bands_name, num_augments, normalize, reproduce_paper,))

    start = time.time()
    p1.start()
    p2.start()
    p3.start()
    end = time.time()

    print(f"Tiempo process: {end-start}")

    p1.join()
    p2.join()
    p3.join()


#from multiprocessing import Process
#
#procesos = []
#
#for n in range(4):
#    proceso = Process(target=function, args=(tupla,))
#    procesos.append(proceso)
#
#for proceso in procesos:
#    proceso.start()
#
#print("------- Espera")
#for proceso in procesos:
#    proceso.join()




#import concurrent.futures
#import time
#
#shared_dictionary = multiprocessing.Manager().dict()
#
#if __name__ == '__main__':
#
#    s = time.perf_counter()
#    with concurrent.futures.ProcessPoolExecutor() as executor:
#        result = executor.map(
#            function,
#            tupla,
#        )
#
#    e = time.perf_counter()
#    print(f"parallel took {e-s} sec to run ")
#    print(shared_dictionary)





#from multiprocessing import Pool
#from datetime import datetime
#
#if __name__ == '__main__':
#    start = datetime.now()
#    with Pool() as pool:
#        results = pool.map(function, tupla)
#    end = datetime.now()
#
#    print(f'Tiempo: {end-start} segundos')





#from multiprocessing import Pool
#
#if __name__ == '__main__':
#    p = Pool(processes=4)
#    result = p.map(function, tupla)
#    p.close()
#    p.join()
#
#    print(result)