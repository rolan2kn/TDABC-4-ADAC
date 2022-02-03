#!/usr/bin/python
# -*- coding: utf-8 -*-

# python imports
import os
import logging
import logging.handlers
import time
from multiprocessing import Process
import sys
from numbers import Number
from collections import Set, Mapping, deque

zero_depth_bases = (str, bytes, Number, range, bytearray)
iteritems = 'items'

logger = logging.getLogger(__file__)


def get_home_dir():
    """Return the user home directory"""
    return os.path.expanduser('~')


def get_app_home(app_name):
    """Return the application home directory. This will be a directory
    in $HOME/.app_name/ but with app_name lower cased.
    """
    return os.path.join(get_home_dir(), '.' + app_name.lower())

def get_pycharm_path():
    """
    compute the path of all python projects PATHS/projects/pycharm
    :return:
    """
    m_path = get_module_path()
    pycharm_path = m_path

    if len(m_path) > 0:
        pycharm_path = "/".join(m_path.split("/")[:-1])

    return pycharm_path

def get_pycharm_project_path(pchrm_project):
    """
    compute the path of a pycharm project
    :return:
    """

    return "{0}/{1}".format(get_pycharm_path(), pchrm_project)

def get_remote_module_handler(project_name, filename, module_name):
    import importlib.machinery

    sphere_path = "{0}/{1}".format(get_pycharm_project_path(project_name),
                                   filename)
    loader = importlib.machinery.SourceFileLoader(module_name, sphere_path)

    return loader.load_module(module_name)

def get_root_dir():
    """Return the root directory of the application"""

    # was run from an executable?

    root_d = os.path.dirname(sys.argv[0])

    if not root_d:
        try:
            path = os.path.abspath(__file__)
            name = __name__.replace(".", "/")
            pos = path.find(name)
            if pos != -1:
                root_d = path[:pos]
        except:
            path = os.path.dirname(__file__)
            root_d = '/'.join(path.split('/')[:-1])

    return root_d


def get_module_path():
    '''Get the path to the current module no matter how it's run.'''

    # if '__file__' in globals():
    #     If run from py
    #    # return os.path.dirname(__file__)

    # If run from command line or an executable

    return get_root_dir()


def get_module_pkg():
    """Return the module's package path.
    Ej:
        if current module is the the call to:
            get_module_pkg()
        should return:
    """

    return '.'.join(__name__.split('.')[:-1])

def get_all_filenames(root_path=None, file_pattern = None):
    root_path = get_module_path() if root_path is None else root_path
    files_list = []

    for path, _, files in os.walk(root_path):
        for _file in files:
            str_to_print = "{0}/{1}".format(path, _file)

            if file_pattern is None or len(file_pattern) == 0:
                files_list.append(str_to_print)
            elif str_to_print.find(file_pattern) != -1:
                files_list.append(str_to_print)

    return files_list

"""
Busqueda exponencial

int exponential_search(T arr[], int size, T key)
{
    if (size == 0) {
        return NOT_FOUND;
    }

    int bound = 1;
    while (bound < size && arr[bound] < key) {
        bound *= 2;
    }

    return binary_search(arr, key, bound/2, min(bound, size));
}
"""
def exponential_searching(elem, _list):
    try:
        size = len(_list)
        if size == 0:
            return -1

        if _list[0] == elem:
            return 0

        i = 1
        while i < size and _list[i] < elem:
            i *= 2

        return binary_search(elem, _list, i//2, min(i, size), size)
    except Exception as e:
        print ("ERROR EN UTILS::exponential_searching(.): ", e)
        return -1

def binary_search(elem, _list, i, f, size):
    '''
    Asignar 0 a L y a R (n − 1).
    Si L > R, la búsqueda termina sin encontrar el valor.
    Sea m (la posición del elemento del medio) igual a la parte entera de (L + R) / 2.
    Si Am < T, igualar L a m + 1 e ir al paso 2.
    Si Am > T, igualar R a m – 1 e ir al paso 2.
    Si Am = T, la búsqueda terminó, retornar m.

    :param elem:
    :param _list:
    :return:
    '''
    l = i
    r = f

    while l <= r:
        m = (l+r)//2

        if m >= size:
            break

        if _list[m] < elem:
            l = m+1
        elif _list[m] > elem:
            r = m-1
        else:
            return m

    return -1


def compare(elemA, elemB, size):
    i = 0
    while i < size and elemA[i] == elemB[i]:
        i += 1

    if i == size: # we stop in the middle of both lists
        return 0

    return -1 if elemA[i] < elemB[i] else 1


def list_exponential_searching(elem, _list):
    try:
        size = len(_list)
        if size == 0:
            return -1

        if _list[0] == elem:
            return 0

        i = 1
        size_e = len(elem)
        result = compare(_list[i], elem, size_e)
        while i < size and result < 0:
            i *= 2
            if i < size:
                result = compare(_list[i], elem, size_e)

        return list_binary_search(elem, _list, i//2, min(i, size), size)
    except Exception as e:
        print ("ERROR EN UTILS::list_exponential_searching(.): ", e)
        return -1

def list_binary_search(elem, _list, i, f, size):
    '''
    Asignar 0 a L y a R (n − 1).
    Si L > R, la búsqueda termina sin encontrar el valor.
    Sea m (la posición del elemento del medio) igual a la parte entera de (L + R) / 2.
    Si Am < T, igualar L a m + 1 e ir al paso 2.
    Si Am > T, igualar R a m – 1 e ir al paso 2.
    Si Am = T, la búsqueda terminó, retornar m.

    :param elem:
    :param _list:
    :return:
    '''
    l = i
    r = f
    size_e = len(elem)
    while l <= r:
        m = (l+r)//2

        if m >= size:
            break

        value = compare(_list[m], elem, size_e)
        if value < 0:
            l = m+1
        elif value > 0:
            r = m-1
        else:
            return m

    return -1

def dict_binary_search(elem, _dict):
    '''
    Asignar 0 a L y a R (n − 1).
    Si L > R, la búsqueda termina sin encontrar el valor.
    Sea m (la posición del elemento del medio) igual a la parte entera de (L + R) / 2.
    Si Am < T, igualar L a m + 1 e ir al paso 2.
    Si Am > T, igualar R a m – 1 e ir al paso 2.
    Si Am = T, la búsqueda terminó, retornar m.

    :param elem:
    :param _list:
    :return:
    '''
    l = 0
    r = len(_dict)
    size = len(_dict)

    while l <= r:
        m = (l+r)//2

        if m >= size:
            break

        if _dict[m] < elem:
            l = m+1
        elif _dict[m] > elem:
            r = m-1
        else:
            return m

    return -1

class ExecutionTime(object):
    """
    Helper that can be used in with statements to have a simple
    measure of the timming of a particular block of code, e.g.
    with ExecutionTime("db flush"):
        db.flush()
    """
    def __init__(self, info="", with_traceback=False):
        self.info = info
        self.with_traceback = with_traceback

    def __enter__(self):
        self.now = time.time()

    def __exit__(self, type, value, stack):
        logger = logging.getLogger(__file__)
        msg = '%s: %s' % (self.info, time.time() - self.now)
        if logger.handlers:
            logger.debug(msg)
        else:
            print(msg)
        if self.with_traceback:
            import traceback
            msg = traceback.format_exc()
            if logger.handlers:
                logger.error(msg)
            print (msg)


def get_obj_size(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size

    return inner(obj_0)
    # return get_obj_size2(obj_0)


# En multiprocesamiento, los procesos se generan creando un objeto Process

# y luego llamando a su método start()

# Process utiliza la API de threading.Thread


def exec_with_timeout(func, args, time):
    """

    Ejecuta una función con un limite de tiempo

    Tiene que recibir:

        func: el nombre de la función a ejecutar

        args: una tupla con los argumentos a pasar a la función

    Devuelve True si ha finalizado la función correctamente


    https://docs.python.org/2/library/multiprocessing.html

    """

    p = Process(target=func, args=args)

    p.start()

    p.join(time)

    if p.is_alive():
        p.terminate()

        print("Ha finalizado por timeout")

        return False

    print("Se ha ejecutado correctamente")

    return True

def sort_parallel_lists(X, Y):
    sorted_by_Xs = sorted(
        list(zip(X, Y)))
    Xs = [X for X, Y in sorted_by_Xs]
    Ys = [Y for X, Y in sorted_by_Xs]

    return Xs, Ys

def is_float_str(value):
    pos = value.find(".")
    if pos == -1:
        return value.isnumeric()

    if value[:pos].isnumeric() and value[pos+1:].isnumeric():
        return True

    return False