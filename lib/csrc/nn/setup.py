#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : setup.py
# Author            : WangZi
# Date              : 14.04.2020
# Last Modified Date: 14.04.2020
# Last Modified By  : WangZi
import os

cuda_include='/usr/local/cuda-9.0/include'
os.system('/usr/local/cuda-9.0/bin/nvcc src/nearest_neighborhood.cu -c -o src/nearest_neighborhood.cu.o -x cu -Xcompiler -fPIC -O2 -arch=sm_52 -I {}'.format(cuda_include))

from cffi import FFI
ffibuilder = FFI()


with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source(
    "_ext",
    """
    #include "src/ext.h"
    """,
    extra_objects=['src/nearest_neighborhood.cu.o',
                   '/usr/local/cuda-9.0/lib64/libcudart.so'],
    libraries=['stdc++']
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("rm src/*.o")
    os.system("rm *.o")
