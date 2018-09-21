#! /usr/bin/env bash
icpc -mkl -g -fPIC -shared spmm.c -o libspmm.so
