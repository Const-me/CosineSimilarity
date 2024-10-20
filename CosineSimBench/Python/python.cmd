#echo off
cls
del python-log.tsv
set PATH=C:\Program Files\Python313;%PATH%
set BENCH=python.exe cosineSimdBench.py
for %%S in (128k, 256k, 512k, 1M, 2M, 4M, 8M, 16M, 32M) do %BENCH% %%S
