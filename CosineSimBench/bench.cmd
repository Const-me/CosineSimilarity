set BENCH=x64\Release\CosineSimBench.exe
for %%A in (Scalar, Naive, Unrolled, Parallel) do for %%S in (128k, 256k, 512k, 1M, 2M, 4M, 8M, 16M, 32M) do %BENCH% %%A %%S