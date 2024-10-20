cmake_minimum_required(VERSION 3.10)
project(cosineSimdBench)

# Set C++ standard
set( CMAKE_CXX_STANDARD 17 )
# Optimized release build
set( CMAKE_BUILD_TYPE Release )
# Enable LTCG
set( CMAKE_INTERPROCEDURAL_OPTIMIZATION ON ) 
# A few other options
SET( CMAKE_CXX_FLAGS  "-O3 -DNDEBUG -march=native -ftree-vectorize")

# OpenMP for simdParallel.cpp
find_package( OpenMP REQUIRED )

# Source files
set( SOURCES
    cosineSimdBench.cpp
    Utils/benchmarkUtils.cpp
    Utils/simdUtils.cpp
    Tests/scalar.cpp
    Tests/simdNaive.cpp
    Tests/simdParallel.cpp
    Tests/simdUnrolled.cpp
)

# Add executable
add_executable(cosineSimdBench ${SOURCES})

# Link OpenMP for simdParallel.cpp
target_link_libraries(cosineSimdBench PRIVATE OpenMP::OpenMP_CXX)