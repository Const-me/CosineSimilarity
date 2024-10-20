#pragma once
#include <assert.h>
#include "../Utils/alignedVector.hpp"

float cosineSimScalar( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 );
float cosineSimSimdNaive( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 );
float cosineSimSimdUnrolled( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 );

float cosineSimSimdParallel( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 );
int getParallelThreadsCount();