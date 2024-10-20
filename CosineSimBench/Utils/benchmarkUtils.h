#pragma once
#include <stdint.h>
#include <float.h>
#include <immintrin.h>
#include "alignedVector.hpp"

// Generate an array of floats in the range [ 0 .. 1 )
// Performance and distribution quality are not exceptional, but sufficient for testing purposes
AlignedVector<float> randomFloats( size_t length, uint32_t seed = 11 );

// Prevent the optimizer from discarding the float value
void doNotOptimize( float v );

// Utility class to summarise the distribution of FP64 samples
class DistributionSummary
{
	// [ sum( e^2 ), sum( e ) ]
	__m128d sum = _mm_setzero_pd();
	// [ max( -e ), max( e ) ]
	__m128d minMax = _mm_set1_pd( -DBL_MAX );
	int64_t count = 0;

public:
	// Add a sample
	void add( double val ) noexcept
	{
		const __m128d vec = _mm_set1_pd( val ); // [ val, val ]
		const __m128d one = _mm_set1_pd( 1.0 );
		const __m128d vec1 = _mm_blend_pd( vec, one, 0b10 ); // [ val, 1.0 ]
		// Accumulate sum of squares and sum of values
		sum = _mm_fmadd_pd( vec, vec1, sum );

		// Update minMax vector
		__m128d tmp = _mm_addsub_pd( _mm_setzero_pd(), vec );	// [ -val, val ]
		minMax = _mm_max_pd( minMax, tmp );

		// Increment sample count
		count++;
	}

	struct Result
	{
		double average, stDev, min, max;
	};

	// Compute the summary
	Result result() const noexcept;
};

// Parse strings like "123", "12k" or "256M" into size_t
bool tryParseLength( size_t& result, const char* str );