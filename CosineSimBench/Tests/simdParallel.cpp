#include "cosineSim.h"
#include "../Utils/simdUtils.h"

namespace
{
	// Count of CPU threads to use for computations
	constexpr int THREADS = 8;

	inline __m128 computeBatch( const float* a, const float* b, size_t length )
	{
		// Compute length of the remainder; 
		// We want a remainder of length [ 1 .. 32  ] instead of [ 0 .. 32 ]
		const ptrdiff_t rem = ( ( length - 1 ) % 32 ) + 1;
		const float* const aEnd = a + length - rem;

		// Each accumulator consumes 3 vectors so 12 total
		// The instruction set defines 16 of them so we're good, need 2 extra for input vectors
		Accumulator a0, a1, a2, a3;

		// Each iteration of the loop handles 32 elements from each input vector
		for( ; a < aEnd; a += 32, b += 32 )
		{
			a0.add<0>( a, b );
			a1.add<1>( a, b );
			a2.add<2>( a, b );
			a3.add<3>( a, b );
		}

		// Handle the last, possibly incomplete batch of length [ 1 .. 32 ]
		a0.add<0>( a, b, rem );
		a1.add<1>( a, b, rem );
		a2.add<2>( a, b, rem );
		a3.add<3>( a, b, rem );

		// Reduce vertically into a0
		a0 += a1;
		a2 += a3;
		a0 += a2;

		// Reduce horizontally but don't compute any products of square roots, return float3 vector
		return a0.reduceToVector();
	}

	// Some template metaprogramming to add small arrays of __m128 using pairwise summation
	template<size_t length>
	__m128 reduceImpl( const __m128* rsi );
	template<>
	__forceinline __m128 reduceImpl<0>( const __m128* rsi )
	{
		return _mm_setzero_ps();
	}
	template<>
	__forceinline __m128 reduceImpl<1>( const __m128* rsi )
	{
		return *rsi;
	}
	template<size_t length>
	__forceinline __m128 reduceImpl( const __m128* rsi )
	{
		constexpr size_t half = ( length + 1 ) / 2;
		__m128 a = reduceImpl<half>( rsi );
		__m128 b = reduceImpl<length - half>( rsi + half );
		return _mm_add_ps( a, b );
	}

#ifndef _MSC_VER
// GCC generates stupid warnings for std::array<__m128> type :-(
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

	template<size_t length>
	__forceinline __m128 reduceThreads( const std::array<__m128, length>& arr )
	{
		return reduceImpl<length>( arr.data() );
	}
}

float cosineSimSimdParallel( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 )
{
	assert( vec1.size() == vec2.size() );

	const size_t lengthFloats = vec1.size();
	const size_t lengthVectors = ( lengthFloats + 7 ) / 8;
	assert( lengthVectors >= THREADS );

	const float* const a = vec1.data();
	const float* const b = vec2.data();

	std::array<__m128, THREADS> results;

#pragma omp parallel for
	for( int i = 0; i < THREADS; i++ )
	{
		const size_t idx = (uint32_t)i;
		const size_t vBegin = idx * lengthVectors / (size_t)THREADS;
		const size_t vEnd = ( idx + 1 ) * lengthVectors / (size_t)THREADS;

		const size_t offset = vBegin * 8;
		const size_t batchLength = std::min( ( vEnd - vBegin ) * 8, lengthFloats - offset );

		results[ idx ] = computeBatch( a + offset, b + offset, batchLength );
	}

	const __m128 res = reduceThreads( results );

	return Accumulator::computeResult( res );
}

int getParallelThreadsCount()
{
	return THREADS;
}