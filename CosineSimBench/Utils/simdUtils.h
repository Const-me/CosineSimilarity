#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <array>
#include <algorithm>

#ifndef _MSC_VER
#define __forceinline inline __attribute__((always_inline))
#endif

#ifndef __AVX2__
extern const std::array<int, 16> s_remainderMask;
#endif

// Generate a mask for loading remainder elements with _mm256_maskload_ps instruction
// The length is clipped into [ 0 .. 8 ] interval
__forceinline __m256i remainderLoadMask( ptrdiff_t length )
{
	ptrdiff_t missingLanes = ptrdiff_t( 8 ) - length;

#ifdef __AVX2__
	// Make a mask of 8 bytes
	// These aren't branches, they should compile to conditional moves
	missingLanes = std::max( missingLanes, (ptrdiff_t)0 );
	uint64_t mask = -(int64_t)( missingLanes < 8 );
	mask >>= missingLanes * 8;
	// Sign extend the bytes into int32 lanes in AVX vector
	__m128i tmp = _mm_cvtsi64_si128( (int64_t)mask );
	return _mm256_cvtepi8_epi32( tmp );
#else
	// These aren't branches, they compile to conditional moves
	missingLanes = std::max( missingLanes, (ptrdiff_t)0 );
	missingLanes = std::min( missingLanes, (ptrdiff_t)8 );
	// Unaligned load from the constant array
	const int* rsi = &s_remainderMask[ missingLanes ];
	return _mm256_loadu_si256( ( const __m256i* )rsi );
#endif
}

// Compute horizontal sum of 3 SSE vectors
// 3 additions, 4 shuffles, 2 immediate blends, one xor
__forceinline __m128 hadd3x4( __m128 a, __m128 b, __m128 c )
{
	// a = [ a.xy + a.zw, b.xy + b.zw ]
	__m128 t0 = _mm_shuffle_ps( a, b, _MM_SHUFFLE( 1, 0, 3, 2 ) );
	__m128 t1 = _mm_blend_ps( a, b, 0b1100 );
	a = _mm_add_ps( t0, t1 );

	// c.xy += c.zw
	c = _mm_add_ps( c, _mm_movehl_ps( c, c ) );
	// c.zw = 0.0
	// can do with _mm_insert_ps however on Skylake it competes for the only shuffling port `p5`
	// xor / blend combo is probably slightly better there
	const __m128 zero = _mm_setzero_ps();
	c = _mm_blend_ps( c, zero, 0b1100 );

	// [ a.x, a.z, c.x, c.w ]
	t0 = _mm_shuffle_ps( a, c, _MM_SHUFFLE( 3, 0, 2, 0 ) );
	// [ a.y, a.w, c.y, c.w ]
	t1 = _mm_shuffle_ps( a, c, _MM_SHUFFLE( 3, 1, 3, 1 ) );

	return _mm_add_ps( t0, t1 );
}

// Compute horizontal sum of 3 AVX vectors
__forceinline __m128 hadd3x8( __m256 a, __m256 b, __m256 c )
{
	// Reduce 3x8 into 3x4
	__m128 a4 = _mm256_extractf128_ps( a, 1 );
	__m128 b4 = _mm256_extractf128_ps( b, 1 );
	__m128 c4 = _mm256_extractf128_ps( c, 1 );

	a4 = _mm_add_ps( a4, _mm256_castps256_ps128( a ) );
	b4 = _mm_add_ps( b4, _mm256_castps256_ps128( b ) );
	c4 = _mm_add_ps( c4, _mm256_castps256_ps128( c ) );

	// Compute horizontal sum of these, making a single vector
	return hadd3x4( a4, b4, c4 );
}

// AVX accumulators for the cosine similarity algorithm
class Accumulator
{
	__m256 a2, b2, dot;

	__forceinline void addVectors( const __m256 a, const __m256 b )
	{
		a2 = _mm256_fmadd_ps( a, a, a2 );
		b2 = _mm256_fmadd_ps( b, b, b2 );
		dot = _mm256_fmadd_ps( a, b, dot );
	}

public:
	// Zero initialize all 3 vectors
	Accumulator()
	{
		a2 = _mm256_setzero_ps();
		b2 = _mm256_setzero_ps();
		dot = _mm256_setzero_ps();
	}

	// Load exactly 8 numbers from each pointer, and accumulate
	template<int offsetVectors = 0>
	__forceinline void add( const float* a, const float* b )
	{
		constexpr ptrdiff_t offsetFloats = offsetVectors * 8;
		const __m256 v1 = _mm256_loadu_ps( a + offsetFloats );
		const __m256 v2 = _mm256_loadu_ps( b + offsetFloats );

		addVectors( v1, v2 );
	}

	// Load up to 8 numbers from each pointer, and accumulate
	template<int offsetVectors = 0>
	__forceinline void add( const float* a, const float* b, ptrdiff_t length )
	{
		constexpr ptrdiff_t offsetFloats = offsetVectors * 8;
		const __m256i mask = remainderLoadMask( length - offsetFloats );
		const __m256 v1 = _mm256_maskload_ps( a + offsetFloats, mask );
		const __m256 v2 = _mm256_maskload_ps( b + offsetFloats, mask );

		addVectors( v1, v2 );
	}

	// Add numbers from another accumulator
	__forceinline void operator +=( const Accumulator& that )
	{
		dot = _mm256_add_ps( dot, that.dot );
		a2 = _mm256_add_ps( a2, that.a2 );
		b2 = _mm256_add_ps( b2, that.b2 );
	}

	// Compute horizontal sum of each of the 3 vectors in this class, 
	// making an SSE vector [ hadd( a2 ), hadd( b2 ), hadd( dot ), 0.0f ]
	__forceinline __m128 reduceToVector() const
	{
		return hadd3x8( a2, b2, dot );
	}

	// v.z / ( sqrt( v.x ) * sqrt( v.y ) )
	static __forceinline float computeResult( __m128 v )
	{
		const float mul = _mm_cvtss_f32( _mm_movehl_ps( v, v ) );
		v = _mm_sqrt_ps( v );
		v = _mm_mul_ss( v, _mm_movehdup_ps( v ) );
		return mul / _mm_cvtss_f32( v );
	}

	__forceinline float result() const
	{
		return computeResult( reduceToVector() );
	}
};