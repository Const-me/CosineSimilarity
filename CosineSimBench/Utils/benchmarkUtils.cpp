#include <cmath> // std::isnan
#include "benchmarkUtils.h"

namespace
{
	// FNV-1a hash function
	inline uint32_t fnv1a( uint32_t val, uint32_t hash = 0 )
	{
		constexpr uint32_t fnv_prime = 16777619;
		hash ^= val;
		hash *= fnv_prime;
		return hash;
	}

	inline uint32_t randomXorShift( uint32_t& x )
	{
		// https://en.wikipedia.org/wiki/Xorshift#Example_implementation
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		return x;
	}

	inline float makeRandomFloat( uint32_t bits )
	{
		constexpr uint32_t mantissaMask = 0x007FFFFFu;
		constexpr uint32_t one = 0x3F800000;
		bits &= mantissaMask;
		bits |= one;
		__m128i iv = _mm_cvtsi32_si128( (int)bits );
		__m128 v = _mm_castsi128_ps( iv );
		return _mm_cvtss_f32( v ) - 1.0f;
	}
}

AlignedVector<float> randomFloats( size_t length, uint32_t seed )
{
	AlignedVector<float> v;
	v.resize( length );

	// The magic number is from https://www.random.org/
	seed ^= 0x5AEC34BF;
	uint32_t state = fnv1a( seed );

	for( float& e : v )
		e = makeRandomFloat( randomXorShift( state ) );
	return v;
}

void doNotOptimize( float v )
{
	if( !std::isnan( v ) )
		return;
#ifdef _MSC_VER
	__debugbreak();
#else
	asm( "int $3" );  // INT 3 assembly instruction for non-MSVC compilers
#endif
}

namespace
{
	inline void reduceSum( __m128d sum21, int64_t count, double& average, double& stdev ) noexcept
	{
		const __m128d zero = _mm_setzero_pd();
		__m128d tmp = _mm_cvtsi64_sd( zero, count );
		tmp = _mm_movedup_pd( tmp );

		// x = ( sum of squares ) / count, y = sum / count = average
		const __m128d scaled = _mm_div_pd( sum21, tmp );

		tmp = _mm_unpackhi_pd( scaled, scaled );
		average = _mm_cvtsd_f64( tmp );

		tmp = _mm_fnmadd_sd( tmp, tmp, scaled );
		tmp = _mm_sqrt_sd( tmp, tmp );
		stdev = _mm_cvtsd_f64( tmp );
	}

	inline void reduceMinMax( __m128d vec, double& min, double& max ) noexcept
	{
		min = -_mm_cvtsd_f64( vec );
		max = _mm_cvtsd_f64( _mm_unpackhi_pd( vec, vec ) );
	}
}

DistributionSummary::Result DistributionSummary::result() const noexcept
{
	Result res;
	reduceSum( sum, count, res.average, res.stDev );
	reduceMinMax( minMax, res.min, res.max );
	return res;
}

#include <cctype>
#include <cstring>
#include <cstdlib>

bool tryParseLength( size_t& result, const char* str )
{
	if( !str || !*str ) return false;
	while( std::isspace( *str ) ) str++;

	char* end;
	// Convert the numeric part of the string to size_t
	result = std::strtoull( str, &end, 10 );
	// Check if the number conversion failed (no valid number)
	if( end == str )
		return false;

	// Skip any spaces after the number
	while( std::isspace( *end ) ) end++;

	// Handle suffix if present
	if( *end == '\0' ) 
		return true; // No suffix, so valid as is

	if( *end == 'k' || *end == 'K' )
		result *= 1024;
	else if( *end == 'M' || *end == 'm' )
		result *= 1024 * 1024;
	else if( *end == 'G' || *end == 'g' )
		result *= 1024 * 1024 * 1024;
	else
		return false; // Invalid suffix

	// Ensure there's nothing else after the suffix
	return *( end + 1 ) == '\0';
}