#include "cosineSim.h"
#include "../Utils/simdUtils.h"

float cosineSimSimdNaive( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 )
{
	assert( vec1.size() == vec2.size() );

	const size_t length = vec1.size();
	assert( length != 0 );

	// Compute length of the remainder; 
	// We want a remainder of length [ 1 .. 8 ] instead of [ 0 .. 7 ]
	const ptrdiff_t rem = ( ( length - 1 ) % 8 ) + 1;

	const float* a = vec1.data();
	const float* b = vec2.data();
	const float* const aEnd = a + length - rem;

	Accumulator acc;
	// Each iteration of the loop 8 elements from each input vector
	for( ; a < aEnd; a += 8, b += 8 )
		acc.add( a, b );

	// Handle the last, possibly incomplete vector of length [ 1 .. 8 ]
	acc.add( a, b, rem );

	return acc.result();
}