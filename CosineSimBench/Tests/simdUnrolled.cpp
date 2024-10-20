#include "cosineSim.h"
#include "../Utils/simdUtils.h"

float cosineSimSimdUnrolled( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 )
{
	assert( vec1.size() == vec2.size() );

	const size_t length = vec1.size();
	assert( length != 0 );

	// Compute length of the remainder; 
	// We want a remainder of length [ 1 .. 32  ] instead of [ 0 .. 32 ]
	const ptrdiff_t rem = ( ( length - 1 ) % 32 ) + 1;

	const float* a = vec1.data();
	const float* b = vec2.data();
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

	// Reduce horizontally
	return a0.result();
}