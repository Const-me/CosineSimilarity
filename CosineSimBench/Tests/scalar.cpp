#include <cmath>
#include "cosineSim.h"

float cosineSimScalar( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 )
{
	assert( vec1.size() == vec2.size() );

	float dot = 0, a2 = 0, b2 = 0;
	const size_t length = vec1.size();

	for( size_t i = 0; i < length; i++ )
	{
		const float a = vec1[ i ];
		const float b = vec2[ i ];
		dot += a * b;
		a2 += a * a;
		b2 += b * b;
	}

	return dot / ( std::sqrt( a2 ) * std::sqrt( b2 ) );
}