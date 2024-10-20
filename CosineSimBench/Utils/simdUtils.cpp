#include "simdUtils.h"

#ifndef __AVX2__
const alignas( 64 ) std::array<int, 16> s_remainderMask =
{
	-1, -1, -1, -1, -1, -1, -1, -1,
	0, 0, 0, 0, 0, 0, 0, 0,
};
#endif