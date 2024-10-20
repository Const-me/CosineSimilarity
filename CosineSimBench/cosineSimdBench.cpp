static const char* const logFileName = "benchmark-log.tsv";

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#else
#include <strings.h>
#endif
#include "Utils/benchmarkUtils.h"
#include "Tests/cosineSim.h"
#include <array>
#include <chrono>
#include <stdio.h>
#include <string>

namespace
{
	enum struct eTestCase: uint8_t
	{
		Scalar = 0,
		Naive = 1,
		Unrolled = 2,
		Parallel = 3,
	};

	static std::string rowHeader( eTestCase tc )
	{
		using namespace std::string_literals;
		switch( tc )
		{
		case eTestCase::Scalar:
			return "Scalar"s;
		case eTestCase::Naive:
			return "Naive"s;
		case eTestCase::Unrolled:
			return "Unroll"s;
		case eTestCase::Parallel:
			return "OMP( "s + std::to_string( getParallelThreadsCount() ) + " )"s;
		}
		return ""s;
	}

	constexpr size_t iterations = 1024;

	class Stopwatch
	{
		using time = std::chrono::high_resolution_clock::time_point;
		time begin;

	public:
		Stopwatch()
		{
			begin = std::chrono::high_resolution_clock::now();
		}
		double elapsedMilliseconds() const noexcept
		{
			time now = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> result = now - begin;
			return result.count();
		}
	};

	inline bool isEqual( const char* a, const char* b )
	{
#ifdef _MSC_VER
		return 0 == _stricmp( a, b );
#else
		return 0 == strcasecmp( a, b );
#endif
	}

	bool tryParseAlgo( const char* str, eTestCase& res )
	{
#define CMP( val ) if( isEqual( str, #val ) ) { res = eTestCase::val; return true; }
		CMP( Scalar );
		CMP( Naive );
		CMP( Unrolled );
		CMP( Parallel );
#undef CMP
		return false;
	}

	using pfnTestCase = float ( * )( const AlignedVector<float>& vec1, const AlignedVector<float>& vec2 );
	static const std::array<pfnTestCase, 4> s_dispatch =
	{
		&cosineSimScalar,
		&cosineSimSimdNaive,
		&cosineSimSimdUnrolled,
		&cosineSimSimdParallel,
	};
}

int main( int argc, const char** argv )
{
	if( argc != 3 )
	{
		printf( "Usage: %s <algo> <length>\n", argv[ 0 ] );
		return -2;
	}

	eTestCase testCase;
	if( !tryParseAlgo( argv[ 1 ], testCase ) )
	{
		printf( "Unable to parse algorithm into a number" );
		return -3;
	}

	size_t arrayLength;
	if( !tryParseLength( arrayLength, argv[ 2 ] ) )
	{
		printf( "Unable to parse string into a number" );
		return -4;
	}

	FILE* const logFile = fopen( logFileName, "a" );
	if( nullptr == logFile )
	{
		printf( "Unable to open the log file" );
		return -5;
	}

	// Generate input vectors
	AlignedVector<float> vec1 = randomFloats( arrayLength, 1 );
	AlignedVector<float> vec2 = randomFloats( arrayLength, 2 );
	const pfnTestCase pfn = s_dispatch[ (uint8_t)testCase ];

	DistributionSummary summary;
	float result;
	for( size_t i = 0; i < iterations; i++ )
	{
		Stopwatch sw;
		result = pfn( vec1, vec2 );
		doNotOptimize( result );
		summary.add( sw.elapsedMilliseconds() );
	}

	const auto res = summary.result();
	std::string header = rowHeader( testCase );

	printf( "%s, %s: average %g ms, range [ %g .. %g ] ms, st.dev %g ms\n",
		header.c_str(), argv[ 2 ],
		res.average, res.min, res.max, res.stDev );

	fprintf( logFile, "%s\t%zu\t%g\t%g\t%g\t%g\t%g\n",
		header.c_str(), arrayLength,
		res.average, res.min, res.max, res.stDev, result );

	fflush( logFile );
	fclose( logFile );
	return 0;
}