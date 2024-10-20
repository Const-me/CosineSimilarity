#pragma once
#include <stdlib.h>
#include <vector>

namespace MemAlloc
{
	// Allocate aligned block of memory
	inline void* alignedMalloc( size_t size, size_t alignment )
	{
#ifdef _MSC_VER
		return _aligned_malloc( size, alignment );
#else
		return aligned_alloc( alignment, size );
#endif
	}

	// Free aligned block of memory
	inline void alignedFree( void* ptr )
	{
#ifdef _MSC_VER
		_aligned_free( ptr );
#else
		free( ptr );
#endif
	}

	// Custom allocator with alignment = 32 bytes
	template <typename T>
	struct AlignedAllocator32
	{
		using value_type = T;
		AlignedAllocator32() = default;
		template <typename U>
		AlignedAllocator32( const AlignedAllocator32<U>& ) {}

		T* allocate( size_t n )
		{
			if( n != 0 )
			{
				void* const ptr = alignedMalloc( n * sizeof( T ), 32 );
				if( nullptr != ptr )
					return (T*)( ptr );
				else
					throw std::bad_alloc();
			}
			return nullptr;
		}

		void deallocate( T* ptr, std::size_t )
		{
			alignedFree( ptr );
		}
	};

	// Equality comparison for allocator
	template <typename T, typename U>
	bool operator==( const AlignedAllocator32<T>&, const AlignedAllocator32<U>& ) { return true; }

	template <typename T, typename U>
	bool operator!=( const AlignedAllocator32<T>&, const AlignedAllocator32<U>& ) { return false; }
}

// Typedef for 32-bytes aligned vector
template<typename T>
using AlignedVector = std::vector<T, MemAlloc::AlignedAllocator32<T>>;