import numpy as np

# FNV-1a hash function
def fnv1a(val, hash_val=0):
	fnv_prime = 16777619
	hash_val ^= val
	hash_val *= fnv_prime
	return hash_val

# XorShift RNG function
def random_xorshift(x):
	x ^= (x << 13) & 0xFFFFFFFF
	x ^= (x >> 17) & 0xFFFFFFFF
	x ^= (x << 5) & 0xFFFFFFFF
	return x & 0xFFFFFFFF

# Function to convert 32-bit integer into random float
def make_random_float(bits):
	mantissa_mask = 0x007FFFFF
	one = 0x3F800000
	bits &= mantissa_mask
	bits |= one
	float_bits = np.array([bits], dtype=np.uint32).view(np.float32)
	return float_bits[0] - 1.0

# Function to generate a list of random floats
def random_floats(length, seed):
	seed ^= 0x5AEC34BF
	state = fnv1a(seed)

	v = np.empty(length, dtype='float32')
	for i in range(length):
		state = random_xorshift(state)
		v[i] = make_random_float(state)

	return np.asarray(v, dtype='float32', order='c')

# Example usage:
# length = 10
# seed = 12345
# random_array = random_floats(length, seed)
# print(random_array)