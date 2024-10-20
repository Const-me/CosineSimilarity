import sys
import time
import numpy as np
from benchmarkUtils import try_parse_length
from randomFloats import random_floats

EXECUTIONS = 1024
logFileName = "python-log.tsv";

def main():
	if len(sys.argv) < 2:
		print("No argument provided")
		return -2

	success, array_length = try_parse_length( sys.argv[ 1 ] )
	if not success:
		print("Unable to parse string into a number")
		return -4

	# print(f"Parsed array length: {array_length}")
	A = random_floats( array_length, 1 )
	B = random_floats( array_length, 2 )
	
	accum = 0
	i = sys.float_info.max

	for _ in range(EXECUTIONS):
		start_time = time.time()
		cos_sim = 1 - np.dot( A, B ) / ( np.linalg.norm( A ) * np.linalg.norm( B ))
		ms = (time.time() - start_time) * 1000
		accum += ms
		i = min( i, ms )

	avg = accum / EXECUTIONS
	print( f'Python, { sys.argv[ 1 ] }: average { avg } ms, min { i }\n' )
	with open( logFileName, 'a' ) as file:
		file.write( f'Python\t{array_length}\t{ avg }\t{ i }\n' )

# Call the main function
if __name__ == "__main__":
	sys.exit( main() )