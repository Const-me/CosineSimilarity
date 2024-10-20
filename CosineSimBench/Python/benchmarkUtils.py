def try_parse_length(s):
	if not s or not s.strip():
		return False, None

	s = s.strip()  # Remove leading/trailing spaces

	# Try to parse the numeric part of the string
	try:
		result = int(s)
		end = ''
	except ValueError:
		# Handle cases where suffix might exist
		for i, char in enumerate(s):
			if not char.isdigit() and char != ' ':
				end = s[i:]
				try:
					result = int(s[:i].strip())
				except ValueError:
					return False, None
				break
		else:
			end = ''
			result = int(s)

	# Skip any spaces after the number
	end = end.strip()

	# Handle suffix if present
	if end == '':
		return True, result  # No suffix, valid as is

	if end in ('k', 'K'):
		result *= 1024
	elif end in ('M', 'm'):
		result *= 1024 * 1024
	elif end in ('G', 'g'):
		result *= 1024 * 1024 * 1024
	else:
		return False, None  # Invalid suffix

	return True, result

# Example usage:
# success, length = try_parse_length("256K")
# if success:
#     print(f"Parsed length: {length}")
# else:
#     print("Failed to parse length")