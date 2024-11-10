"""
Working code to encode 8 random generated data into 8 cdma_codes
Add all cdma codes and generate the summed_cdma_output
Pass summed_cdma_output to decode_cdma function and retrive
all 8 bytes and compare the transmitted and received signal.
"""


import numpy as np
import time

# Function to generate OVSF codes of SF=8
def generate_ovsf_codes(sf):
    if sf == 8:
        return np.array([
            [1,  1,  1,  1,  1,  1,  1,  1],
            [1, -1,  1, -1,  1, -1,  1, -1],
            [1,  1, -1, -1,  1,  1, -1, -1],
            [1, -1, -1,  1,  1, -1, -1,  1],
            [1,  1,  1,  1, -1, -1, -1, -1],
            [1, -1,  1, -1, -1,  1, -1,  1],
            [1,  1, -1, -1, -1, -1,  1,  1],
            [1, -1, -1,  1, -1,  1,  1, -1]
        ])

# Function to encode data using CDMA
def encode_cdma(data, pn_sequence):
    """Encode data using the selected PN sequence in CDMA."""
    data_mapped = np.where(data == 1, 1, -1)  # Map 1 to 1 and 0 to -1
    cdma_output = np.zeros(len(data_mapped) * len(pn_sequence))

    # Generate CDMA output by multiplying each data bit with the PN sequence
    for i, bit in enumerate(data_mapped):
        cdma_output[i * len(pn_sequence):(i + 1) * len(pn_sequence)] = bit * pn_sequence  # Multiply and place

    return cdma_output

# Function to decode CDMA
def decode_cdma(summed_cdma_output, ovsf_code_matrix):
    """
    Function to decode CDMA data using a matrix of OVSF codes.
    """
    # Determine the length of each OVSF code
    code_length = len(ovsf_code_matrix[0])
    num_codes = len(ovsf_code_matrix)
    num_chunks = len(summed_cdma_output) // code_length

    # Initialize an array to store the decoded data
    decoded_data = []

    # Loop through each OVSF code and perform despreading
    for code in ovsf_code_matrix:
        decoded_values = []
        for i in range(num_chunks):
            # Extract a chunk of the summed CDMA output
            chunk = summed_cdma_output[i * code_length:(i + 1) * code_length]
            # Perform the dot product to despread
            despread_value = np.dot(chunk, code)
            # Divide by the code length to get the original symbol (normalization step)
            decoded_symbol = despread_value // code_length
            decoded_values.append(decoded_symbol)
        decoded_data.append(decoded_values)

    return np.array(decoded_data)

# Convert -1 to 0 after decoding
def convert_to_original(decoded_data):
    """Convert -1 to 0 to get the original binary data."""
    return np.where(decoded_data == -1, 0, decoded_data)

# Step 1: Generate 8 random 8-bit bytes
num_bytes = 8
random_data = np.random.randint(0, 2, (num_bytes, 8))  # 8 bytes of random bits (0s and 1s)

# Step 2: Generate OVSF codes of SF=8
ovsf_codes = generate_ovsf_codes(8)

print("Original Data (8 bits each):")
print(random_data)

# Step 3: Initialize the summed CDMA output
summed_cdma_output = np.zeros(8 * 8)  # Initialize an array to hold the summed output

# Encode each byte using different OVSF codes and sum the results
print("\nEncoded CDMA Output for each byte:")
start_time_ns = time.time_ns()
for i in range(num_bytes):
    byte_data = random_data[i]  # Get the random byte
    ovsf_code = ovsf_codes[i]    # Get the corresponding OVSF code
    cdma_output = encode_cdma(byte_data, ovsf_code)  # Encode the byte
    print(cdma_output)
    summed_cdma_output += cdma_output  # Add to the summed output
end_time_ns = time.time_ns()
# Calculate the elapsed time in nanoseconds
execution_time_ns = end_time_ns - start_time_ns
print(f"Execution time EnCoding: {execution_time_ns} nanoseconds")

# Step 4: Decode the summed output using each OVSF code and check against original data
start_time_ns = time.time_ns()
decoded_data = decode_cdma(summed_cdma_output, ovsf_codes)

# Step 5: Convert decoded values (-1) to 0
converted_decoded_data = convert_to_original(decoded_data)
end_time_ns = time.time_ns()
# Calculate the elapsed time in nanoseconds
execution_time_ns = end_time_ns - start_time_ns
print(f"Execution time CDMA DeCoding: {execution_time_ns} nanoseconds")

# Print the results
print("\nSummed CDMA Output:")
print(summed_cdma_output)

print("\nDecoded Data (before conversion):")
print(decoded_data)

print("\nDecoded Data (after conversion, -1 to 0):")
print(converted_decoded_data)

# Step 6: Verify the decoded data matches the original data (bitwise comparison)
print("\nDecoded Data matches the Original Data:")
print(np.array_equal(converted_decoded_data, random_data))
