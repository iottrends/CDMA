"""

Working code to encode 8 random generated data into 8 cdma_codes
Generate a composite qpsk symbol.
Decode composite qpsk symbol to generate composite cdma_sym
decode and retrive all 8 bytes from the composite cdma_sym and
verify the data 8 bytes.
all 8 bytes and compare the transmitted and received signal.
"""


import numpy as np

# Function to generate OVSF codes of SF=8
def generate_ovsf_codes(sf):
    if sf == 8:
        return np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1]
        ])

# Function to encode data using CDMA
def encode_cdma(data, pn_sequence):
    data_mapped = np.where(data == 1, 1, -1)  # Map 1 to 1 and 0 to -1
    cdma_output = np.zeros(len(data_mapped) * len(pn_sequence))

    for i, bit in enumerate(data_mapped):
        cdma_output[i * len(pn_sequence):(i + 1) * len(pn_sequence)] = bit * pn_sequence

    return cdma_output

# Convert -1 to 0 after decoding
def convert_to_original(decoded_data):
    return np.where(decoded_data == -1, 0, decoded_data)

# Function to perform QPSK modulation
def qpsk_modulate(cdma_output):
    real_part = cdma_output[::2]  # Even-indexed bits
    imag_part = cdma_output[1::2]  # Odd-indexed bits
    qpsk_symbols = real_part + 1j * imag_part
    return qpsk_symbols  # No power normalization here

def qpsk_to_cdma_map(qpsk_signal):
    """
    Convert QPSK modulated signal to CDMA format by interleaving real and imaginary parts.

    Parameters:
    qpsk_signal (numpy array): Complex QPSK signal array.

    Returns:
    numpy array: Interleaved real and imaginary parts.
    """
    # Extract real and imaginary parts
    real_parts = np.real(qpsk_signal)
    imaginary_parts = np.imag(qpsk_signal)
    interleaved_array = np.zeros(8 * 8)
    # Interleave real and imaginary parts
    #interleaved_array = np.empty(qpsk_signal.shape[0] * 2, dtype=qpsk_signal.dtype)
    interleaved_array[0::2] = real_parts
    interleaved_array[1::2] = imaginary_parts

    return interleaved_array


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



############################################################
def main():
    # Step 1: Generate 8 random 8-bit bytes
    num_bytes = 8
    random_data = np.random.randint(0, 2, (num_bytes, 8))

    # Step 2: Generate OVSF codes of SF=8
    ovsf_codes = generate_ovsf_codes(8)

    # Step 3: Initialize the combined QPSK signal
    combined_qpsk_signal = np.zeros(8 * 4, dtype=complex)
    summed_cdma_output = np.zeros(8 * 8)  # Initialize an array to hold the summed output
    print("Original Data (8 bits each):")
    print(random_data)

    #Step 4: convert from cdma to composite_qpsk_signal
    # Encode each byte, spread it using OVSF codes, modulate with QPSK, and sum the results
    for i in range(num_bytes):
        byte_data = random_data[i]
        ovsf_code = ovsf_codes[i]
        cdma_output = encode_cdma(byte_data, ovsf_code)
        summed_cdma_output += cdma_output  # Add to the summed output
        qpsk_symbols = qpsk_modulate(cdma_output)
        combined_qpsk_signal += qpsk_symbols

    print("summed_cdma_output \n")
    print(summed_cdma_output)

    print("combined_qpsk_signal\n")
    print(combined_qpsk_signal)

    ###Demodulation########
    # Step 5: Extract composite_cdma_output from combined QPSK signal
    composite_cdma = qpsk_to_cdma_map(combined_qpsk_signal)
    print("composite cdma output:\n")
    print(composite_cdma)
    # Initialize an array to store the decoded data
    decoded_data = decode_cdma(summed_cdma_output, ovsf_codes)
    # Step 5: Convert decoded values (-1) to 0
    converted_decoded_data = convert_to_original(decoded_data)
    print("\nDecoded Data (before conversion):")
    print(decoded_data)

    print("\nDecoded Data (after conversion, -1 to 0):")
    print(converted_decoded_data)

    # Step 6: Verify the decoded data matches the original data (bitwise comparison)
    print("\nDecoded Data matches the Original Data:")
    print(np.array_equal(converted_decoded_data, random_data))

if __name__ == "__main__":
    main()
