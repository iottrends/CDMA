"""

Working code to encode 8 random generated data into 8 cdma_codes
Generate a composite qpsk symbol.
Decode composite qpsk symbol to generate composite cdma_sym
decode and retrive all 8 bytes from the composite cdma_sym and
verify the data 8 bytes.
all 8 bytes and compare the transmitted and received signal.
Add AWGN
"""


import numpy as np
import matplotlib.pyplot as plt
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


def add_awgn(composite_qpsk_signal, snr_db):
    """
    Add AWGN noise to a composite QPSK signal.

    Parameters:
    composite_qpsk_signal (numpy.ndarray): Input composite QPSK signal.
    snr_db (float): Signal-to-noise ratio in dB.

    Returns:
    numpy.ndarray: QPSK signal with added AWGN noise.
    """
    # Calculate the signal power
    signal_power = np.mean(np.abs(composite_qpsk_signal) ** 2)

    # Calculate the noise power based on the desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate AWGN noise (real and imaginary parts independently)
    noise = np.sqrt(noise_power / 2) * (
                np.random.randn(*composite_qpsk_signal.shape) + 1j * np.random.randn(*composite_qpsk_signal.shape))

    # Add the noise to the signal
    noisy_signal = composite_qpsk_signal + noise

    return noisy_signal


def plot_qpsk_signals(combined_qpsk_signal, noisy_signal):
    """
    Plot the constellation diagram for both the original QPSK signal and the noisy signal.

    Parameters:
    combined_qpsk_signal (numpy.ndarray): Original composite QPSK signal.
    noisy_signal (numpy.ndarray): Noisy QPSK signal after adding AWGN.
    """
    plt.figure(figsize=(10, 10))

    # Plot original QPSK signal
    plt.scatter(combined_qpsk_signal.real, combined_qpsk_signal.imag, color='blue', label='Original QPSK', marker='o',
                alpha=0.7)

    # Plot noisy QPSK signal
    plt.scatter(noisy_signal.real, noisy_signal.imag, color='red', label='Noisy QPSK', marker='x', alpha=0.7)

    # Add title and labels
    plt.title('QPSK Constellation Diagram: Original vs Noisy', fontsize=14)
    plt.xlabel('In-Phase (Real)', fontsize=12)
    plt.ylabel('Quadrature (Imaginary)', fontsize=12)

    # Add legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.show()


############################################################
def main():
    # Step 1: Generate 8 random 8-bit bytes
    num_bytes = 8
    snr_db = 10
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

    print("\nsummed_cdma_output")
    print(summed_cdma_output)

    print("\n combined_qpsk_signal")
    print(combined_qpsk_signal)
#Add noise to the signal.

    # Add AWGN noise
    noisy_signal = add_awgn(combined_qpsk_signal, snr_db)
    print("\n noisy Signal")
    print(noisy_signal)


    ###Demodulation########
    # Step 5: Extract composite_cdma_output from combined QPSK signal
    #composite_cdma = qpsk_to_cdma_map(combined_qpsk_signal)
    composite_cdma = qpsk_to_cdma_map(noisy_signal)
    print("composite cdma output:\n")
    print(composite_cdma)
    # Initialize an array to store the decoded data
    #decoded_data = decode_cdma(summed_cdma_output, ovsf_codes)
    decoded_data = decode_cdma(composite_cdma, ovsf_codes)
    # Step 6: Convert decoded values (-1) to 0
    converted_decoded_data = convert_to_original(decoded_data)
    print("\nDecoded Data (before conversion):")
    print(decoded_data)

    print("\nDecoded Data (after conversion, -1 to 0):")
    print(converted_decoded_data)

    # Step 7: Verify the decoded data matches the original data (bitwise comparison)
    print("\nDecoded Data matches the Original Data:")
    print(np.array_equal(converted_decoded_data, random_data))
    print("Original Data (8-bit bytes):\n", random_data)
#step 8 Plot the composite qpsk signal and noisy signal
    plot_qpsk_signals(combined_qpsk_signal, noisy_signal)


if __name__ == "__main__":
    main()
