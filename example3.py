"""
Approach2 implemented here:
Random 8 bytes --> 16-qam modulation
16qam_sym x OVSF Code = Spread Signal
Despreading and retriving original qpsk_sym
and Decoding
this is done for single ovsf Code.

"""
import numpy as np
import matplotlib.pyplot as plt


# Function to generate random 8-bit data
def generate_random_data(num_bytes=8):
    random_data = np.random.randint(0, 2, (num_bytes, 8))
    return random_data


# Function for 16-QAM modulation
def qam16_modulate(data_bits):
    # Map bits to 16-QAM symbols
    # 4 bits per symbol (for 16-QAM, 16 symbols can be represented by 4 bits)
    bit_groups = data_bits.reshape(-1, 4)

    # Mapping 4 bits to 16-QAM symbols (using Gray coding or direct mapping for simplicity):
    # This is an example mapping:
    symbol_map = {
        (0, 0, 0, 0): 3 + 3j,
        (0, 0, 0, 1): 3 + 1j,
        (0, 0, 1, 0): 3 - 3j,
        (0, 0, 1, 1): 3 - 1j,
        (0, 1, 0, 0): 1 + 3j,
        (0, 1, 0, 1): 1 + 1j,
        (0, 1, 1, 0): 1 - 3j,
        (0, 1, 1, 1): 1 - 1j,
        (1, 0, 0, 0): -3 + 3j,
        (1, 0, 0, 1): -3 + 1j,
        (1, 0, 1, 0): -3 - 3j,
        (1, 0, 1, 1): -3 - 1j,
        (1, 1, 0, 0): -1 + 3j,
        (1, 1, 0, 1): -1 + 1j,
        (1, 1, 1, 0): -1 - 3j,
        (1, 1, 1, 1): -1 - 1j
    }

    qam16_symbols = np.array([symbol_map[tuple(bit_group)] for bit_group in bit_groups])

    return qam16_symbols


# Function to spread 16-QAM symbols using an OVSF code
def spread_symbols(qam16_symbols, ovsf_code, sf):
    spread_symbols = np.repeat(qam16_symbols, sf) * np.tile(ovsf_code, len(qam16_symbols))
    return spread_symbols

# Function to despread the signal
def despread_signal(spread_symbols, ovsf_code, sf):
    reshaped_signal = spread_symbols.reshape(-1, sf)
    despread_signal = np.sum(reshaped_signal * np.tile(ovsf_code, (reshaped_signal.shape[0], 1)), axis=1) / sf
    return despread_signal

# Function to map the complex despread signal back to 16-QAM symbols
def retrieve_qam16_symbols(despread_signal):
    # We need to map the real and imaginary parts back to 16-QAM symbols
    qam16_symbols = np.zeros(len(despread_signal), dtype=complex)

    # For simplicity, we assume the mapping is based on the real and imaginary parts being in the range of [-3, 3] or [-1, 1]
    real_part = np.round(despread_signal.real / 2) * 2  # Map real part to {-3, -1, 1, 3}
    imag_part = np.round(despread_signal.imag / 2) * 2  # Map imaginary part to {-3, -1, 1, 3}

    qam16_symbols.real = real_part
    qam16_symbols.imag = imag_part

    return qam16_symbols


# Function to decode the 16-QAM symbols back to bits
def qam16_demodulate(qam16_symbols):
    decoded_bits = np.zeros(4 * len(qam16_symbols), dtype=int)

    # Map the real and imaginary parts back to the 4-bit symbols for 16-QAM
    decoded_bits[0::4] = (qam16_symbols.real > 0).astype(int)  # real > 0 --> 1, else 0
    decoded_bits[1::4] = (qam16_symbols.imag > 0).astype(int)  # imag > 0 --> 1, else 0
    decoded_bits[2::4] = (qam16_symbols.real > 1).astype(int)  # Map the 16-QAM levels
    decoded_bits[3::4] = (qam16_symbols.imag > 1).astype(int)  # Map the 16-QAM levels

    return decoded_bits


# Main execution flow
def main():
    # Parameters
    sf = 8  # Spreading factor
    ovsf_code = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # OVSF code for spreading
    num_bytes = 8  # Number of bytes to generate

    # Step 1: Generate random 8-bit bytes
    random_data = generate_random_data(num_bytes)
    print("Original Data (8-bit bytes):\n", random_data)

    # Step 2: 16-QAM Modulation
    data_bits = random_data.flatten()  # Flatten the data to get a bit sequence
    qam16_symbols = qam16_modulate(data_bits)
    print("16-QAM Symbols:\n", qam16_symbols)

    # Step 3: Spreading 16-QAM Symbols
    spread_symbols_result = spread_symbols(qam16_symbols, ovsf_code, sf)
    print("Spread Symbols:\n", spread_symbols_result)

    # Step 4: Despreading the signal
    despread_signal_result = despread_signal(spread_symbols_result, ovsf_code, sf)
    print("Despread Signal:\n", despread_signal_result)

    if np.array_equal(despread_signal_result , qam16_symbols):
        print("MATCH yes")
        print("Demod-qam16 and qam16 are equal ")

# Step 5: Retrieve 16-QAM symbols
   # retrieved_qam16_symbols = retrieve_qam16_symbols(despread_signal_result)
    #print("Retrieved 16-QAM Symbols:\n", retrieved_qam16_symbols)

   # Step 6: 16-QAM Demodulation
    decoded_bits = qam16_demodulate(despread_signal_result)
    print("Decoded Bits:\n", decoded_bits)

    # Step 7: Reconstruct the decoded data (bits to bytes)
    decoded_bytes = decoded_bits.reshape(num_bytes, 8)
    print("Decoded Data (8-bit bytes):\n", decoded_bytes)

    # Step 8: Verify if the decoded data matches the original data
    data_match = np.array_equal(random_data, decoded_bytes)
    print("\nData Match:", data_match)
    print("Original Data (8-bit bytes):\n", random_data)


if __name__ == "__main__":
    main()
