import numpy as np


# Function to generate random 8-bit data for each user
def generate_random_data(num_bytes):
    return np.random.randint(0, 2, (num_bytes, 8))


# Function to modulate data using QPSK
def qpsk_modulate(data_bits):
    # Ensure that data_bits has an even number of bits (pairs of bits for QPSK)
    if len(data_bits) % 2 != 0:
        data_bits = np.append(data_bits, 0)  # Add a zero bit if odd

    # Map pairs of bits to QPSK symbols (-1, 1)
    symbols = 2 * data_bits.reshape(-1, 2) - 1  # Reshape to pairs and map to (-1, 1)
    qpsk_symbols = symbols[:, 0] + 1j * symbols[:, 1]  # Map pairs of bits to complex QPSK symbols
    return qpsk_symbols


# Function to spread QPSK symbols using OVSF codes
def spread_symbols(qpsk_symbols, ovsf_code, sf):
    spread_symbols = []
    for symbol in qpsk_symbols:
        spread_symbol = symbol * np.array(ovsf_code)
        spread_symbols.extend(spread_symbol)
    return np.array(spread_symbols)


# Function to add AWGN noise to the signal
def add_awgn(signal, snr_db):
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    noisy_signal = signal + noise
    return noisy_signal


# Function to despread the received noisy signal using the OVSF codes

def despread_signal(noisy_signal, ovsf_code, sf):
    # Calculate how many repetitions of the OVSF code are needed to match the length of the noisy signal
    num_repeats = len(noisy_signal) // len(ovsf_code)

    # Repeat the OVSF code to match the length of the noisy signal
    extended_ovsf_code = np.tile(ovsf_code, num_repeats)

    # Despread the noisy signal by multiplying with the extended OVSF code
    despreaded_signal = noisy_signal * extended_ovsf_code
    return despreaded_signal


# Function to demodulate QPSK symbols (from complex to binary bits)
def qpsk_demodulate(despread_signal):
    real_part = np.real(despread_signal)
    imag_part = np.imag(despread_signal)
    bits = np.array([(1 if real_part[i] > 0 else 0, 1 if imag_part[i] > 0 else 0) for i in range(len(real_part))])
    return bits.flatten()


# Function to convert bits to bytes
def bits_to_bytes(bits, num_bytes):
    bytes_data = np.array([bits[i:i + 8] for i in range(0, len(bits), 8)])
    return bytes_data[:num_bytes]


# Transmit data for all users in parallel
# Transmit data for all users in parallel
def qpsk_tx_chain(random_data, ovsf_codes, sf):
    qpsk_symbols_list = []
    spread_symbols_list = []

    # Modulate and spread data for all users in parallel
    for i, data in enumerate(random_data):
        qpsk_symbols = qpsk_modulate(data.flatten())  # Flatten and modulate
        user_spread_symbols = spread_symbols(qpsk_symbols, ovsf_codes[i], sf)  # Spread using respective OVSF code
        qpsk_symbols_list.append(qpsk_symbols)
        spread_symbols_list.append(user_spread_symbols)

    # Combine all users' spread symbols
    composite_qpsk_signal = np.sum(np.array(spread_symbols_list), axis=0)

    # Add AWGN noise
    snr_db = 10  # Signal-to-noise ratio in dB
    noisy_signal = add_awgn(composite_qpsk_signal, snr_db)
    return noisy_signal, random_data


# Receiver function to handle parallel transmission and demodulate
def qpsk_rx_chain(noisy_signal, ovsf_codes, sf, num_users):
    decoded_data = []

    # Perform despreading and demodulation for each user
    for i, ovsf_code in enumerate(ovsf_codes):
        # Despread the noisy signal using the respective OVSF code
        despread_signal_result = despread_signal(noisy_signal, ovsf_code, sf)

        # QPSK Demodulation
        decoded_bits = qpsk_demodulate(despread_signal_result)

        # Convert bits to bytes
        decoded_bytes = bits_to_bytes(decoded_bits, num_users)

        decoded_data.append(decoded_bytes)

    return decoded_data


# Main execution flow for parallel transmission
def main():
    sf = 8  # Spreading factor
    num_bytes = 8  # Number of bytes per user
    num_users = 8  # Number of users
    ovsf_codes = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1]
    ]

    # Step 1: Generate random data for each user
    random_data = [generate_random_data(num_bytes) for _ in range(num_users)]
    print("Original Data (8-bit bytes):\n", random_data)

    # Step 2: Transmit data for all users in parallel
    noisy_signal, original_data = qpsk_tx_chain(random_data, ovsf_codes, sf)
    print("\nNoisy Signal (Composite QPSK):\n", noisy_signal)

    # Step 3: Receive and demodulate the noisy signal
    decoded_data = qpsk_rx_chain(noisy_signal, ovsf_codes, sf, num_users)
    print("\ndecoded_data")
    print(decoded_data)
""""
    # Step 4: Verify if the decoded data matches the original data
    for i, decoded in enumerate(decoded_data):
        data_match = np.array_equal(original_data[i], decoded)
        print(f"Data Match for User {i + 1}: {data_match}")
        print("Original Data (8-bit bytes):\n", original_data[i])
        print("Decoded Data (8-bit bytes):\n", decoded)
"""

if __name__ == "__main__":
    main()
