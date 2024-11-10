"""
Approach2 implemented here:
Random 8 bytes --> qpsk modulation
qpsk_sym x OVSF Code = Spread Signal
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

# Function for QPSK modulation
def qpsk_modulate(data_bits):
    # Map bits to QPSK symbols: 00 -> 1+1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> -1-1j
    bit_pairs = data_bits.reshape(-1, 2)
    qpsk_symbols = (1 - 2 * bit_pairs[:, 0]) + 1j * (1 - 2 * bit_pairs[:, 1])
    return qpsk_symbols

# Function to spread QPSK symbols using an OVSF code
def spread_symbols(qpsk_symbols, ovsf_code, sf):
    spread_symbols = np.repeat(qpsk_symbols, sf) * np.tile(ovsf_code, len(qpsk_symbols))
    return spread_symbols

# Function to despread the signal
def despread_signal(spread_symbols, ovsf_code, sf):
    num_spread_symbols = len(spread_symbols) // sf
    reshaped_signal = spread_symbols[:num_spread_symbols * sf].reshape(num_spread_symbols, sf)
    despread_signal = reshaped_signal @ ovsf_code / sf
    return despread_signal

# Function to demodulate QPSK symbols and extract bits
def qpsk_demodulate(despread_signal):
    # Map the real and imaginary parts of the QPSK symbols to bits
    decoded_bits = np.zeros(2 * len(despread_signal), dtype=int)
    decoded_bits[0::2] = (despread_signal.real < 0).astype(int)  # 0 for real part < 0, 1 for real part > 0
    decoded_bits[1::2] = (despread_signal.imag < 0).astype(int)  # 0 for imag part < 0, 1 for imag part > 0
    return decoded_bits

# Function to reconstruct decoded bits into bytes
def bits_to_bytes(decoded_bits, num_bytes):
    decoded_bytes = decoded_bits.reshape(num_bytes, 8)
    return decoded_bytes



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


def qpsk_tx_chain(random_data, ovsf_code, sf):
    # Step 2: QPSK Modulation
    data_bits = random_data.flatten()  # Flatten the data to get a bit sequence
    qpsk_symbols = qpsk_modulate(data_bits)
    print("QPSK Symbols:\n", qpsk_symbols)

    # Step 3: Spreading
    spread_symbols_result = spread_symbols(qpsk_symbols, ovsf_code, sf)
    print("STEP-3 Spread Symbols:\n", spread_symbols_result)
    print("spread_symbol len:", len(spread_symbols_result))

    # Add AWGN noise
    snr_db = 10
    noisy_signal = add_awgn(spread_symbols_result, snr_db)
    print("\n noisy Signal")
    print(noisy_signal)
    # plot_qpsk_signals(qpsk_symbols, noisy_signal)
    return noisy_signal


def qpsk_rx_chain(qpsk_cdma_noisy_signal, ovsf_code, sf):
    num_bytes = 8
    # Step 4: Despread and Demodulate
    # despread_signal_result = despread_signal(spread_symbols_result, ovsf_code, sf)
    despread_signal_result = despread_signal(qpsk_cdma_noisy_signal, ovsf_code, sf)
    print("STEP4 Despread Signal:\n", despread_signal_result)

    # Step 5: QPSK Demodulation
    decoded_bits = qpsk_demodulate(despread_signal_result)
    print(" STEP5 Decoded Bits:\n", decoded_bits)

    # Step 6: Reconstruct Decoded Bytes
    decoded_bytes = bits_to_bytes(decoded_bits, num_bytes)
    print("STEP 6 Decoded Data (8-bit bytes):\n", decoded_bytes)
    return decoded_bytes

# Main execution flow
def main():
    # Parameters
    sf = 8  # Spreading factor
   # ovsf_code = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # OVSF code for user 1
    num_bytes = 8  # Number of bytes to generate
    # Define 8 OVSF codes
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
    num_of_users = 8
    print("inside main!!!")
    # Step 1: Generate 8 random 8-bit bytes
    for i in range(num_of_users):

        random_data = generate_random_data(num_bytes)
        print("Original Data (8-bit bytes):\n", random_data)
        qpsk_cdma_noisy_signal = qpsk_tx_chain(random_data,ovsf_codes[i],sf)

    ####DEMODULATION#######
        print("#####DEMODULATION#####")
        decoded_bytes = qpsk_rx_chain(qpsk_cdma_noisy_signal, ovsf_codes[i], sf)
    # Step 7: Verify if the decoded data matches the original data
        data_match = np.array_equal(random_data, decoded_bytes)
        print("\nData Match:", data_match)
        print("Original Data (8-bit bytes):\n", random_data)


if __name__ == "__main__":
    main()
