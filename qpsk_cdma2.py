"""
working code with
multi-SF composite signal.
Here we are generating 8 bytes of random data for each user[8 users]
converting those 8 bytes to qpsk symbol.
multiplying qpsk symbol by ovsf_code[user] generating the Spread signal.
then adding it for all 8 users to generate a composite_cdma_signal.
therefore, 64 bytes of data is encoded and spread in composite_cdma_signal.
then Added Noise to this Signal.

After this signal is decoded by rx_chain
and we are able to extract the qpsk_sym[user] after despreading the noisy composite cdma signal.
We are getting noisy qpsk signal[-1.2 , +1.2] and this is good!!.
We successfuly spread the qpsk symoland despread it from the composite symbol.
What it means?
we packed 64 bytes of data using 8 ovsf codes, into a composite signal.
and then we were able to extract back all the individual qpsk signal for each user [-1 to +1 with some noise]
and then qpsk demod we mapped back to bits.
compared the received bytes with original and it is matching all 64 bits.

"""


import numpy as np
import matplotlib.pyplot as plt
import time

# Function to generate random 8-bit data for each user
def generate_random_data(num_bytes):
    return np.random.randint(0, 2, (num_bytes, 8))

# Function to modulate data using QPSK
def qpsk_modulate(data_bits):
    if len(data_bits) % 2 != 0:
        data_bits = np.append(data_bits, 0)  # Add a zero bit if odd

    symbols = 2 * data_bits.reshape(-1, 2) - 1  # Map to (-1, 1)
    qpsk_symbols = symbols[:, 0] + 1j * symbols[:, 1]
    return qpsk_symbols

# Function to spread QPSK symbols using OVSF codes
def spread_symbols(qpsk_symbols, ovsf_code, sf):
    # Repeat each QPSK symbol by the spreading factor
    spread_symbols = np.repeat(qpsk_symbols, sf)

    # Tile the OVSF code to match the length of the repeated symbols
    spread_symbol = spread_symbols * np.tile(ovsf_code, len(qpsk_symbols))
    return spread_symbol

# Function to add AWGN noise
def add_awgn(signal, snr_db):
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

# Function to despread the received signal using OVSF codes
def despread_signal(noisy_signal, ovsf_code, sf):
    # Reshape noisy signal into chunks of size sf
    reshaped_signal = noisy_signal.reshape(-1, sf)

    # Perform dot product between each chunk and OVSF code, then normalize by dividing by sf
    despread_signal = np.dot(reshaped_signal, ovsf_code) / sf

    return despread_signal

# Function to demodulate QPSK symbols
def qpsk_demodulate(despread_signal):
    real_part = np.real(despread_signal)
    imag_part = np.imag(despread_signal)
    bits = np.array([(1 if real > 0 else 0, 1 if imag > 0 else 0) for real, imag in zip(real_part, imag_part)])
    return bits.flatten()

# Function to convert bits to bytes
def bits_to_bytes(bits, num_bytes):
    bytes_data = np.array([bits[i:i + 8] for i in range(0, len(bits), 8)])
    return bytes_data[:num_bytes]

# Function to transmit QPSK for a user
def qpsk_tx_chain(random_data, ovsf_code, sf):
    qpsk_symbols = qpsk_modulate(random_data.flatten())
    spread_symbols_output = spread_symbols(qpsk_symbols, ovsf_code, sf)
    return spread_symbols_output

# Function to decode QPSK for a user
def qpsk_rx_chain(noisy_signal, ovsf_code, sf, num_bytes):
    despread_signal_result = despread_signal(noisy_signal, ovsf_code, sf)
    #print("\n Despread Signal")
    #print(despread_signal_result)
    decoded_bits = qpsk_demodulate(despread_signal_result)
    decoded_bytes = bits_to_bytes(decoded_bits, num_bytes)
    return decoded_bytes

# Function to plot QPSK symbols for all users
def plot_qpsk_symbols(original_data_list, noisy_signal, ovsf_codes, snr_db, num_of_users, sf):
    plt.figure(figsize=(8, 8))

    # Plot QPSK symbols without noise
    for i in range(num_of_users):
        qpsk_symbols = qpsk_modulate(original_data_list[i].flatten())
        plt.scatter(np.real(qpsk_symbols), np.imag(qpsk_symbols), label=f'User {i + 1} (No Noise)', marker='o')

    # Plot QPSK symbols with noise
    for i in range(num_of_users):
        qpsk_symbols = qpsk_modulate(original_data_list[i].flatten())
        noisy_qpsk_symbols = add_awgn(qpsk_symbols, snr_db)
        plt.scatter(np.real(noisy_qpsk_symbols), np.imag(noisy_qpsk_symbols), label=f'User {i + 1} (With Noise)', marker='x')

    plt.title(f'QPSK Symbols for All Users (SNR = {snr_db} dB)')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    sf = 8  # Spreading factor
    num_bytes = 8  # Number of bytes
    num_of_users = 8  # Number of users

    # Define 8 OVSF codes (for demonstration, could be changed)
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

    # Initialize composite CDMA signal
    composite_cdma_signal = np.zeros(num_bytes * 4 * sf, dtype=complex)

    original_data_list = []

    print("Starting Multi-User CDMA Transmission!")
    start_time_ns = time.time_ns()
    # Step 1: Generate data for each user and transmit
    for i in range(num_of_users):
        random_data = generate_random_data(num_bytes)
        original_data_list.append(random_data)
        #print(f"User {i + 1} Original Data:\n", random_data)

        # QPSK Transmission Chain for user i
        qpsk_cdma_noisy_signal = qpsk_tx_chain(random_data, ovsf_codes[i], sf)
        # Sum the spread signals from all users
        composite_cdma_signal += qpsk_cdma_noisy_signal
    end_time_ns = time.time_ns()
    # Calculate the elapsed time in nanoseconds
    execution_time_ns = end_time_ns - start_time_ns
    print(f"Execution time EnCoding: {execution_time_ns/1000} micro-seconds")
    # Step 2: Add AWGN noise to the composite signal
    snr_db = 15
    #print("\n composite signal w/o noise")
    #print(composite_cdma_signal)
    noisy_signal = add_awgn(composite_cdma_signal, snr_db)  # Adding noise here
    #print("\nComposite signal with noise:")
   # print(noisy_signal)

    # Step 3: Plot all QPSK symbols
   # plot_qpsk_symbols(original_data_list, noisy_signal, ovsf_codes, snr_db, num_of_users, sf)

    print("RX chain start....")
    # Step 4: Despread and decode for each user
    for i in range(num_of_users):
        print(f"\n##### DE-MODULATION FOR USER {i + 1} #####")
        decoded_bytes = qpsk_rx_chain(noisy_signal, ovsf_codes[i], sf, num_bytes)

        # Verify if the decoded data matches the original data
        data_match = np.array_equal(original_data_list[i], decoded_bytes)
        print(f"User {i + 1} Data Match:", data_match)
        if not data_match:
            print(f"Decoded Data:\n{decoded_bytes}\nOriginal Data:\n{original_data_list[i]}")
            # Print additional details for debugging, if necessary
            print("Difference between original and decoded data:")
            diff = np.abs(original_data_list[i] - decoded_bytes)
            print(diff)

        if data_match:
            print("\nSuccess")
            print("\n Decoded data ")
            print(decoded_bytes)
            print("\n Original Data")
            print(original_data_list[i])
           # print(f"Length of noisy_signal: {len(noisy_signal)}")


if __name__ == "__main__":
    main()
