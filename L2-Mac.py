import numpy as np
import matplotlib.pyplot as plt

# Parameters for two-user QPSK CDMA system
num_symbols = 1000
sf = 8  # Spreading factor (orthogonal codes)
snr_db = 10  # Signal-to-Noise Ratio in dB

# Generate random QPSK symbols for both users
qpsk_user1 = (2 * (np.random.randint(0, 2, num_symbols) - 0.5)) + 1j * (2 * (np.random.randint(0, 2, num_symbols) - 0.5))
qpsk_user2 = (2 * (np.random.randint(0, 2, num_symbols) - 0.5)) + 1j * (2 * (np.random.randint(0, 2, num_symbols) - 0.5))

# Normalize the power of QPSK symbols
qpsk_user1 /= np.sqrt(2)
qpsk_user2 /= np.sqrt(2)

# Generate orthogonal spreading codes for both users
code_user1 = np.random.choice([-1, 1], sf)
code_user2 = np.random.choice([-1, 1], sf)

# Spread the QPSK symbols
spread_user1 = np.repeat(qpsk_user1, sf) * np.tile(code_user1, num_symbols)
spread_user2 = np.repeat(qpsk_user2, sf) * np.tile(code_user2, num_symbols)
print("spread_user1\n")
print(spread_user1)

# Superpose the signals and add AWGN noise
combined_signal = spread_user1 + spread_user2
noise_power = 10 ** (-snr_db / 10)
noise = np.sqrt(noise_power / 2) * (np.random.randn(len(combined_signal)) + 1j * np.random.randn(len(combined_signal)))
received_signal = combined_signal + noise

# Despread the signals for both users
despread_user1 = np.dot(received_signal.reshape(-1, sf), code_user1) / sf
despread_user2 = np.dot(received_signal.reshape(-1, sf), code_user2) / sf
print("\ndespread_user1")
print(despread_user1)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Combined Signal (Before Despreading)
axs[0].scatter(received_signal.real, received_signal.imag, color='blue', alpha=0.5, label='Combined Signal')
axs[0].set_title("Combined Received Signal (Before Despreading)")
axs[0].set_xlabel("In-Phase")
axs[0].set_ylabel("Quadrature")
axs[0].grid()
axs[0].legend()

# Despread Signal for User 1
axs[1].scatter(despread_user1.real, despread_user1.imag, color='red', alpha=0.5, label='User 1 (QPSK)')
axs[1].set_title("Constellation for User 1 (After Despreading)")
axs[1].set_xlabel("In-Phase")
axs[1].set_ylabel("Quadrature")
axs[1].grid()
axs[1].legend()

# Despread Signal for User 2
axs[2].scatter(despread_user2.real, despread_user2.imag, color='green', alpha=0.5, label='User 2 (QPSK)')
axs[2].set_title("Constellation for User 2 (After Despreading)")
axs[2].set_xlabel("In-Phase")
axs[2].set_ylabel("Quadrature")
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()
