import numpy as np
import matplotlib.pyplot as plt

def generate_noise(number_of_samples: int) -> np.ndarray:
    return np.random.randn(number_of_samples)

def generate_x_n(number_of_samples: int) -> np.ndarray:
    i_n = generate_noise(number_of_samples)
    x_n = np.zeros(number_of_samples)

    for index in range(number_of_samples):
        if index - 1 < 0:
            x_n[index] = (2 / 3) * i_n[index]
        else:
            x_n[index] = (2 / 3) * i_n[index] - (1 / 3) * i_n[index - 1] + (x_n[index - 1] / 3)
    return x_n

def calculate_theoretical_auto_correlation(max_lag: int, number_of_samples: int) -> np.ndarray:
    frequencies = np.linspace(0, 2 * np.pi, number_of_samples, endpoint=False)
    S_x = (5 - 4 * np.cos(frequencies)) / (10 - 6 * np.cos(frequencies))

    auto_correlation = np.fft.ifft(S_x).real  # Use real part since Rx(m) is real
    symmetric_auto_correlation = np.concatenate((auto_correlation[-max_lag:], auto_correlation[:max_lag + 1]))
    return symmetric_auto_correlation

def plot_empirical_and_theoretical(
        theoretical_auto_correlation: np.ndarray,
        empirical_auto_correlation: np.ndarray,
        lags: np.ndarray,
        display: bool = True,
        file_name: str = "Theoretical_vs_Empirical_Auto_correlation.png"
):
    plt.figure(figsize=(10, 6))
    plt.plot(lags, theoretical_auto_correlation, 'b-', label='Theoretical')
    plt.plot(lags, empirical_auto_correlation, 'r--o', label='Empirical')
    plt.xlabel('Lag')
    plt.ylabel('Auto-correlation')
    plt.title('Theoretical vs Empirical Auto-correlation')
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name)
    if display:
        plt.show()
    else:
        plt.close()

def main():
    NUMBER_OF_SAMPLES = 10 ** 4
    NUM_TRIALS = 500
    MAX_LAG = 20

    # Create symmetric range of lags
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)

    theoretical_correlations = calculate_theoretical_auto_correlation(
        max_lag=MAX_LAG,
        number_of_samples=NUMBER_OF_SAMPLES
    )

    empirical_correlations = np.zeros(len(lags))
    CALCULATION_METHOD = 1 # Set to One for one point calculation of Rx(m)
    for trial in range(NUM_TRIALS):
        x_n = generate_x_n(NUMBER_OF_SAMPLES)
        for i, lag in enumerate(lags):
            if CALCULATION_METHOD == 0:
                if lag >= 0:
                    empirical_correlations[i] += np.mean(x_n[lag:] * x_n[:-lag] if lag > 0 else x_n * x_n)
                else:
                     empirical_correlations[i] += np.mean(x_n[:lag] * x_n[-lag:])
            else:
                empirical_correlations[i] += x_n[1500+lag] * x_n[1500]

    empirical_correlations /= NUM_TRIALS

    plot_empirical_and_theoretical(theoretical_correlations, empirical_correlations, lags)

    print("\nNumerical Results:")
    print("Lag".ljust(8) + "Theoretical".ljust(15) + "Empirical".ljust(15))
    print("-" * 38)
    for i, lag in enumerate(lags[17:24]):
        print(f"{lag:3d}".ljust(8) +
              f"{theoretical_correlations[i+17]:8.4f}".ljust(15) +
              f"{empirical_correlations[i+17]:8.4f}".ljust(15))

if __name__ == '__main__':
    main()
