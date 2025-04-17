import sounddevice as sd
import numpy as np
from scipy.signal import correlate, butter, filtfilt, windows, chirp, find_peaks
import matplotlib.pyplot as plt
import time # Added for potential delays if needed

# --- Configuration ---
FS = 44100           # Sampling rate (Hz)
CHIRP_DURATION = 0.01 # Chirp duration (seconds)
RECORD_DURATION = 1.0 # Total recording duration (seconds)
REPEAT_COUNT = 5      # Number of pings to average
SPEED_OF_SOUND = 343 # m/s
MAX_DISTANCE = 5     # meters (max echo range to search)
LOW_CUT = 18000      # Near-ultrasound bandpass low freq (Hz)
HIGH_CUT = 22000     # High freq (Hz) (limit of phone/laptop speaker/mic)
PEAK_THRESHOLD_RATIO = 0.5 # Minimum correlation peak relative to max peak
MIN_PEAK_DISTANCE_SEC = 0.005 # Minimum time between detectable peaks (seconds)
CHIRP_AMPLITUDE = 0.5 # Amplitude scaling for the chirp

# --- Signal Generation ---
def generate_chirp_signal(fs, chirp_duration, total_duration, low_freq, high_freq, amplitude):
    """Generates the chirp signal padded with silence."""
    t = np.linspace(0, chirp_duration, int(fs * chirp_duration), endpoint=False)
    ping = chirp(t, f0=low_freq, f1=high_freq, t1=chirp_duration, method='linear')
    ping *= windows.hann(len(ping))  # Apply Hann window
    ping *= amplitude

    # Append silence
    num_silence_samples = int(fs * (total_duration - chirp_duration))
    if num_silence_samples < 0:
        num_silence_samples = 0 # Ensure non-negative silence samples
        print("Warning: Chirp duration is longer than total duration. No silence appended.")

    silence = np.zeros(num_silence_samples)
    play_signal = np.concatenate((ping, silence))
    return play_signal, ping

# --- Signal Processing ---
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a bandpass filter to the data."""
    nyq = 0.5 * fs # Nyquist frequency
    low = lowcut / nyq # Normalize frequency
    high = highcut / nyq # Normalize frequency
    
    # Ensure cutoff frequencies are valid
    low = max(0.01, low) # Avoid zero or negative frequencies
    high = min(0.99, high) # Avoid frequencies too close to Nyquist
    
    if low >= high:
        raise ValueError("Low cut frequency must be lower than high cut frequency.")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- Echo Detection ---
def find_echo_distances(recording, original_ping, fs, lowcut, highcut, max_distance, speed_of_sound, peak_threshold_ratio, min_peak_dist_sec):
    """Filters recording, finds correlation peaks, and calculates distances."""
    filtered_recording = bandpass_filter(recording, lowcut, highcut, fs)

    # Cross-correlation
    correlation = correlate(filtered_recording, original_ping, mode='full')
    # Center the correlation result around the zero lag point
    correlation = correlation[len(original_ping)-1:] # Keep only non-negative lags

    # Define search range based on max_distance
    max_delay_samples = int(fs * (2 * max_distance / speed_of_sound))
    search_end_index = min(max_delay_samples, len(correlation)) # Ensure index is within bounds
    correlation_search_range = correlation[:search_end_index]

    if len(correlation_search_range) == 0:
        return [], correlation_search_range, [] # No valid search range

    # Detect peaks
    min_peak_distance_samples = int(min_peak_dist_sec * fs)
    max_corr_value = np.max(correlation_search_range)
    if max_corr_value <= 0: # Avoid issues if max correlation is zero or negative
        return [], correlation_search_range, []

    peak_indices, _ = find_peaks(
        correlation_search_range,
        height=peak_threshold_ratio * max_corr_value,
        distance=min_peak_distance_samples
    )

    # Calculate distances
    distances = []
    for peak_index in peak_indices:
        delay_sec = peak_index / fs
        distance = (speed_of_sound * delay_sec) / 2
        distances.append(distance)

    return distances, correlation_search_range, peak_indices

# --- Plotting ---
def plot_results(correlation_data, peak_indices, title="Cross-Correlation"):
    """Plots the cross-correlation and detected peaks."""
    plt.figure() # Create a new figure for each plot
    plt.plot(correlation_data, label='Cross-Correlation')
    if len(peak_indices) > 0:
        plt.plot(peak_indices, correlation_data[peak_indices], "rx", label='Detected Peaks')
    plt.title(title)
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---
def main():
    """Main function to run the distance estimation."""
    play_signal, ping_signal = generate_chirp_signal(
        FS, CHIRP_DURATION, RECORD_DURATION, LOW_CUT, HIGH_CUT, CHIRP_AMPLITUDE
    )

    all_detected_distances = []
    last_correlation = None
    last_peak_indices = []

    print("Starting pings...")
    for i in range(REPEAT_COUNT):
        print(f"  Ping {i+1}/{REPEAT_COUNT}...")
        try:
            # Play and record simultaneously
            recording = sd.playrec(play_signal, samplerate=FS, channels=1, blocking=True)
            # sd.wait() # blocking=True makes sd.wait() redundant
            recording = recording.flatten() # Ensure it's a 1D array

            distances, correlation, peak_indices = find_echo_distances(
                recording, ping_signal, FS, LOW_CUT, HIGH_CUT, MAX_DISTANCE,
                SPEED_OF_SOUND, PEAK_THRESHOLD_RATIO, MIN_PEAK_DISTANCE_SEC
            )

            if distances:
                print(f"    Detected {len(distances)} echo(s): ", end="")
                print(", ".join([f"{d:.3f}m" for d in distances]))
                all_detected_distances.extend(distances)
                last_correlation = correlation # Store last correlation for plotting
                last_peak_indices = peak_indices
            else:
                print("    No distinct echoes detected in this ping.")
                # Optionally store the correlation even if no peaks are found for debugging
                last_correlation = correlation
                last_peak_indices = []


        except Exception as e:
            print(f"    Error during ping {i+1}: {e}")
            # Optionally add a small delay before retrying
            # time.sleep(0.1)

    print("\nPinging complete.")

    # --- Final Analysis ---
    if all_detected_distances:
        average_distance = np.mean(all_detected_distances)
        median_distance = np.median(all_detected_distances)
        std_dev_distance = np.std(all_detected_distances)
        print(f"\n--- Results ---")
        print(f"Total echoes detected across {REPEAT_COUNT} pings: {len(all_detected_distances)}")
        print(f"Average distance: {average_distance:.3f} meters")
        print(f"Median distance: {median_distance:.3f} meters")
        print(f"Standard deviation: {std_dev_distance:.3f} meters")
    else:
        print("\nNo valid echoes detected across all pings.")

    # --- Plot Last Correlation ---
    if last_correlation is not None:
         plot_results(last_correlation, last_peak_indices, title=f"Cross-Correlation (Last Ping - {REPEAT_COUNT})")
    else:
        print("No correlation data to plot.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # Specific error handling for sounddevice issues
        if "PortAudioError" in str(e):
             print("PortAudioError: Could not open audio stream. Ensure audio devices are connected and configured correctly.")
             print("You might need to install PortAudio or check device permissions.")
