import sounddevice as sd
import numpy as np
import time
import os
from scipy.signal import correlate, butter, filtfilt, windows, chirp, find_peaks
import warnings

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning) # Example: Suppress SoundDevice warnings if needed

# --- Settings ---
FS = 44100  # Sampling rate (Hz)
DURATION = 1.0 # Total recording duration (seconds) - Ensure this is long enough for MAX_DISTANCE round trip
CHIRP_TIME = 0.01 # Chirp duration (seconds)
REPEAT = 3 # Number of pings to average/combine results over
SPEED_OF_SOUND = 343 # Speed of sound in air (m/s)
MAX_DISTANCE = 15 # Maximum distance to search for echoes (meters)
LOWCUT = 18000 # Near-ultrasound bandpass low frequency (Hz)
HIGHCUT = 22000 # High frequency (Hz) (limit of phone/laptop speaker/mic)
PEAK_THRESHOLD_RATIO = 0.1 # Relative threshold for peak detection (adjust based on noise)
MIN_PEAK_DISTANCE_SEC = 0.01 # Minimum time separation between peaks (avoid multiple detections of same echo)
DATASET_DIR = "dataset" # Directory to save echo data

# --- Create chirp ---
t = np.linspace(0, CHIRP_TIME, int(FS * CHIRP_TIME), endpoint=False) # Time vector
ping = chirp(t, f0=LOWCUT, f1=HIGHCUT, t1=CHIRP_TIME, method='linear') # Generate chirp signal
ping *= windows.hann(len(ping))  # Apply Hann window to reduce spectral splatter
ping *= 0.5 # Amplitude scaling (adjust volume if needed, max 1.0)
# Ensure play_signal fits within DURATION
silence_duration = DURATION - CHIRP_TIME
if silence_duration < 0:
    print("Warning: DURATION is less than CHIRP_TIME. Adjust settings.")
    silence_duration = 0
silence = np.zeros(int(FS * silence_duration)) # Silence padding
play_signal = np.concatenate((ping, silence)) # Complete signal to play


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a bandpass filter to the data."""
    nyq = fs / 2.0 # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    # Check for valid frequency range
    if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0 or low >= high:
        print(f"Error: Invalid filter frequencies low={lowcut}, high={highcut} for fs={fs}")
        return data # Return unfiltered data on error
    try:
        b, a = butter(order, [low, high], btype='band') # Bandpass filter design
        return filtfilt(b, a, data) # Apply filter to data (zero-phase)
    except Exception as e:
        print(f"Error applying filter: {e}")
        return data # Return unfiltered data on error

def detect_echoes():
    """Plays chirp, records, processes, and returns detected distances."""
    all_distances = []
    min_peak_distance_samples = int(MIN_PEAK_DISTANCE_SEC * FS)

    # Calculate max delay samples based on MAX_DISTANCE
    max_delay_samples = int(FS * (2 * MAX_DISTANCE / SPEED_OF_SOUND))
    # Ensure recording duration is sufficient
    required_record_samples = len(ping) + max_delay_samples
    if len(play_signal) < required_record_samples:
         print(f"Warning: Recording duration ({DURATION}s) might be too short for MAX_DISTANCE ({MAX_DISTANCE}m).")
         print(f"         Need at least {required_record_samples/FS:.2f}s.")


    print(f"Starting {REPEAT} pings...")
    for i in range(REPEAT):
        try:
            # Play the chirp signal and record the response
            # Use blocking=True to ensure recording completes before processing
            rec = sd.playrec(play_signal, samplerate=FS, channels=1, blocking=True)
            # sd.wait() # Not needed if blocking=True

            if rec is None or len(rec) == 0:
                print(f"Ping {i+1}/{REPEAT}: Recording failed.")
                continue

            rec = rec.flatten() # Flatten the recorded signal

            # --- Signal Processing ---
            filtered = bandpass_filter(rec, LOWCUT, HIGHCUT, FS) # Filter the recorded signal

            # Cross-correlation
            # Use 'valid' mode if ping is shorter than filtered signal, adjust search window start
            # Using 'full' is okay if we slice correctly
            corr = correlate(filtered, ping, mode='full', method='fft') # Use FFT for speed

            # Calculate the relevant part of the correlation output
            # The peak corresponding to the direct sound path (or very close reflection)
            # should occur near the length of the recording minus the length of the ping.
            # However, finding the *start* lag is simpler with 'full' mode.
            # Lag 0 corresponds to index len(filtered) - 1 in the 'full' output.
            # We search *after* the direct signal transmission.
            start_search_index = len(ping) - 1 # Index corresponding roughly to lag 0
            end_search_index = start_search_index + max_delay_samples
            # Ensure end index is within bounds
            end_search_index = min(end_search_index, len(corr))

            if start_search_index >= end_search_index:
                 print(f"Ping {i+1}/{REPEAT}: Invalid search window indices.")
                 continue

            search_corr = corr[start_search_index:end_search_index] # Search window for echoes

            if len(search_corr) == 0:
                 print(f"Ping {i+1}/{REPEAT}: Correlation search window is empty.")
                 continue

            # --- Peak Finding ---
            max_corr_val = np.max(search_corr)
            if max_corr_val <= 0: # Avoid issues if correlation is all non-positive
                 print(f"Ping {i+1}/{REPEAT}: No positive correlation found.")
                 continue

            peak_height_threshold = PEAK_THRESHOLD_RATIO * max_corr_val

            # Find peaks in the correlation search window
            peaks_relative_indices, props = find_peaks(
                search_corr,
                height=peak_height_threshold,
                distance=min_peak_distance_samples
            )

            # Convert relative peak indices back to time delays (samples from start of search)
            # The index 'p' in peaks_relative_indices corresponds to a delay of 'p' samples *after* the start of the search window.
            distances = [(SPEED_OF_SOUND * (p / FS)) / 2.0 for p in peaks_relative_indices] # Calculate distances

            if distances:
                 print(f"Ping {i+1}/{REPEAT}: Found {len(distances)} echoes. Distances: {[f'{d:.2f}m' for d in distances]}")
                 all_distances.extend(distances) # Add distances from this ping to the list

            else:
                 print(f"Ping {i+1}/{REPEAT}: No significant echoes detected.")

        except sd.PortAudioError as pae:
            print(f"PortAudioError during ping {i+1}: {pae}")
            print("Check your audio device settings (input/output).")
            # Optionally break or return early if audio device fails
            return [] # Return empty list on critical audio error
        except Exception as e:
            print(f"Error during ping {i+1}: {e}")
            # Continue to next ping if possible, or handle error as needed

    print(f"Finished pings. Total echoes found across {REPEAT} pings: {len(all_distances)}")
    # Return the combined list of distances from all successful pings
    return all_distances

def save_echo_data(distances, label):
    """Saves the list of detected distances to a .npy file within a label-specific folder."""
    if not distances: # Don't save if the list is empty
        print("No distances to save.")
        return
    try:
        # Sanitize label for folder name
        safe_label = "".join(c if c.isalnum() else "_" for c in label).strip('_')
        if not safe_label: safe_label = "unknown"

        # Create the label-specific directory path
        target_dir = os.path.join(DATASET_DIR, safe_label)
        os.makedirs(target_dir, exist_ok=True) # Ensure directory exists

        # Use timestamp as the filename within the label folder
        timestamp = int(time.time())
        filename = f"{timestamp}.npy"
        path = os.path.join(target_dir, filename)

        np.save(path, np.array(distances, dtype=np.float32)) # Save as numpy array of floats
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Error saving data: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting Sonar Echo Collection ---")
    # Ensure sounddevice uses default devices or configure specific ones if needed
    # print("Available audio devices:", sd.query_devices())
    # sd.default.device = [input_device_index, output_device_index] # Optional: Set devices

    detected_distances = detect_echoes()

    if detected_distances:
        print(f"\nFinal combined distances: {[f'{d:.2f}m' for d in detected_distances]}")
        try:
            # Get label from user
            label_input = input("Enter label for this data (e.g., wall, person, empty): ").strip()
            if label_input:
                save_echo_data(detected_distances, label_input)
            else:
                print("No label entered. Data not saved.")
        except EOFError:
             print("\nInput interrupted. Data not saved.")
    else:
        print("\nNo echoes detected in any ping.")

    print("\n--- Collection Script Finished ---")
