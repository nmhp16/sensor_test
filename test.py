import sounddevice as sd
import numpy as np
from scipy.signal import correlate, butter, filtfilt, windows, chirp, find_peaks
import matplotlib.pyplot as plt

# --- Config ---
duration = 1.0       # seconds (recording duration)
fs = 44100           # Sampling rate
chirp_time = 0.01    # Chirp duration
repeat = 5           # Number of pings to average
speed_of_sound = 343 # m/s
max_distance = 5     # meters (max echo range to search)
lowcut = 18000       # Near-ultrasound bandpass low freq
highcut = 22000      # High freq (limit of phone/laptop speaker/mic)
peak_threshold = 0.5 # Minimum correlation peak to be considered valid

# --- Generate chirp signal ---
t = np.linspace(0, chirp_time, int(fs * chirp_time), False)
ping = chirp(t, f0=lowcut, f1=highcut, t1=chirp_time, method='linear')
ping *= windows.hann(len(ping))  # Apply Hann window
ping *= 0.5  # Scale volume

# Append silence after chirp to match total duration
silence = np.zeros(int(fs * (duration - chirp_time)))
play_signal = np.concatenate((ping, silence))

# --- Bandpass filter ---
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# --- Estimate distance over multiple pings ---
all_detected_distances = []
for i in range(repeat):
    print(f"Ping {i+1}/{repeat}...")
    recording = sd.playrec(play_signal, samplerate=fs, channels=1)
    sd.wait()
    recording = recording.flatten()

    # Filter the recorded signal
    filtered = bandpass_filter(recording, lowcut, highcut, fs)

    # Cross-correlation
    corr = correlate(filtered, ping, mode='full')
    center = len(corr) // 2
    max_delay_samples = int(fs * (2 * max_distance / speed_of_sound))
    search_corr = corr[center:center + max_delay_samples]

    # Detect peaks (multiple echoes)
    min_distance_samples = int(0.005 * fs)  # Prevent double peaks for close echoes
    peak_indices, properties = find_peaks(search_corr, height=peak_threshold * np.max(search_corr), distance=min_distance_samples)

    # Calculate distance for each detected peak
    detected_distances = []
    for peak in peak_indices:
        delay_sec = peak / fs
        distance = (speed_of_sound * delay_sec) / 2
        detected_distances.append(distance)
        print(f" -> Echo at {distance:.3f} meters")

    print(f"  Detected {len(detected_distances)} object(s) in this ping\n")
    all_detected_distances.extend(detected_distances)


# --- Final average ---
if all_detected_distances:
    average_distance = np.mean(all_detected_distances)
    print(f"\n Average distance across all echoes: {average_distance:.3f} meters")
else:
    print("\n No valid echoes detected.")

# --- Plot last correlation with detected peaks ---
plt.plot(search_corr, label='Cross-Correlation')
plt.plot(peak_indices, search_corr[peak_indices], "rx", label='Detected Peaks')
plt.title("Cross-Correlation (Last Ping)")
plt.xlabel("Lag (samples)")
plt.ylabel("Correlation")
plt.legend()
plt.grid()
plt.show()
