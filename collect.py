# collect.py
import sounddevice as sd
import numpy as np
import time
import os
from scipy.signal import correlate, butter, filtfilt, windows, chirp, find_peaks

# --- Settings ---
fs = 44100  # Sampling rate (Hz)
duration = 1.0 # Total recording duration (seconds)
chirp_time = 0.01 # Chirp duration (seconds)
repeat = 3 # Number of pings to average
speed_of_sound = 343 # Speed of sound in air (m/s)
max_distance = 5 # Maximum distance to search for echoes (meters)
lowcut = 18000 # Near-ultrasound bandpass low frequency (Hz)
highcut = 22000 # High frequency (Hz) (limit of phone/laptop speaker/mic)
peak_threshold = 0.5 # Minimum correlation peak relative to max peak
dataset_dir = "dataset" # Directory to save echo data

# --- Create chirp ---
t = np.linspace(0, chirp_time, int(fs * chirp_time), False) # Time vector
ping = chirp(t, f0=lowcut, f1=highcut, t1=chirp_time, method='linear') # Generate chirp signal
ping *= windows.hann(len(ping))  # Apply Hann window
ping *= 0.5 # Amplitude scaling
silence = np.zeros(int(fs * (duration - chirp_time))) # Silence padding
play_signal = np.concatenate((ping, silence)) # Complete signal to play


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = fs / 2 # Nyquist frequency
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band') # Bandpass filter design
    return filtfilt(b, a, data) # Apply filter to data

def detect_echoes():
    all_distances = []
    
    # Play the chirp signal and record the response
    for i in range(repeat):
        print(f"Ping {i+1}/{repeat}...")
        rec = sd.playrec(play_signal, samplerate=fs, channels=1) # Play and record
        sd.wait() # Wait for the sound to finish playing
        rec = rec.flatten() # Flatten the recorded signal

        filtered = bandpass_filter(rec, lowcut, highcut, fs) # Filter the recorded signal
        corr = correlate(filtered, ping, mode='full') # Cross-correlation
        center = len(corr) // 2 # Center of the correlation
        max_delay_samples = int(fs * (2 * max_distance / speed_of_sound)) # Maximum delay in samples
        search = corr[center:center + max_delay_samples] # Search window for echoes

        min_distance_samples = int(0.005 * fs) # Minimum distance in samples
        
        # Find peaks in the correlation
        peaks, _ = find_peaks(search, height=peak_threshold * np.max(search), distance=min_distance_samples)

        distances = [(speed_of_sound * (p / fs)) / 2 for p in peaks] # Calculate distances from peaks
        all_distances.extend(distances) # Store all distances

    return all_distances

def save_echo_data(distances, label):
    os.makedirs(dataset_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"{label}_{timestamp}.npy"
    path = os.path.join(dataset_dir, filename)
    np.save(path, np.array(distances))
    print(f"Saved: {path}")

if __name__ == "__main__":
    distances = detect_echoes()
    if distances:
        print(f"Detected distances: {distances}")
        label = input("Enter label (e.g., wall, human, empty): ")
        save_echo_data(distances, label)
    else:
        print("No echoes detected.")
