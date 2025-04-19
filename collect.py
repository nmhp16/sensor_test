import sounddevice as sd
import numpy as np
import time
import os
from scipy.signal import butter, filtfilt, windows, chirp, stft
import warnings
import matplotlib.pyplot as plt
import json
import sys

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
MIN_VALID_DISTANCE = 0.5  # Minimum distance (in meters) to consider as valid echo
DEBUG_VISUALIZATION = True # <--- CHANGE THIS TO True
CALIBRATION_FILE = "calibration.json"

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

def analyze_spectrogram():
    """Plays chirp, records, and displays the spectrogram of the received signal."""
    print(f"Starting {REPEAT} pings for spectrogram analysis...")
    
    # STFT parameters (tune these)
    nperseg = 256  # Length of each segment (window size)
    noverlap = nperseg // 2 # Amount of overlap between segments
    nfft = 512 # Length of FFT used, if different from nperseg

    for i in range(REPEAT):
        try:
            rec = sd.playrec(play_signal, samplerate=FS, channels=1, blocking=True)

            if rec is None or len(rec) == 0:
                print(f"Ping {i+1}/{REPEAT}: Recording failed.")
                continue

            rec = rec.flatten()

            # --- Signal Processing (Optional but recommended) ---
            filtered_rec = bandpass_filter(rec, LOWCUT, HIGHCUT, FS)

            # --- Spectrogram Calculation ---
            f, t_spec, Zxx = stft(filtered_rec, fs=FS, window='hann',
                                  nperseg=nperseg, noverlap=noverlap, nfft=nfft)

            # --- Visualization ---
            plt.figure(figsize=(12, 6))
            # Use logarithmic scale for magnitude (dB) for better visibility
            plt.pcolormesh(t_spec, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
            # Limit frequency axis to area of interest if desired
            plt.ylim(LOWCUT * 0.8, HIGHCUT * 1.2)
            plt.title(f'Spectrogram - Ping {i+1}')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar(label='Magnitude')
            # plt.show() 
            # print(f"Ping {i+1}/{REPEAT}: Spectrogram displayed.")
            
            # --- Return data for saving ---
            # For simplicity, let's just return the first ping's data for now
            # A better approach would average or concatenate Zxx across pings
            if i == 0: 
                first_ping_data = {'Zxx': Zxx, 't': t_spec, 'f': f}

        except sd.PortAudioError as pae:
            print(f"PortAudioError during ping {i+1}: {pae}")
            break # Stop if audio device fails
        except Exception as e:
            print(f"Error during ping {i+1}: {e}")

    print("Finished pings.")
    # This function primarily visualizes; returning data would require processing Zxx
    # Return the collected data (e.g., from the first ping)
    # Modify this return logic based on how you want to handle multiple pings
    return first_ping_data if 'first_ping_data' in locals() else None

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting Sonar Spectrogram Analysis & Data Collection ---")
    
    spectrogram_data = analyze_spectrogram()

    if spectrogram_data:
        try:
            label = input(f"Enter label for this data (e.g., wall, person, empty) or leave blank to skip saving: ").strip()
            if label:
                # Create dataset directory if it doesn't exist
                if not os.path.exists(DATASET_DIR):
                    os.makedirs(DATASET_DIR)
                
                # Create label subdirectory if it doesn't exist
                label_dir = os.path.join(DATASET_DIR, label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)

                # Generate unique filename (e.g., using timestamp)
                timestamp = int(time.time())
                filename = os.path.join(label_dir, f"{timestamp}.npz")
                
                # Save the spectrogram data and axes
                np.savez(filename, Zxx=spectrogram_data['Zxx'], t=spectrogram_data['t'], f=spectrogram_data['f'])
                print(f"Saved data for label '{label}' to: {filename}")
            else:
                print("No label entered. Data not saved.")
        except EOFError:
             print("\nInput interrupted. Data not saved.")
    else:
        print("No spectrogram data collected.")

    print("\n--- Collection Script Finished ---")
