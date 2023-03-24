import numpy as np
from scipy.io.wavfile import write
import random
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

# Function to generate a sine wave
def generate_sine_wave_fm(freq, duration, sample_rate):
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, False)

    # Generate modulator signal
    mod_freq = random.uniform(0.1, 10)  # Random modulator frequency between 0.1 and 10 Hz
    mod_index = random.uniform(0, 2)    # Random modulation index between 0 and 2
    modulator = mod_index * np.sin(2 * np.pi * mod_freq * t)

    # Generate carrier signal with FM modulation
    carrier_freq = freq + modulator
    sine_wave = 0.5 * np.sin(2 * np.pi * carrier_freq * t)
    return sine_wave

# Function to apply distortion
def apply_distortion(audio, amount):
    return audio * amount

# Function to apply pitch bend
def apply_pitch_bend(audio, sample_rate, bend_start, bend_end):
    length = len(audio) / sample_rate
    t = np.linspace(0, length, len(audio), False)
    t_bend = t * np.linspace(bend_start, bend_end, len(audio))

    # Interpolate the new audio with pitch bend effect
    pitch_bend_audio = np.interp(t_bend, t, audio)
    return pitch_bend_audio

# Function to apply saturation
def apply_saturation(audio, amount):
    return np.tanh(amount * audio)

num_sounds = 10
sample_rate = 44100

for i in range(num_sounds):
    freq = random.uniform(30, 200) # Random frequency between 30 and 200 Hz
    duration = random.uniform(1, 3) # Random duration between 1 and 3 seconds

    # Generate sine wave
    sine_wave = generate_sine_wave_fm(freq, duration, sample_rate)

    # Apply random pitch bend
    bend_start = random.uniform(0.5, 2)  # Random bend start between 0.5 (1 octave down) and 2 (1 octave up)
    bend_end = random.uniform(0.5, 2)    # Random bend end between 0.5 (1 octave down) and 2 (1 octave up)
    sine_wave = apply_pitch_bend(sine_wave, sample_rate, bend_start, bend_end)

    # Apply distortion
    distortion_amount = random.uniform(5, 20)  # Random distortion amount between 5 and 20
    distorted_bass = apply_distortion(sine_wave, distortion_amount)

    # Apply saturation
    saturation_amount = 5  # You can adjust this value for the desired saturation level
    saturated_bass = apply_saturation(distorted_bass, saturation_amount)

    # Convert NumPy array to PyDub AudioSegment for compression
    saturated_bass_int16 = (saturated_bass * 32767).astype(np.int16)
    saturated_audio_segment = AudioSegment(
        saturated_bass_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=saturated_bass_int16.dtype.itemsize,
        channels=1
    )

    # Apply compression
    compressed_audio_segment = compress_dynamic_range(saturated_audio_segment, threshold=-20, ratio=6.0, attack=5, release=100)

    # Convert PyDub AudioSegment back to NumPy array
    compressed_bass = np.frombuffer(compressed_audio_segment.raw_data, dtype=np.int16).astype(np.float32) / 32767

    # Check if audio data is valid
    epsilon = 1e-8
    if not np.isnan(compressed_bass).any() and not np.isinf(compressed_bass).any() and len(compressed_bass) > 0:
        # Normalize the audio
        compressed_bass = compressed_bass / (np.max(np.abs(compressed_bass)) + epsilon)

        # Save the sound as a WAV file
        write(f"distorted_bass_{i + 1}.wav", sample_rate, (compressed_bass * 32767).astype(np.int16))
    else:
        print(f"Skipping distorted_bass_{i + 1} due to invalid audio data")
