import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import butter, lfilter, lfilter_zi, convolve
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

    # Apply fade-in effect
    fade_in_duration = 0.05  # Adjust this value for the desired fade-in length
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_in_window = np.linspace(0, 1, fade_in_samples)
    sine_wave[:fade_in_samples] *= fade_in_window
    
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

# Function to apply ADSR envelope
def apply_adsr_envelope(audio, attack, decay, sustain, release, sample_rate):
    num_samples = len(audio)
    envelope = np.zeros(num_samples)

    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)

    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay
    envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain, decay_samples)

    # Sustain (until the release phase begins)
    sustain_start = attack_samples + decay_samples
    sustain_end = num_samples - release_samples
    envelope[sustain_start:sustain_end] = sustain

    # Release
    envelope[sustain_end:] = np.linspace(sustain, 0, release_samples)

    return audio * envelope

# Add Reverb
def add_reverb(audio, sr, decay=1.5, room_scale=0.8):
    impulse_response_length = int(sr * room_scale)
    impulse_response = np.random.randn(impulse_response_length)
    impulse_response *= np.exp(-decay * np.arange(impulse_response_length) / impulse_response_length)
    reverb_audio = convolve(audio, impulse_response, mode='same')
    return audio + reverb_audio

# Add Chorus
def add_chorus(audio, sr, num_voices=3, delay_range=(25, 50), depth_range=(0.1, 0.3)):
    chorus_audio = np.zeros_like(audio)
    for _ in range(num_voices):
        delay = np.random.uniform(delay_range[0], delay_range[1])
        depth = np.random.uniform(depth_range[0], depth_range[1])
        delay_samples = int(delay * sr / 1000)
        modulated_audio = np.roll(audio, delay_samples)
        chorus_audio += depth * modulated_audio
    return audio + chorus_audio

# Apply Filtering (Low-pass)
def low_pass_filter(audio, sr, cutoff_freq, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    zi = lfilter_zi(b, a)
    filtered_audio, _ = lfilter(b, a, audio, zi=zi * audio[0])
    return filtered_audio

# Add Delay
def add_delay(audio, sr, delay_time=0.5, feedback=0.5, mix=0.5):
    delay_samples = int(delay_time * sr)
    delayed_audio = np.zeros_like(audio)
    buffer_audio = audio.copy()

    for _ in range(int(sr * delay_time)):
        delayed_audio += buffer_audio
        buffer_audio = np.roll(buffer_audio, delay_samples)
        buffer_audio *= feedback

    return audio * (1 - mix) + delayed_audio * mix

# Add Stereo Width (simple panning method)
def add_stereo_width_mono(audio, pan):
    left = audio * (1 - pan)
    right = audio * pan
    stereo_audio = np.vstack((left, right))
    return stereo_audio

# Modulate Volume
def modulate_volume(audio, sr, mod_freq=5, mod_depth=0.5):
    modulator = (1 - mod_depth) + mod_depth * np.sin(2 * np.pi * mod_freq * np.arange(len(audio)) / sr)
    return audio * modulator

# Layer Multiple Sounds
def layer_sounds(sounds, amplitudes=None):
    if amplitudes is None:
        amplitudes = [1] * len(sounds)
    return np.sum([amp * sound for amp, sound in zip(amplitudes, sounds)], axis=0)

def pink_noise(N):
    white = np.random.normal(0, 1, N)
    freqs = np.fft.rfftfreq(N)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1 / np.sqrt(freqs[1:])
    pink = irfft(scaling * rfft(white), n=N)
    return pink / np.max(np.abs(pink))

def brown_noise(N):
    white = np.random.normal(0, 1, N)
    freqs = np.fft.rfftfreq(N)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1 / freqs[1:]
    brown = irfft(scaling * rfft(white), n=N)
    return brown / np.max(np.abs(brown))

def add_noise(audio, noise_type='white', snr=20):
    if noise_type == 'white':
        generated_noise = np.random.normal(0, 1, len(audio))
    elif noise_type == 'pink':
        generated_noise = pink_noise(len(audio))
    elif noise_type == 'brown':
        generated_noise = brown_noise(len(audio))
    else:
        raise ValueError("Invalid noise type. Supported types: 'white', 'pink', 'brown'")

    signal_power = np.mean(np.abs(audio) ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    generated_noise = np.sqrt(noise_power) * generated_noise
    return audio + generated_noise

num_sounds = 100
sample_rate = 44100

for i in range(num_sounds):
    freq = random.uniform(20, 80) # Random frequency between 30 and 200 Hz
    duration = random.uniform(1, 6) # Random duration between 1 and 3 seconds

    # Generate sine wave
    sine_wave = generate_sine_wave_fm(freq, duration, sample_rate)

    # Apply random ADSR envelope
    attack = random.uniform(0.01, 0.5) # Random attack between 0.01 and 0.5 seconds
    decay = random.uniform(0.1, 0.5) # Random decay between 0.1 and 0.5 seconds
    sustain = random.uniform(0.1, 1) # Random sustain level between 0.1 and 1
    release = random.uniform(0.1, 1) # Random release between 0.1 and 1 seconds
    sine_wave = apply_adsr_envelope(sine_wave, attack, decay, sustain, release, sample_rate)

    # Randomly apply effects
    if random.random() < 0.5:
        sine_wave = add_reverb(sine_wave, sample_rate)

    if random.random() < 0.5:
        sine_wave = add_chorus(sine_wave, sample_rate)

    if random.random() < 0.5:
        sine_wave = low_pass_filter(sine_wave, sample_rate, cutoff_freq=random.uniform(100, 500))

    if random.random() < 0.5:
        sine_wave = add_delay(sine_wave, sample_rate)

    if random.random() < 0.5:
        sine_wave = modulate_volume(sine_wave, sample_rate)

    if random.random() < 0.5:
        sine_wave = add_noise(sine_wave, noise_type=random.choice(['white', 'pink', 'brown']))

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
