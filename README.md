# Distorted Bass Generator

Distorted Bass Generator is a Python script that generates random bass sounds with heavy distortion, pitch bending, and FM modulation. The script creates a specified number of audio files in WAV format, each containing a unique bass sound.

## Features

- Randomly generates bass frequencies between 30 and 200 Hz.
- Applies random pitch bends for dramatic effects.
- FM modulation with random frequency and modulation index.
- Distorts the bass sounds with a random distortion amount.
- Applies saturation to enhance the overall character.
- Compresses the dynamic range of the audio.

## Requirements

- Python 3.6+
- NumPy
- SciPy
- PyDub

To install the required packages, run:

```bash
pip install numpy scipy pydub
```

## Usage

1. Download or clone the repository.

2. Install dependencies:

```bash
pip install numpy scipy pydub
```

3. Run the script:

```bash
python distorted_bass_generator.py
```

4. The script will generate WAV files named `distorted_bass_1.wav`, `distorted_bass_2.wav`, and so on, in the same directory as the script.

## Customization

You can adjust the following parameters in the script to customize the generated bass sounds:

- `num_sounds`: the number of sounds to generate
- `sample_rate`: the sample rate of the generated sounds
- `freq`: the frequency range of the generated sounds
- `duration`: the duration range of the generated sounds
- `attack`: the attack time range of the ADSR envelope
- `decay`: the decay time range of the ADSR envelope
- `sustain`: the sustain level range of the ADSR envelope
- `release`: the release time range of the ADSR envelope
- `distortion_amount`: the distortion amount range of the distortion effect
- `saturation_amount`: the saturation amount of the saturation effect
- `grain_duration`: the duration range of the grains in the granulizer effect
- `overlap`: the overlap range of the grains in the granulizer effect
- `num_grains`: the number of grains in the granulizer effect
- `noise_type`: the type of noise used in the noise effect
- `snr`: the signal-to-noise ratio of the noise effect
- `delay_time`: the delay time range of the delay effect
- `feedback`: the feedback amount of the delay effect
- `mix`: the mix amount of the delay effect
- `pan`: the panning amount of the stereo width effect
- `mod_freq`: the modulation frequency of the volume modulation effect
- `mod_depth`: the modulation depth of the volume modulation effect
- `num_voices`: the number of voices in the chorus effect
- `delay_range`: the delay time range of each voice in the chorus effect
- `depth_range`: the depth range of each voice in the chorus effect
- `cutoff_freq`: the cutoff frequency range of the low-pass filter effect
- `decay`: the decay time of the reverb effect
- `room_scale`: the room scale of the reverb effect
- `threshold`: the threshold of the compression effect
- `ratio`: the compression ratio of the compression effect
- `attack`: the attack time of the compression effect
- `release`: the release time of the compression effect

Adjust these values in the script to create different types of bass sounds.

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
