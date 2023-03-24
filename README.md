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

- Python 3.x
- NumPy
- SciPy
- PyDub

To install the required packages, run:

```bash
pip install numpy scipy pydub
```

## Usage

1. Download or clone the repository.

2. Modify the `num_sounds` variable in the script to specify the number of bass sounds you want to generate:

```python
num_sounds = 10  # Adjust this value to generate more or fewer sounds
```

3. Run the script:

```bash
python distorted_bass_generator.py
```

4. The script will generate WAV files named `distorted_bass_1.wav`, `distorted_bass_2.wav`, and so on, in the same directory as the script.

## Customization

You can adjust the following parameters in the script to customize the generated bass sounds:

- `freq`: The frequency range of the bass sounds (default: 30-200 Hz)
- `duration`: The duration range of the bass sounds (default: 1-3 seconds)
- `bend_start` and `bend_end`: The pitch bend range (default: 0.5-2)
- `distortion_amount`: The distortion amount range (default: 5-20)
- `saturation_amount`: The saturation level (default: 5)
- `mod_freq`: The modulator frequency range for FM modulation (default: 0.1-10 Hz)
- `mod_index`: The modulation index range for FM modulation (default: 0-2)

Adjust these values in the script to create different types of bass sounds.

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
