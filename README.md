# Voice-to-Text Transcription 🗣️➡️📝

This Python script uses a pre-trained model from Hugging Face's `transformers` library to transcribe speech from an audio file (e.g., .wav, .mp3) into text. It leverages the power of automatic speech recognition (ASR) to turn spoken words into written content with high accuracy. 

## Requirements 📋

Before you can run the script, make sure you have the following installed:

- Python 3.7 or later
- PyTorch (with CUDA support for GPU acceleration)
- Hugging Face `transformers` library
- `argparse` (included in the standard Python library)

### To install the necessary dependencies:

You can install the required libraries using `pip`. It's recommended to create a virtual environment before installing the dependencies.

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the required libraries:

   ```bash
   pip install torch transformers
   ```

## File Structure 📂

```
.
├── voice2text.py         # The Python script for voice-to-text transcription
└── README.md             # This README file
```

## How to Run the Script ⚙️

### Basic Usage 🎬

To run the script, use the following command in your terminal:

```bash
python voice2text.py input_audio.wav -o output.txt --model distil-whisper/distil-large-v3
```

#### Arguments:
- `input_audio` (required): Path to the input audio file (e.g., `.wav`, `.mp3`).
- `-o` or `--output` (optional): Path to the output text file where the transcription will be saved. If not provided, the output file will default to `<input_audio_name>.txt`.
- `--model` (optional): The pre-trained model to be used for transcription (default is `distil-whisper/distil-large-v3`).

#### Example:

```bash
python voice2text.py my_audio_file.mp3
```

This will transcribe the audio file `my_audio_file.mp3` and save the transcription in `my_audio_file.txt`.

### Example with custom output file:

```bash
python voice2text.py my_audio_file.wav -o transcription.txt
```

This will save the transcription in `transcription.txt` instead of the default file name.

## How the Code Works 🧠

1. **Model Loading** 🚀: The script loads a pre-trained automatic speech recognition model (default: `distil-whisper/distil-large-v3`) from Hugging Face's `transformers` library.
   
2. **Audio File Processing 🎧**: The script processes the input audio file using the model to recognize and transcribe the speech. The `pipeline` from `transformers` is used for automatic speech recognition (ASR).

3. **Device Selection 🖥️**: The script checks if CUDA (GPU support) is available for faster processing. If not, it will fall back to the CPU.

4. **Transcription** ✍️: The recognized speech is converted into text and saved to the specified output file. If no output file is specified, the script will save the result in a `.txt` file with the same name as the input audio file.

5. **Saving the Transcript 💾**: After transcription, the output text is saved into the provided output file.

### Example Workflow 📂:

1. Run the script with the audio file:
   
   ```bash
   python voice2text.py example_audio.mp3
   ```

2. The script will process the audio and create a text file (`example_audio.txt`) with the transcription.

3. Check the output file for your transcribed text!


## Alternative Models 🧑‍🏫

You can use different pre-trained models from Hugging Face's `transformers` library for transcription, depending on your needs. Here are a few alternatives to `distil-whisper/distil-large-v3`:

1. **`whisper-large-v3`** 📢  
   A larger and more powerful variant of the Whisper model. It's suitable for higher accuracy, especially for more complex audio.  
   ```bash
   --model openai/whisper-large-v3
   ```

2. **`whisper-medium`** 🎧  
   A medium-sized Whisper model that offers a good balance between speed and accuracy. It may perform faster on some machines, with a slight trade-off in accuracy.  
   ```bash
   --model openai/whisper-medium
   ```

3. **`facebook/wav2vec2-large-960h`** 🗣️  
   A popular ASR model from Facebook that is highly accurate, especially for English audio, but may require more resources.  
   ```bash
   --model facebook/wav2vec2-large-960h
   ```

4. **`google/voice-search`** 🎙️  
   This model is fine-tuned for voice search and conversational audio, offering good transcription for casual speech or commands.  
   ```bash
   --model google/voice-search
   ```

You can replace the model argument in the command with any of these alternatives, depending on your specific use case.

## Troubleshooting ❓

- **CUDA errors**: If you encounter errors related to CUDA, ensure you have a compatible GPU and the correct version of PyTorch installed. If you don't have a GPU, the script will automatically use the CPU.
- **Audio format issues**: The script should work with most common audio formats like `.wav`, `.mp3`, etc. Make sure the input file is accessible and in a supported format.

## License 📜

This script is provided "as-is" without warranty. You are free to use, modify, and distribute it as needed.

---

Happy transcribing! 🎉 Feel free to open an issue or contribute if you have improvements or feedback. 😊
