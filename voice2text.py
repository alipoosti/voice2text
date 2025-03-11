import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

########################################
########### Transcribe Audio ###########
########################################

# Function to transcribe audio to text
def transcribe_audio(input_wav: str, output_txt: str, model: str = "distil-whisper/distil-large-v3"):
    # Check if a GPU is available, otherwise fall back to CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Set the data type based on the device (use float16 on GPU, float32 on CPU)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = model  # The model to be used for transcription

    # Load the pre-trained model for speech-to-text
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)  # Move the model to the selected device (GPU or CPU)

    # Load the processor associated with the model (tokenizer and feature extractor)
    processor = AutoProcessor.from_pretrained(model_id)
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id  # Set pad token for the model

    # Create a pipeline for automatic speech recognition (ASR)
    pipe = pipeline(
        "automatic-speech-recognition",  # Define the task
        model=model,  # Provide the model
        tokenizer=processor.tokenizer,  # Provide the tokenizer
        feature_extractor=processor.feature_extractor,  # Provide the feature extractor
        max_new_tokens=128,  # Limit the number of tokens generated
        chunk_length_s=25,  # Process the audio in chunks of 25 seconds
        batch_size=16,  # Process in batches of 16
        torch_dtype=torch_dtype,  # Set the data type for the model
        device=device,  # Specify the device (GPU or CPU)
    )

    # Transcribe the input audio using the pipeline
    result = pipe(input_wav)
    transcript = result["text"]  # Extract the transcribed text from the result
    
    # Save the transcription result to the specified output text file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    # Print a message indicating that transcription is complete
    print(f"Transcription complete. Output saved to {output_txt}")

########################################
################# Main #################
########################################

# Main function to handle command-line inputs and call the transcription function
def main():
    # Set up argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio to text using a speech recognition model.")
    
    # Command-line argument for the input audio file (e.g., .wav, .mp3)
    parser.add_argument("input_audio", help="Path to the input audio file (e.g., .wav, .mp3)")
    
    # Optional argument for the output text file (default is input_audio_name.txt)
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to the output text file (default is <input_audio_name>.txt)"
    )
    
    # Optional argument for specifying the model (default is 'distil-whisper/distil-large-v3')
    parser.add_argument(
        "--model",
        default="distil-whisper/distil-large-v3",
        help="Model name or path for transcription (default is 'distil-whisper/distil-large-v3')"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # If the output file is not provided, create a default output file name based on the input file name
    if args.output is None:
        output_file = f"{args.input_audio.rsplit('.', 1)[0]}.txt"  # Strip the extension and add .txt
    else:
        output_file = args.output  # Use the provided output file name

    # Call the transcription function with the provided input audio, output file, and model
    transcribe_audio(args.input_audio, output_file, args.model)

########################################
############## Call Main ###############
########################################

if __name__ == "__main__":
    main()  
