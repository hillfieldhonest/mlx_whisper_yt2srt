import os
import sys
import argparse
import subprocess
import mlx_whisper
from pathlib import Path

def download_youtube_audio(url, format='mp3'):
    """Function to download audio from YouTube URL (using yt-dlp)"""
    if not url:
        print('URL is not provided.')
        return None

    try:
        # Create a workspace directory in the current directory
        workspace_dir = os.path.join(os.getcwd(), 'whisper_workspace')
        os.makedirs(workspace_dir, exist_ok=True)
        os.chdir(workspace_dir)

        # Extract video ID and create output template
        # Example: youtube_XXXXXXXXXXX.mp3
        video_id = url.split('watch?v=')[-1][:11]
        expected_file = f'youtube_{video_id}.{format}'
        
        # Check if the file already exists
        if os.path.exists(expected_file):
            abs_path = os.path.abspath(expected_file)
            print(f'Audio file already exists: {abs_path}')
            return abs_path
            
        # If file doesn't exist, download it
        output_template = f'youtube_{video_id}.%(ext)s'

        # yt-dlp command (with specified options)
        command = (
            f"yt-dlp "
            f"--hls-use-mpegts "
            f"--force-overwrites "
            f"-x --audio-format {format} "
            f"-o '{output_template}' "
            f"'{url}'"
        )
        print('Downloading audio...')
        subprocess.run(command, shell=True, check=True)

        # Check if the downloaded file exists as expected
        if os.path.exists(expected_file):
            abs_path = os.path.abspath(expected_file)
            print(f'Download complete: {abs_path}')
            return abs_path
        else:
            print('Audio file not found.')
            return None

    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return None

def generate_srt(
    audio_file: str,
    model_size: str = "turbo-4bit",
    language: str = "auto"
) -> str:
    """
    Use mlx-whisper to transcribe audio file (e.g., mp3) and
    generate an SRT format subtitle file, then return its path.
    """

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return None

    try:
        # Map model sizes to MLX-whisper model names
        model_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-mlx",
            "large-4bit": "mlx-community/whisper-large-v3-mlx-4bit",
            "turbo": "mlx-community/whisper-large-v3-turbo",
            "turbo-4bit": "mlx-community/whisper-large-v3-turbo-q4"
        }
        
        # Get the appropriate model path
        model_path = model_map.get(model_size.lower(), "mlx-community/whisper-large-v3-turbo-q4")
        
        print(f"Loading MLX Whisper model '{model_path}'...")
        
        # Set language parameter if specified
        transcribe_kwargs = {}
        if language.lower() != "auto":
            transcribe_kwargs["language"] = language
            
        # Transcribe using MLX Whisper
        result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=model_path, **transcribe_kwargs)

        # Create SRT filename with model name suffix
        base_name = os.path.splitext(audio_file)[0]
        model_short_name = model_size.lower()
        srt_base_path = f"{base_name}_{model_short_name}.srt"
        
        # Check if file already exists and add sequential number if needed
        srt_path = srt_base_path
        counter = 2
        while os.path.exists(srt_path):
            srt_path = f"{base_name}_{model_short_name}_{counter:02d}.srt"
            counter += 1

        # Write results in SRT format
        write_srt(result["segments"], srt_path)
        print(f"Generated SRT file: {srt_path}")

        return srt_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def write_srt(segments, srt_path: str):
    """
    Helper function to generate SRT format file based on
    Whisper's result["segments"].
    """
    def sec_to_timestamp(sec: float):
        hours = int(sec // 3600)
        minutes = int((sec % 3600) // 60)
        seconds = int(sec % 60)
        milliseconds = int((sec - int(sec)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start_time = sec_to_timestamp(seg["start"])
            end_time = sec_to_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def process_youtube_to_srt(url, audio_format='mp3', model_size='turbo-4bit', language='auto'):
    """
    1. Download audio from specified URL
    2. Generate subtitles using Whisper (SRT)
    """
    # 1. Download (using the previously defined function)
    audio_path = download_youtube_audio(url, format=audio_format)
    if not audio_path:
        print("Audio download failed. Exiting.")
        return None

    # 2. Generate subtitles with Whisper
    print("Starting subtitle generation...")

    srt_path = generate_srt(audio_path, model_size=model_size, language=language)

    if not srt_path:
        print("Subtitle generation failed.")
        return None

    return srt_path

def interactive_mode(url=None, language=None, model_size=None, audio_format='mp3'):
    """Run in interactive mode, asking for inputs if not provided"""
    print("YouTube to SRT Converter (Interactive Mode)")
    print("==========================================")
    
    # Get YouTube URL if not provided
    if not url:
        url = input('Enter YouTube video URL: ').strip()
    else:
        print(f'Using provided YouTube URL: {url}')
    
    # Get language preference if not provided
    if not language:
        lang_input = input('Language code (options: ja, en, auto / default: auto): ').strip()
        language = lang_input if lang_input else 'auto'
    else:
        print(f'Using provided language: {language}')
    
    # Get model size preference if not provided
    if not model_size:
        print("Available model sizes:")
        print("- tiny: Fastest, least accurate")
        print("- base: Fast, basic accuracy")
        print("- small: Good balance of speed and accuracy")
        print("- medium: More accurate, slower")
        print("- large: Most accurate, slowest")
        print("- large-4bit: Better performance than large")
        print("- turbo: Fastest large model")
        print("- turbo-4bit: Recommended (fastest large model with 4-bit quantization)")
        
        model_size_input = input('Model size (default: turbo-4bit): ').strip()
        model_size = model_size_input if model_size_input else 'turbo-4bit'
    else:
        print(f'Using provided model size: {model_size}')
    
    # Display the selected configuration
    print(f"\nProcessing YouTube URL: {url}")
    print(f"Using language: {language}")
    print(f"Using model: {model_size}")
    print(f"Audio format: {audio_format}")
    
    # Process the video
    srt_file = process_youtube_to_srt(
        url=url,
        audio_format=audio_format,
        model_size=model_size,
        language=language
    )
    
    # SRT file path output
    if srt_file:
        print(f"\nDone! The SRT file is saved at: {srt_file}")
        print(f"You can find it in the 'whisper_workspace' directory.")
    else:
        print("\nFailed to create SRT file. Please check the error messages above.")

def main():
    """Main function that handles both CLI and interactive modes"""
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Convert YouTube video to SRT subtitle file using MLX Whisper')
    parser.add_argument('youtube_url', nargs='?', help='YouTube video URL (e.g., https://www.youtube.com/watch?v=XXXXXXXXXXX)')
    parser.add_argument('--language', '-l', help='Language code (e.g., en, ja, auto). Default: auto')
    parser.add_argument('--model', '-m',
                      choices=['tiny', 'base', 'small', 'medium', 'large', 'large-4bit', 'turbo', 'turbo-4bit'],
                      help='Whisper model size to use. Default: turbo-4bit')
    parser.add_argument('--audio-format', '-a',
                      choices=['mp3', 'wav', 'm4a'],
                      help='Audio format to download. Default: mp3')
    
    args = parser.parse_args()
    
    # Check if any arguments were provided, if not switch to interactive mode
    if len(sys.argv) == 1 or args.youtube_url is None:
        interactive_mode()
    else:
        # Use provided arguments but get defaults for any missing ones
        language = args.language if args.language else 'auto'
        model_size = args.model if args.model else 'turbo-4bit'
        audio_format = args.audio_format if args.audio_format else 'mp3'
        
        # If only URL is provided (without flags), offer to get remaining parameters interactively
        if args.youtube_url and not args.language and not args.model and not args.audio_format:
            use_interactive = input('Do you want to set the remaining parameters interactively? (y/n, default: n): ').strip().lower()
            if use_interactive and use_interactive.startswith('y'):
                interactive_mode(url=args.youtube_url)
                return
        
        # Process using CLI parameters
        print(f"Processing YouTube URL: {args.youtube_url}")
        print(f"Using language: {language}")
        print(f"Using model: {model_size}")
        print(f"Audio format: {audio_format}")
        
        srt_file = process_youtube_to_srt(
            url=args.youtube_url,
            audio_format=audio_format,
            model_size=model_size,
            language=language
        )
        
        # SRT file path output
        if srt_file:
            print(f"Done! The SRT file is saved at: {srt_file}")
            print(f"You can find it in the 'whisper_workspace' directory.")
        else:
            print("Failed to create SRT file. Please check the error messages above.")

if __name__ == "__main__":
    main()
