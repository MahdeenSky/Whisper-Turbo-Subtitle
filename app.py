import click
import gradio as gr
from utils import language_dict # Assuming this file exists and is correct
from whisperx.utils import WriteTXT # WriteSRT is not used directly if using SubtitlesProcessor
from SubtitlesProcessor import SubtitlesProcessor # Use the updated version
import math
import torch
import gc
import time
import os
import re
import uuid
import shutil
import yt_dlp
import logging
import whisperx
import subprocess

logging.basicConfig(level=logging.INFO) # Set root logger level
logging.getLogger("whisperx").setLevel(logging.INFO)
# For more detailed whisperx logs if needed:
# logging.getLogger("whisperx").setLevel(logging.DEBUG)


def get_language_name(lang_code):
    # Assuming language_dict is globally available or passed correctly
    for language, details in language_dict.items():
        if details["lang_code"] == lang_code:
            return language
    return lang_code


def clean_file_name(file_path):
    file_name_full = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name_full)
    # Allow more characters like hyphens, keep original casing for a bit
    cleaned = re.sub(r'[^\w\-\s\.]', '', file_name) # Allow word chars, hyphen, space, dot
    cleaned = re.sub(r'\s+', '_', cleaned) # Replace spaces with underscores
    # Remove any leading/trailing underscores and ensure it's not empty
    clean_file_name_base = cleaned.strip('_')
    if not clean_file_name_base: # If name becomes empty after cleaning
        clean_file_name_base = "media_file"

    random_uuid = uuid.uuid4().hex[:6]
    # Construct the new name, ensuring it's not too long if needed
    # For simplicity, let's assume the cleaning is sufficient
    final_file_name = f"{clean_file_name_base}_{random_uuid}{file_extension}"
    clean_file_path = os.path.join(os.path.dirname(file_path), final_file_name)
    logging.info(f"Original name: {file_name_full}, Cleaned file name for output: {final_file_name}")
    return clean_file_path


def download_audio(link):
    logging.info(f"Downloading audio from link: {link}")
    # Assuming temp_folder is globally defined or passed
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'outtmpl': os.path.join(temp_folder, '%(title)s.%(ext)s'), # Use os.path.join
        'postprocessors': [{ # Ensure it's audio
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3', # or 'wav', 'm4a' etc.
            'preferredquality': '192',
        }],
        'keepvideo': False, # Don't keep video if it was a video link
        'retries': 5,
    }
    downloaded_file_path = None
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=True)
        # ydl.prepare_filename might give the original template, not the postprocessed one.
        # It's safer to find the downloaded .mp3 (or chosen audio format) file.
        # For simplicity, let's assume it's the title + .mp3 in temp_folder
        base_name = ydl.prepare_filename(info_dict).rsplit('.', 1)[0]
        # The extension might change due to postprocessing. Check for common audio extensions.
        possible_extensions = ['.mp3', '.wav', '.m4a', '.ogg'] # Add others if needed
        for ext in possible_extensions:
            potential_file = base_name + ext
            if os.path.exists(potential_file):
                downloaded_file_path = potential_file
                break
        if not downloaded_file_path and 'requested_downloads' in info_dict and info_dict['requested_downloads']:
             downloaded_file_path = info_dict['requested_downloads'][0]['filepath']


    if not downloaded_file_path or not os.path.exists(downloaded_file_path):
        # Fallback: search for the newest audio file in temp_folder if specific name fails
        list_of_files = [os.path.join(temp_folder, f) for f in os.listdir(temp_folder) if os.path.isfile(os.path.join(temp_folder, f))]
        if list_of_files:
            downloaded_file_path = max(list_of_files, key=os.path.getctime)
            logging.warning(f"Could not determine exact downloaded filename, using newest file: {downloaded_file_path}")
        else:
            raise FileNotFoundError("yt-dlp downloaded a file, but it could not be found in the temp_folder.")


    logging.info(f"Downloaded audio file: {downloaded_file_path}")
    # Clean the downloaded file name before returning
    cleaned_path = clean_file_name(downloaded_file_path)
    if downloaded_file_path != cleaned_path and os.path.exists(downloaded_file_path):
        if os.path.exists(cleaned_path): # Safety check if clean_file_name is not perfectly unique
            os.remove(cleaned_path)
        os.rename(downloaded_file_path, cleaned_path)
        logging.info(f"Renamed downloaded file to: {cleaned_path}")
        return cleaned_path
    return downloaded_file_path


def get_audio_file(uploaded_file_path_temp): # Gradio provides a temp path
    # Assuming temp_folder is globally defined or passed
    # Create a non-temporary copy with a cleaned name in our temp_folder
    target_base_name = os.path.basename(uploaded_file_path_temp)
    # Use clean_file_name to get a unique and clean name in temp_folder
    permanent_file_path = clean_file_name(os.path.join(temp_folder, target_base_name))

    shutil.copy(uploaded_file_path_temp, permanent_file_path)
    logging.info(f"Copied uploaded file from {uploaded_file_path_temp} to: {permanent_file_path}")
    return permanent_file_path


def get_media_duration(file_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True # Raise an exception for non-zero exit codes
        )
        return float(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"ffprobe error: {e.stdout.decode()}")
        raise
    except FileNotFoundError:
        logging.error("ffprobe not found. Please ensure ffmpeg (and ffprobe) is installed and in your PATH.")
        raise


# Global variable to store the loaded model
global_model = None
global_align_model = None
global_align_model_lang = None # To track language of loaded align model
metadata = None # metadata is usually associated with align_model


def whisper_subtitle(
    uploaded_file_path: str,
    Source_Language: str,
    model_name: str,
    translate: bool = False,
    device: str = "cuda",
    compute_type: str = "float16",
    align: bool = True,
    word_level_timestamps_enabled: bool = False # New parameter
):
    global language_dict, base_path, subtitle_folder, temp_folder
    global global_model, global_align_model, global_align_model_lang, metadata

    logging.info("Starting transcription process...")
    total_start_time = time.time()

    try:
        duration = math.ceil(get_media_duration(uploaded_file_path))
        logging.info(f"Audio duration: {duration} seconds")
    except Exception as e:
        logging.error(f"Could not get media duration for {uploaded_file_path}: {e}")
        # Optionally, re-raise or return an error state
        raise

    current_language_code = language_dict[Source_Language]['lang_code']

    # --- Model Loading ---
    if global_model is None or global_model.model_name != model_name: # Also check if model_name changed
        loading_start = time.time()
        logging.info(f"Loading WhisperX model: {model_name} on {device} with {compute_type}...")
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU.")
            device = "cpu"
            compute_type = "auto" # let whisperx decide best for CPU, e.g. int8

        # Unload previous model if it exists
        if global_model is not None:
            del global_model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        global_model = whisperx.load_model(
            model_name, device, compute_type=compute_type, language=current_language_code if current_language_code else None)
        global_model.model_name = model_name # Store for comparison
        logging.info("WhisperX model loaded in {:.2f} seconds.".format(time.time() - loading_start))
    model = global_model

    # --- Transcription ---
    whisper_start_time = time.time()
    logging.info(f"Loading audio from: {uploaded_file_path}")
    audio = whisperx.load_audio(uploaded_file_path)
    task = "translate" if translate else "transcribe"
    logging.info(f"Transcribing audio with task: {task}, language: {current_language_code or 'auto-detect'}")
    result = model.transcribe(audio, batch_size=16, task=task, language=current_language_code if not translate else None) # provide lang for transcribe, not for translate
    whisper_end_time = time.time()
    logging.info(f"Transcription (Whisper) took {whisper_end_time - whisper_start_time:.2f} seconds.")

    detected_language_code = result["language"]
    logging.info(f"Detected language by Whisper: {get_language_name(detected_language_code)} ({detected_language_code})")

    # If user specified a source language and it differs from detected, prefer user's for alignment
    # unless user selected "Auto-Detect", in which case use detected.
    align_language_code = current_language_code if Source_Language != "Auto-Detect" else detected_language_code
    if translate: # If translating, the source for alignment is the detected original language
        align_language_code = detected_language_code

    # --- Alignment ---
    # Word-level timestamps require alignment. If translation is active, alignment is on the original language.
    # WhisperX alignment does not work on the *translated* text directly.
    # It aligns the *original* language transcription.
    # If word_level_timestamps_enabled is True for a translated output, the word timings will be for the *source* words,
    # which is generally not what users expect for translated word-level subtitles.
    # For now, if translate=True and word_level_timestamps_enabled=True, SubtitlesProcessor will do its best by estimating.

    actual_align_needed = align # User's general preference
    if word_level_timestamps_enabled and not translate:
        if not align:
            logging.info("INFO: Word-level timestamps chosen, enabling alignment (if not translating).")
        actual_align_needed = True
    
    alignment_performed_successfully = False
    if actual_align_needed and not translate:
        if global_align_model is None or global_align_model_lang != align_language_code:
            loading_start = time.time()
            logging.info(f"Loading alignment model for language: {align_language_code}...")
            # Unload previous align model
            if global_align_model is not None: del global_align_model
            if metadata is not None: del metadata
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            try:
                global_align_model, metadata = whisperx.load_align_model(language_code=align_language_code, device=device)
                global_align_model_lang = align_language_code
                logging.info("Alignment model loaded in {:.2f} seconds.".format(time.time() - loading_start))
            except Exception as e:
                logging.error(f"Failed to load alignment model for {align_language_code}: {e}")
                global_align_model = None # Ensure it's None if loading failed
                actual_align_needed = False # Cannot perform alignment
                
        if global_align_model: # Check if model was loaded successfully
            logging.info("Aligning transcription results...")
            alignment_start_time = time.time()
            try:
                result = whisperx.align(result["segments"], global_align_model, metadata, audio, device, return_char_alignments=False)
                alignment_performed_successfully = True
                logging.info(f"Alignment took {time.time() - alignment_start_time:.2f} seconds.")
            except Exception as e:
                logging.error(f"Error during whisperx.align: {e}. Proceeding without word-level alignment from whisperx.")
                alignment_performed_successfully = False
        else:
            logging.warning(f"Alignment model for {align_language_code} not available or failed to load. Skipping whisperx alignment.")
            actual_align_needed = False # Cannot perform alignment

    elif translate and word_level_timestamps_enabled:
        logging.warning("Word-level timestamps for translated text will be estimated as WhisperX aligns original language.")
    elif not actual_align_needed and word_level_timestamps_enabled:
        logging.warning("Word-level timestamps requested without alignment. Timings will be estimated by SubtitleProcessor and may be less accurate.")


    # --- Output File Naming and Path ---
    # Use the cleaned base name of the *original* uploaded file for output.
    # uploaded_file_path is already cleaned and in temp_folder
    base_name_for_output = os.path.basename(uploaded_file_path).rsplit('.', 1)[0]
    # Remove the trailing _uuid if present from clean_file_name
    base_name_for_output = re.sub(r'_[a-f0-9]{6}$', '', base_name_for_output)


    # Determine language suffix for filename
    # If translated, output is English. Otherwise, it's the source/detected language.
    output_lang_code = "en" if translate else align_language_code # align_language_code is best guess for source
    output_lang_suffix = f"_{output_lang_code}"
    if word_level_timestamps_enabled:
        output_lang_suffix += "_wordlevel"

    srt_name = os.path.join(subtitle_folder, f"{base_name_for_output}{output_lang_suffix}.srt")
    txt_name = os.path.join(subtitle_folder, f"{base_name_for_output}{output_lang_suffix}.txt")


    # --- Subtitle Processing and Saving ---
    processor_lang_for_subprocessor = output_lang_code

    subtitles_processor = SubtitlesProcessor(
        segments=result["segments"], # segments may or may not have "words" from alignment
        lang=processor_lang_for_subprocessor,
        max_line_length=42, # Default for line-based, not used by save_word_by_word
        min_char_length_splitter=15, # Default for line-based
        is_vtt=False # Outputting SRT
    )

    if word_level_timestamps_enabled:
        # save_word_by_word will use result["segments"] which includes "words" if alignment was successful.
        # Otherwise, it will use its internal estimation logic.
        subtitles_processor.save_word_by_word(srt_name)
        logging.info(f"Writing WORD-LEVEL SRT file to: {srt_name}")
    else:
        # This is the original line-based subtitle generation
        subtitles_processor.save(srt_name, advanced_splitting=True)
        logging.info(f"Writing LINE-BASED SRT file to: {srt_name}")

    # Write TXT file (always, regardless of word-level choice for SRT)
    # whisperx.utils.WriteTXT expects result with "segments" key
    # If translation happened, result["segments"] text is translated.
    # If alignment happened, result["segments"] has word timings for original lang.
    # WriteTXT just dumps the segment text.
    txt_write_options = {"max_line_width": None, "max_line_count": None, "highlight_words": False}
    writer = WriteTXT(output_dir=subtitle_folder) # Specify output_dir for WriteTXT
    # WriteTXT expects the second arg to be the base of the audio file path, it constructs the name.
    # Let's manually control the name.
    with open(txt_name, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            f.write(segment["text"].strip() + "\n")
    logging.info(f"Writing TXT file to: {txt_name}")


    # --- Cleanup and Stats ---
    if os.path.exists(uploaded_file_path): # This is the file in temp_folder
        try:
            os.remove(uploaded_file_path)
            logging.info(f"Removed temporary audio file: {uploaded_file_path}")
        except OSError as e:
            logging.warning(f"Could not remove temporary audio file {uploaded_file_path}: {e}")


    beep_audio_path = os.path.join(base_path, "beep.wav") # Ensure beep.wav exists
    total_end_time = time.time()

    logging.info(f"Transcription Process completed in {total_end_time - total_start_time:.2f} seconds.")
    logging.info(f"WhisperX transcription time: {whisper_end_time - whisper_start_time:.2f} seconds.")
    if actual_align_needed and alignment_performed_successfully and not translate:
        # Recalculate alignment duration if it happened
        # align_end_time = alignment_start_time + (time.time() - alignment_start_time) # This is wrong, alignment already finished
        # We logged alignment duration when it happened.
        pass # Already logged

    processing_speed = duration / (whisper_end_time - whisper_start_time)
    logging.info(f"Speed of WhisperX transcription: {processing_speed:.2f}x real-time")

    # Clean up large objects
    del audio, result
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return srt_name, txt_name, beep_audio_path


def subtitle_maker(
    Audio_or_Video_File_Temp, # This is the temp file path from Gradio's gr.File
    Link,
    File_Path_Input, # User provided string path
    Source_Language,
    model_name,
    translate,
    align,
    word_level_timestamps, # New parameter from Gradio
    device,
    compute_type
):
    processed_audio_file = None
    try:
        if Link:
            logging.info(f"Processing YouTube link: {Link}")
            processed_audio_file = download_audio(Link)
        elif File_Path_Input: # User typed a path
            logging.info(f"Processing file path input: {File_Path_Input}")
            if not os.path.exists(File_Path_Input):
                raise FileNotFoundError(f"Provided file path does not exist: {File_Path_Input}")
            # Copy to temp_folder with cleaned name to handle spaces/special chars and work locally
            processed_audio_file = get_audio_file(File_Path_Input)
        elif Audio_or_Video_File_Temp and hasattr(Audio_or_Video_File_Temp, 'name'): # Gradio uploaded file
            logging.info(f"Processing uploaded file: {Audio_or_Video_File_Temp.name}")
            processed_audio_file = get_audio_file(Audio_or_Video_File_Temp.name)
        else:
            # No input provided
            raise ValueError("No input provided. Please upload a file, provide a YouTube link, or enter a file path.")

        if not processed_audio_file or not os.path.exists(processed_audio_file):
             raise FileNotFoundError(f"Audio file could not be prepared or found: {processed_audio_file}")

        srt_path, txt_path, beep_audio_path = whisper_subtitle(
            processed_audio_file, Source_Language, model_name,
            translate=translate, device=device, compute_type=compute_type,
            align=align, word_level_timestamps_enabled=word_level_timestamps # Pass it here
        )
        return srt_path, txt_path, beep_audio_path

    except Exception as e:
        logging.error(f"Error in subtitle_maker: {e}", exc_info=True)
        # Clean up the processed_audio_file if it exists and an error occurred
        if processed_audio_file and os.path.exists(processed_audio_file):
            try:
                os.remove(processed_audio_file)
                logging.info(f"Cleaned up intermediate file due to error: {processed_audio_file}")
            except OSError as e_clean:
                logging.warning(f"Could not clean up {processed_audio_file} after error: {e_clean}")
        # Re-raise or return error message for Gradio
        # Gradio expects a tuple of outputs, so return Nones or error messages
        # For file outputs, None is fine. For text, an error message.
        # return None, str(e), None # if you have a text output for errors
        raise gr.Error(f"Processing failed: {str(e)}") # This will show up in Gradio UI


# --- Setup Paths and Directories ---
base_path = os.path.abspath(".") # Get absolute path
subtitle_folder = os.path.join(base_path, "generated_subtitle")
temp_folder = os.path.join(base_path, "subtitle_audio_temp") # Renamed for clarity
os.makedirs(subtitle_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)
logging.info(f"Base path: {base_path}")
logging.info(f"Subtitle folder: {subtitle_folder}")
logging.info(f"Temporary audio folder: {temp_folder}")


# --- Language and Model Lists ---
# Ensure language_dict is loaded from utils.py
# If utils.py is not in the same dir, adjust sys.path or how it's imported
try:
    from utils import language_dict
except ImportError:
    logging.error("Failed to import language_dict from utils.py. Please ensure utils.py is accessible.")
    # Provide a fallback or exit
    language_dict = {"English": {"lang_code": "en"}, "Spanish": {"lang_code": "es"}} # Minimal fallback

# Add "Auto-Detect" option
source_lang_list_with_auto = ["Auto-Detect"] + list(language_dict.keys())
default_lang = "English" if "English" in source_lang_list_with_auto else source_lang_list_with_auto[0]


# Model list - consider adding more options or making it dynamic if possible
model_list = [
    "large-v3", # OpenAI's Whisper large-v3
    "large-v2", # OpenAI's Whisper large-v2
    "medium", "small", "base", "tiny",
    "distil-large-v2", # Distil-Whisper
    # Add any faster-whisper specific models if you are using a loader that supports them
    # e.g., "Systran/faster-whisper-large-v3" - but whisperx.load_model expects HF path or local path for CT2 models
]
default_model = "large-v3" if "large-v3" in model_list else model_list[0]


@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode for Gradio.")
@click.option("--share", is_flag=True, default=False, help="Enable Gradio sharing link.")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use (cuda or cpu)")
@click.option("--compute_type", default="float16", type=click.Choice(["float16", "int8", "float32", "auto"]), help="Compute type for Whisper model (e.g., float16, int8, auto)")
def main(debug, share, device, compute_type):
    global global_model, global_align_model, metadata # Ensure these are accessible if needed for cleanup

    # Check CUDA availability at startup and adjust device/compute_type if necessary
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA selected but not available. Switching to CPU.")
        device = "cpu"
        if compute_type not in ["int8", "auto"]: # float16/32 not ideal for CPU
             logging.info(f"Compute type {compute_type} may be suboptimal for CPU. Consider 'int8' or 'auto'. Forcing 'auto'.")
             compute_type = "auto"


    description = (
        "**Auto Subtitle Generator using WhisperX**\n\n"
        "Upload an audio/video file, provide a YouTube link, or specify a local file path.\n"
        "**Note**: For large video files, consider extracting audio first for faster processing.\n"
        "Word-level timestamps require alignment and work best when not translating."
    )

    # Define Gradio inputs
    input_file = gr.File(label="Upload Audio or Video File (.mp3, .wav, .mp4, .mkv, etc.)")
    input_youtube_link = gr.Textbox(label="YouTube Link (Optional)", placeholder="e.g., https://www.youtube.com/watch?v=...")
    input_file_path = gr.Textbox(label="Local File Path (Optional)", placeholder="e.g., /path/to/your/audio.mp3")

    dropdown_language = gr.Dropdown(label="Source Language", choices=source_lang_list_with_auto, value=default_lang, info="Select 'Auto-Detect' for automatic language detection.")
    dropdown_model = gr.Dropdown(label="Whisper Model", choices=model_list, value=default_model, info="Larger models are more accurate but slower.")

    checkbox_translate = gr.Checkbox(label="Translate to English", value=False, info="If checked, output will be English subtitles.")
    checkbox_align = gr.Checkbox(label="Perform Alignment", value=True, info="Aligns transcription with audio for accurate timing. Highly recommended, especially for word-level.")
    checkbox_word_level = gr.Checkbox(label="Word-Level Timestamps", value=False, info="Generate subtitles with timestamps for each word. Requires alignment. May be less accurate if translating.") # New

    gradio_inputs = [
        input_file, input_youtube_link, input_file_path,
        dropdown_language, dropdown_model,
        checkbox_translate, checkbox_align, checkbox_word_level # Added new checkbox
    ]

    # Define Gradio outputs
    output_srt_file = gr.File(label="Generated SRT Subtitle File")
    output_txt_file = gr.File(label="Generated TXT Transcript File")
    output_beep_sound = gr.Audio(label="Notification Sound", autoplay=True, visible=False) # Often hidden or used for actual audio playback

    interface = gr.Interface(
        fn=lambda audio_f, link, path_str, lang, model, trans, al, wl_ts: subtitle_maker(
            audio_f, link, path_str, lang, model, trans, al, wl_ts, # Pass word_level_timestamps
            device=device, compute_type=compute_type
        ),
        inputs=gradio_inputs,
        outputs=[output_srt_file, output_txt_file, output_beep_sound],
        title="Auto Subtitle Generator (WhisperX)",
        description=description,
        allow_flagging="never",
        examples=[
            [None, "https://www.youtube.com/watch?v=dQw4w9WgXcQ", None, "Auto-Detect", "large-v3", False, True, False],
            [None, None, None, "English", "medium", False, True, True], # Will need a file upload for this example to run
        ]
    )

    try:
        logging.info("Launching Gradio interface...")
        interface.queue().launch(debug=debug, share=share, inbrowser=True)
    except Exception as e:
        logging.error(f"Error during Gradio launch: {e}", exc_info=True)
    finally:
        # Cleanup global models (optional, Gradio might handle process exit)
        logging.info("Cleaning up models...")
        if global_model is not None:
            del global_model
        if global_align_model is not None:
            del global_align_model
        if metadata is not None:
            del metadata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Models cleaned up.")

        # Clean up temp_folder contents
        logging.info(f"Cleaning up temporary files in {temp_folder}...")
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')
        logging.info(f"Temporary files in {temp_folder} cleaned.")


if __name__ == "__main__":
    # This structure allows Click to process CLI args before Gradio launches
    main()