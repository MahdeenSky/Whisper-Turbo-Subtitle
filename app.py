import click
import gradio as gr
from utils import language_dict
from whisperx.utils import WriteTXT
from SubtitlesProcessor import SubtitlesProcessor
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

logging.basicConfig()
logging.getLogger("whisperx").setLevel(logging.INFO)


def get_language_name(lang_code):
    global language_dict
    for language, details in language_dict.items():
        if details["lang_code"] == lang_code:
            return language
    return lang_code


def clean_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')
    random_uuid = uuid.uuid4().hex[:6]
    clean_file_path = os.path.join(os.path.dirname(
        file_path), clean_file_name + f"_{random_uuid}" + file_extension)
    print(f"Cleaned file name: {clean_file_path}")
    return clean_file_path


def download_audio(link):
    print(f"Downloading audio from link: {link}")
    global temp_folder
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'outtmpl': f'{temp_folder}/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=True)
        if 'entries' in info_dict:
            audio_file = temp_folder + "/" + \
                info_dict['entries'][0]['title'] + \
                "." + info_dict['entries'][0]['ext']
        else:
            audio_file = ydl.prepare_filename(info_dict)
    print(f"Downloaded audio file: {audio_file}")
    return audio_file


def get_audio_file(uploaded_file):
    global temp_folder
    file_path = os.path.join(temp_folder, os.path.basename(uploaded_file))
    file_path = clean_file_name(file_path)
    shutil.copy(uploaded_file, file_path)
    print(f"Copied uploaded file to: {file_path}")
    return file_path


def get_media_duration(file_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)


# Global variable to store the loaded model
global_model = None
global_align_model = None
metadata = None


def whisper_subtitle(uploaded_file, Source_Language, translate=False, device="cuda", compute_type="float16"):
    global language_dict, base_path, subtitle_folder, global_model, global_align_model, metadata

    print("Starting transcription process...")
    total_start = time.time()

    duration = math.ceil(get_media_duration(uploaded_file))
    print(f"Audio duration: {duration} seconds")

    if global_model is None:
        loading_start = time.time()
        print("Loading WhisperX model...")
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device = "cpu"
            compute_type = "int8"

        global_model = whisperx.load_model(
            "large-v2", device, compute_type=compute_type)
        print("WhisperX model loaded in {:.2f} seconds.".format(
            time.time() - loading_start))
    model = global_model

    whisper_start = time.time()
    audio = whisperx.load_audio(uploaded_file)
    task = "translate" if translate else "transcribe"
    print(f"Transcribing audio with task: {task}")
    result = model.transcribe(audio, batch_size=32 if device == "cuda" else 1, task=task,
                              language=language_dict[Source_Language]['lang_code'])
    whisper_end = time.time()
    print(f"Whisper took {whisper_end - whisper_start:.2f} seconds.")

    src_lang = Source_Language
    print(f"Using provided source language: {src_lang}")
    language_code = language_dict[src_lang]['lang_code']

    if global_align_model is None:
        loading_start = time.time()
        print("Loading alignment model...")
        global_align_model, metadata = whisperx.load_align_model(
            language_code=language_code, device=device)
        print("Alignment model loaded in {:.2f} seconds.".format(
            time.time() - loading_start))
    align_model = global_align_model

    print("Aligning transcription results...")
    alignment_start = time.time()
    result = whisperx.align(result["segments"], align_model,
                            metadata, audio, device, return_char_alignments=False)
    alignment_end = time.time()
    print(f"Alignment took {alignment_end - alignment_start:.2f} seconds.")

    if os.path.exists(uploaded_file):
        os.remove(uploaded_file)
        print(f"Removed temporary uploaded file: {uploaded_file}")

    base_name = os.path.basename(uploaded_file).rsplit('.', 1)[0][:30]
    srt_name = clean_file_name(f"{subtitle_folder}/{base_name}_{src_lang}.srt")
    txt_name = srt_name.replace(".srt", ".txt")

    txt_options = {"max_line_width": None,
                   "max_line_count": None,
                   "highlight_words": False}

    srt_options = {"max_line_width": 100,
                   "min_char_length_splitter": 70,
                   "is_vtt": False,
                   "lang": language_code}

    result["language"] = language_code
    WriteTXT(subtitle_folder)(result, txt_name, txt_options)
    print(f"Writing TXT file to: {txt_name}")

    subtitles_processor = SubtitlesProcessor(
        result["segments"],
        lang=srt_options["lang"],
        max_line_length=srt_options["max_line_width"],
        min_char_length_splitter=srt_options["min_char_length_splitter"],
        is_vtt=srt_options["is_vtt"],
    )
    # output_path is a str with your desired filename
    subtitles_processor.save(srt_name, advanced_splitting=True)
    print(f"Writing SRT file to: {srt_name}")

    beep_audio_path = os.path.join(base_path, "beep.wav")
    total_end = time.time()

    print(
        f"Transcription Process completed in {total_end - total_start:.2f} seconds.")
    print(f"WhisperX time: {whisper_end - whisper_start:.2f} seconds")
    print(f"Alignment time: {alignment_end - alignment_start:.2f} seconds")
    print(
        f"Speed of WhisperX: {duration / (whisper_end - whisper_start):.2f}x real-time")
    print(
        f"Speed of WhisperX + Alignment: {duration / (whisper_end - whisper_start + alignment_end - alignment_start):.2f}x real-time")

    del audio
    gc.collect()

    return srt_name, txt_name, beep_audio_path


def subtitle_maker(Audio_or_Video_File, Link, File_Path, Source_Language, translate, device, compute_type):
    if Link:
        print(f"Processing YouTube link: {Link}")
        Audio_or_Video_File = download_audio(Link)
    elif File_Path:
        print(f"Processing file path: {File_Path}")
        Audio_or_Video_File = File_Path
    elif not Audio_or_Video_File:
        raise ValueError(
            "Either an audio/video file, a YouTube link, or a file path must be provided.")

    try:
        srt_path, txt_path, beep_audio_path = whisper_subtitle(
            Audio_or_Video_File, Source_Language, translate=translate, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Error in whisper_subtitle: {e}")
        srt_path, txt_path, beep_audio_path = None, None, None

    return srt_path, txt_path, beep_audio_path


base_path = "."
subtitle_folder = f"{base_path}/generated_subtitle"
temp_folder = f"{base_path}/subtitle_audio"
os.makedirs(subtitle_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)
print(f"Created directories: {subtitle_folder}, {temp_folder}")

source_lang_list = list(language_dict.keys())


@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
@click.option("--device", default="cuda", help="Device to use (cuda or cpu)")
@click.option("--compute_type", default="float16", help="Compute type (float16, float32 or int8)")
def main(debug, share, device, compute_type):
    global global_model, global_align_model, metadata
    description = "**Note**: Avoid uploading large video files. Instead, upload the audio from the video for faster processing."

    gradio_inputs = [
        gr.File(label="Upload Audio or Video File"),
        gr.Textbox(label="YouTube Link (optional)",
                   placeholder="Enter link here if not uploading a file"),
        gr.Textbox(label="File Path (optional)",
                   placeholder="Enter file path here if not uploading a file or link"),
        gr.Dropdown(label="Language", choices=source_lang_list,
                    value="English"),
        gr.Checkbox(label="Translate to English", value=False)
    ]

    gradio_outputs = [
        gr.File(label="Default SRT File", show_label=True),
        gr.File(label="Text File", show_label=True),
        gr.Audio(label="Beep Sound", autoplay=True)
    ]

    demo = gr.Interface(
        fn=lambda *args, **kwargs: subtitle_maker(
            *args, device=device, compute_type=compute_type, **kwargs),
        inputs=gradio_inputs, outputs=gradio_outputs,
        title="Auto Subtitle Generator Using WhisperX", description=description
    )

    try:
        print("Launching Gradio interface...")
        demo.queue().launch(debug=debug, share=share)
    except Exception as e:
        print(f"Error during launch: {e}")
    finally:
        if global_model is not None:
            del global_model
            gc.collect()
            torch.cuda.empty_cache()
            print("WhisperX model unloaded.")

        if global_align_model is not None:
            del global_align_model
            gc.collect()
            torch.cuda.empty_cache()
            print("Alignment model unloaded.")

        del metadata


if __name__ == "__main__":
    main()
