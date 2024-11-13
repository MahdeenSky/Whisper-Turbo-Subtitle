import click
import gradio as gr
from utils import language_dict
import math
import torch
import gc
import time
from faster_whisper import WhisperModel
import os
import re
import uuid
import shutil
import yt_dlp
from pydub import AudioSegment
import requests


def get_language_name(lang_code):
    global language_dict
    # Iterate through the language dictionary
    for language, details in language_dict.items():
        # Check if the language code matches
        if details["lang_code"] == lang_code:
            return language  # Return the language name
    return lang_code


def clean_file_name(file_path):
    # Get the base file name and extension
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)

    # Replace non-alphanumeric characters with an underscore
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)

    # Remove any multiple underscores
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')

    # Generate a random UUID for uniqueness
    random_uuid = uuid.uuid4().hex[:6]

    # Combine cleaned file name with the original extension
    clean_file_path = os.path.join(os.path.dirname(
        file_path), clean_file_name + f"_{random_uuid}" + file_extension)

    return clean_file_path


# Function to download the audio from a link using yt-dlp and convert to WAV
def download_and_convert_to_wav(link):
    global temp_folder
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{temp_folder}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=True)
        audio_file = ydl.prepare_filename(info_dict).rsplit(".", 1)[0] + ".wav"
    return audio_file


def format_segments(segments):
    saved_segments = list(segments)
    sentence_timestamp = []
    words_timestamp = []
    speech_to_text = ""

    for i in saved_segments:
        temp_sentence_timestamp = {}
        # Store sentence information in sentence_timestamp
        text = i.text.strip()
        # Get the current index for the new entry
        sentence_id = len(sentence_timestamp)
        sentence_timestamp.append({
            "id": sentence_id,  # Use the index as the id
            "text": text,
            "start": i.start,
            "end": i.end,
            "words": []  # Initialize words as an empty list within the sentence
        })
        speech_to_text += text + " "

        # Process each word in the sentence
        for word in i.words:
            word_data = {
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end
            }

            # Append word timestamps to the sentence's word list
            sentence_timestamp[sentence_id]["words"].append(word_data)

            # Optionally, add the word data to the global words_timestamp list
            words_timestamp.append(word_data)

    return sentence_timestamp, words_timestamp, speech_to_text


def combine_word_segments(words_timestamp, max_words_per_subtitle=8, min_silence_between_words=0.5):
    if max_words_per_subtitle <= 1:
        max_words_per_subtitle = 1
    before_translate = {}
    id = 1
    text = ""
    start = None
    end = None
    word_count = 0
    last_end_time = None

    for i in words_timestamp:
        try:
            word = i['word']
            word_start = i['start']
            word_end = i['end']

            # Check for sentence-ending punctuation
            is_end_of_sentence = word.endswith(('.', '?', '!'))

            # Check for conditions to create a new subtitle
            if ((last_end_time is not None and word_start - last_end_time > min_silence_between_words)
                or word_count >= max_words_per_subtitle
                    or is_end_of_sentence):

                # Store the previous subtitle if there's any
                if text:
                    before_translate[id] = {
                        "text": text,
                        "start": start,
                        "end": end
                    }
                    id += 1

                # Reset for the new subtitle segment
                text = word
                start = word_start  # Set the start time for the new subtitle
                word_count = 1
            else:
                if word_count == 0:  # First word in the subtitle
                    start = word_start  # Ensure the start time is set
                text += " " + word
                word_count += 1

            end = word_end  # Update the end timestamp
            last_end_time = word_end  # Update the last end timestamp

        except KeyError as e:
            print(f"KeyError: {e} - Skipping word")
            pass

    # After the loop, make sure to add the last subtitle segment
    if text:
        before_translate[id] = {
            "text": text,
            "start": start,
            "end": end
        }

    return before_translate


def convert_time_to_srt_format(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def write_subtitles_to_file(subtitles, filename="subtitles.srt"):

    # Open the file with UTF-8 encoding
    with open(filename, 'w', encoding='utf-8') as f:
        for id, entry in subtitles.items():
            # Write the subtitle index
            f.write(f"{id}\n")
            if entry['start'] is None or entry['end'] is None:
                print(id)
            # Write the start and end time in SRT format
            start_time = convert_time_to_srt_format(entry['start'])
            end_time = convert_time_to_srt_format(entry['end'])
            f.write(f"{start_time} --> {end_time}\n")

            # Write the text and speaker information
            f.write(f"{entry['text']}\n\n")


def word_level_srt(words_timestamp, srt_path="world_level_subtitle.srt"):
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, word_info in enumerate(words_timestamp, start=1):
            start_time = convert_time_to_srt_format(word_info['start'])
            end_time = convert_time_to_srt_format(word_info['end'])
            srt_file.write(
                f"{i}\n{start_time} --> {end_time}\n{word_info['word']}\n\n")


def generate_srt_from_sentences(sentence_timestamp, srt_path="default_subtitle.srt"):
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for index, sentence in enumerate(sentence_timestamp):
            start_time = convert_time_to_srt_format(sentence['start'])
            end_time = convert_time_to_srt_format(sentence['end'])
            srt_file.write(
                f"{index + 1}\n{start_time} --> {end_time}\n{sentence['text']}\n\n")


def get_audio_file(uploaded_file):
    global temp_folder
    file_path = os.path.join(temp_folder, os.path.basename(uploaded_file))
    file_path = clean_file_name(file_path)
    shutil.copy(uploaded_file, file_path)
    return file_path


def whisper_subtitle(uploaded_file, Source_Language, max_words_per_subtitle=8):
    global language_dict, base_path, subtitle_folder
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"
    faster_whisper_model = WhisperModel(
        "deepdml/faster-whisper-large-v3-turbo-ct2", device=device, compute_type=compute_type)

    if Source_Language == "Automatic":
        segments, d = faster_whisper_model.transcribe(
            uploaded_file, word_timestamps=True)
        lang_code = d.language
        src_lang = get_language_name(lang_code)
    else:
        lang = language_dict[Source_Language]['lang_code']
        segments, d = faster_whisper_model.transcribe(
            uploaded_file, word_timestamps=True, language=lang)
        src_lang = Source_Language

    sentence_timestamp, words_timestamp, text = format_segments(segments)
    if os.path.exists(uploaded_file):
        os.remove(uploaded_file)
    del faster_whisper_model
    gc.collect()
    torch.cuda.empty_cache()

    word_segments = combine_word_segments(
        words_timestamp, max_words_per_subtitle=max_words_per_subtitle)
    base_name = os.path.basename(uploaded_file).rsplit('.', 1)[0][:30]
    save_name = f"{subtitle_folder}/{base_name}_{src_lang}.srt"
    original_srt_name = clean_file_name(save_name)
    original_txt_name = original_srt_name.replace(".srt", ".txt")
    word_level_srt_name = original_srt_name.replace(".srt", "_word_level.srt")
    customize_srt_name = original_srt_name.replace(".srt", "_customize.srt")

    generate_srt_from_sentences(sentence_timestamp, srt_path=original_srt_name)
    word_level_srt(words_timestamp, srt_path=word_level_srt_name)
    write_subtitles_to_file(word_segments, filename=customize_srt_name)

    with open(original_txt_name, 'w', encoding='utf-8') as f1:
        f1.write(text)
    
    beep_audio_path = os.path.join(base_path, "beep.wav")
    return original_srt_name, customize_srt_name, word_level_srt_name, original_txt_name, beep_audio_path


def subtitle_maker(Audio_or_Video_File, Link, Source_Language, max_words_per_subtitle):
    if Link:
        Audio_or_Video_File = download_and_convert_to_wav(Link)
    elif not Audio_or_Video_File:
        raise ValueError("Either an audio/video file or a YouTube link must be provided.")
    
    try:
        default_srt_path, customize_srt_path, word_level_srt_path, text_path, beep_audio_path = whisper_subtitle(
            Audio_or_Video_File, Source_Language, max_words_per_subtitle=max_words_per_subtitle
        )
    except Exception as e:
        print(f"Error in whisper_subtitle: {e}")
        default_srt_path, customize_srt_path, word_level_srt_path, text_path, beep_audio_path = None, None, None, None, None

    return default_srt_path, customize_srt_path, word_level_srt_path, text_path, beep_audio_path


# Updated Gradio interface to accept a link as well

base_path = "."
subtitle_folder = f"{base_path}/generated_subtitle"
temp_folder = f"{base_path}/subtitle_audio"
if not os.path.exists(subtitle_folder):
    os.makedirs(subtitle_folder, exist_ok=True)
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder, exist_ok=True)

source_lang_list = ['Automatic']
available_language = language_dict.keys()
source_lang_list.extend(available_language)


@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    description = "**Note**: Avoid uploading large video files. Instead, upload the audio from the video for faster processing."
    gradio_inputs = [
        gr.File(label="Upload Audio or Video File"),  # Removed optional=True
        gr.Textbox(label="YouTube Link (optional)",
                   placeholder="Enter link here if not uploading a file"),
        gr.Dropdown(label="Language", choices=source_lang_list,
                    value="Automatic"),
        gr.Number(
            label="Max Word Per Subtitle Segment [Useful for Vertical Videos]", value=8)
    ]

    gradio_outputs = [
        gr.File(label="Default SRT File", show_label=True),
        gr.File(label="Customize SRT File", show_label=True),
        gr.File(label="Word Level SRT File", show_label=True),
        gr.File(label="Text File", show_label=True),
        gr.Audio(label="Beep Sound", autoplay=True)
    ]

    demo = gr.Interface(
        fn=subtitle_maker, inputs=gradio_inputs, outputs=gradio_outputs,
        title="Auto Subtitle Generator Using Whisper-Large-V3-Turbo-Ct2", description=description
    )

    demo.queue().launch(debug=debug, share=share)


if __name__ == "__main__":
    main()
