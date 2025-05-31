import math
# Assuming conjunctions.py exists and has these functions
# For standalone testing, you might need to mock them:
# def get_conjunctions(lang): return []
# def get_comma(lang): return ","
from conjunctions import get_conjunctions, get_comma
from typing import TextIO

def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def format_timestamp(seconds: float, is_vtt: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    separator = '.' if is_vtt else ','
    
    hours_marker = f"{hours:02d}:"
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"
    )


class SubtitlesProcessor:
    def __init__(self, segments, lang, max_line_length = 45, min_char_length_splitter = 30, is_vtt = False):
        self.comma = get_comma(lang)
        self.conjunctions = set(get_conjunctions(lang))
        self.segments = segments
        self.lang = lang
        self.max_line_length = max_line_length
        self.min_char_length_splitter = min_char_length_splitter
        self.is_vtt = is_vtt
        complex_script_languages = ['th', 'lo', 'my', 'km', 'am', 'ko', 'ja', 'zh', 'ti', 'ta', 'te', 'kn', 'ml', 'hi', 'ne', 'mr', 'ar', 'fa', 'ur', 'ka']
        if self.lang in complex_script_languages:
            self.max_line_length = 30
            self.min_char_length_splitter = 20

    def estimate_timestamp_for_word(self, words, i, next_segment_start_time=None):
        # (Original estimate_timestamp_for_word code remains unchanged)
        k = 0.25 # Proportional factor for word length based duration estimation
        current_word_obj = words[i]
        
        # Ensure word_obj is a dict, if not, try to make it one (basic case)
        if not isinstance(current_word_obj, dict):
            words[i] = {'word': str(current_word_obj)} # Convert to dict
            current_word_obj = words[i]


        # Check if timestamps already exist
        if 'start' in current_word_obj and 'end' in current_word_obj:
            # If 'end' is 0 and 'start' is 0, it might be an uninitialized word.
            # Or if start and end are identical for a non-empty word.
            if not (current_word_obj['start'] == 0 and current_word_obj['end'] == 0 and len(current_word_obj['word']) > 0) and \
               not (current_word_obj['start'] == current_word_obj['end'] and len(current_word_obj['word']) > 0) :
                return # Timestamps seem valid or intentionally set, do nothing.


        word_len_factor = len(current_word_obj.get('word', '')) * k

        has_prev_end = i > 0 and isinstance(words[i-1], dict) and 'end' in words[i-1]
        has_next_start = i < len(words) - 1 and isinstance(words[i+1], dict) and 'start' in words[i+1]

        if has_prev_end:
            current_word_obj['start'] = words[i-1]['end']
            if has_next_start:
                # If the gap to the next word is very small, or negative (overlap),
                # adjust based on a small positive duration or split the difference.
                # For simplicity, let's assume next word's start is the end if it's too close.
                if words[i+1]['start'] - current_word_obj['start'] < 0.05 and len(current_word_obj.get('word', '')) > 0 : # tiny duration
                     current_word_obj['end'] = current_word_obj['start'] + 0.05 # minimum duration
                else:
                    current_word_obj['end'] = words[i+1]['start']
            else: # No next word with a start time in this segment
                if next_segment_start_time:
                    # End before the next segment, or with a small duration
                    potential_end = next_segment_start_time - 0.01 # a tiny gap
                    current_word_obj['end'] = max(current_word_obj['start'] + word_len_factor, current_word_obj['start'] + 0.05) # ensure min duration
                    if current_word_obj['end'] > potential_end:
                         current_word_obj['end'] = potential_end
                else: # This is the last word of the last segment
                    current_word_obj['end'] = current_word_obj['start'] + word_len_factor
                    if current_word_obj['end'] <= current_word_obj['start'] and len(current_word_obj.get('word', '')) > 0: # ensure positive duration
                        current_word_obj['end'] = current_word_obj['start'] + 0.05


        elif has_next_start:
            current_word_obj['end'] = words[i+1]['start']
            current_word_obj['start'] = words[i+1]['start'] - word_len_factor
            if current_word_obj['start'] >= current_word_obj['end'] and len(current_word_obj.get('word', '')) > 0 : # ensure positive duration
                 current_word_obj['start'] = current_word_obj['end'] - 0.05


        else: # No adjacent words with timestamps, or this is a single word segment without prior context
            # This case is tricky. If segment times are available, use them.
            # For word-by-word, this word might be the only one, or isolated.
            # Let's assume it's within its parent segment's time if available.
            # If this function is called from get_word_by_word_subtitles, segment_start/end might not be directly passed.
            # For now, a simple estimation based on length or default.
            
            # Fallback: try to use segment start/end if available via words[i]['segment_start_time']
            # This part is more heuristic if isolated.
            parent_segment_start = words[i].get('segment_start_time', 0) # Needs to be passed somehow if not in word dict
            
            current_word_obj['start'] = parent_segment_start
            current_word_obj['end'] = parent_segment_start + word_len_factor
            if current_word_obj['end'] <= current_word_obj['start'] and len(current_word_obj.get('word', '')) > 0:
                current_word_obj['end'] = current_word_obj['start'] + 0.05 # min duration

        # Ensure start is not before previous word's end (if available and makes sense)
        if i > 0 and isinstance(words[i-1], dict) and 'end' in words[i-1] and current_word_obj['start'] < words[i-1]['end']:
            current_word_obj['start'] = words[i-1]['end']
        
        # Ensure end is not after next word's start (if available and makes sense)
        if i < len(words) - 1 and isinstance(words[i+1], dict) and 'start' in words[i+1] and current_word_obj['end'] > words[i+1]['start']:
            current_word_obj['end'] = words[i+1]['start']

        # Final check: end must be greater than start for non-empty words
        if len(current_word_obj.get('word','')) > 0 and current_word_obj['end'] <= current_word_obj['start']:
            current_word_obj['end'] = current_word_obj['start'] + 0.05 # Arbitrary small duration


    def process_segments(self, advanced_splitting=True):
        # (Original process_segments code remains unchanged)
        subtitles = []
        for i, segment in enumerate(self.segments):
            next_segment_start_time = self.segments[i + 1]['start'] if i + 1 < len(self.segments) else None
            
            if advanced_splitting:
                split_points = self.determine_advanced_split_points(segment, next_segment_start_time)
                subtitles.extend(self.generate_subtitles_from_split_points(segment, split_points, next_segment_start_time))
            else:
                # This part is for segment-level, not word-level.
                # We need to ensure words have timestamps if they are to be used by other methods.
                words = segment.get('words')
                if words: # Only estimate if words exist
                    for word_idx, word_data in enumerate(words):
                        if not isinstance(word_data, dict) or 'start' not in word_data or 'end' not in word_data:
                            self.estimate_timestamp_for_word(words, word_idx, next_segment_start_time)
                
                subtitles.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text']
                })
        return subtitles

    def determine_advanced_split_points(self, segment, next_segment_start_time=None):
        # (Original determine_advanced_split_points code remains unchanged)
        split_points = []
        last_split_point = 0
        char_count = 0

        # Ensure words exist and have timestamps before proceeding
        words = segment.get('words', []) # Default to empty list if not present
        if not words and 'text' in segment: # If no 'words' but 'text' exists, split text
            words = [{'word': w} for w in segment['text'].split()]
            segment['words'] = words # Store it back for consistency
        
        if not words: # If still no words, cannot determine split points
            return []

        for i, word_data in enumerate(words):
            if not isinstance(word_data, dict): # Ensure it's a dict
                 words[i] = {'word': str(word_data)} # Convert to dict
                 word_data = words[i]
            if 'start' not in word_data or 'end' not in word_data:
                self.estimate_timestamp_for_word(words, i, next_segment_start_time)

        add_space = 0 if self.lang in ['zh', 'ja'] else 1
        total_char_count = sum(len(word['word']) + add_space for word in words if isinstance(word, dict) and 'word' in word)
        char_count_after = total_char_count

        for i, word in enumerate(words):
            word_text = word.get('word','')
            word_length = len(word_text) + add_space
            char_count += word_length
            char_count_after -= word_length
            char_count_before = char_count - word_length

            if char_count >= self.max_line_length:
                midpoint = normal_round((last_split_point + i) / 2)
                if char_count_before >= self.min_char_length_splitter:
                    split_points.append(midpoint)
                    last_split_point = midpoint + 1
                    char_count = sum(len(words[j]['word']) + add_space for j in range(last_split_point, i + 1) if isinstance(words[j], dict) and 'word' in words[j])

            elif word_text.endswith(self.comma) and char_count_before >= self.min_char_length_splitter and char_count_after >= self.min_char_length_splitter:
                split_points.append(i)
                last_split_point = i + 1
                char_count = 0

            elif word_text.lower() in self.conjunctions and char_count_before >= self.min_char_length_splitter and char_count_after >= self.min_char_length_splitter:
                split_points.append(i - 1)
                last_split_point = i
                char_count = word_length
        return split_points
    
    def generate_subtitles_from_split_points(self, segment, split_points, next_start_time=None):
        # (Original generate_subtitles_from_split_points code remains unchanged)
        subtitles = []
        
        words = segment.get('words', [])
        if not words and 'text' in segment:
            words = [{'word': w} for w in segment['text'].split()] # basic split
        
        if not words: return []


        # Ensure all words are dicts and have estimated timestamps if needed
        for i, word_data in enumerate(words):
            if not isinstance(word_data, dict):
                words[i] = {'word': str(word_data)}
                word_data = words[i]
            if 'start' not in word_data or 'end' not in word_data:
                 # Pass segment start/end to estimator if available
                if 'start' in segment: word_data['segment_start_time'] = segment['start']
                self.estimate_timestamp_for_word(words, i, next_start_time)


        total_word_count = len(words)
        # Ensure segment start/end exist, default if not for duration calculation
        segment_start_time = segment.get('start', words[0].get('start', 0) if words else 0)
        segment_end_time = segment.get('end', words[-1].get('end', segment_start_time) if words else segment_start_time)
        total_time = segment_end_time - segment_start_time
        if total_time <=0 and total_word_count > 0 : # Avoid division by zero or negative duration
            total_time = sum( (w.get('end',0) - w.get('start',0)) for w in words if w.get('end',0) > w.get('start',0))
            if total_time <=0: total_time = total_word_count * 0.2 # fallback average word duration


        elapsed_time = segment_start_time
        prefix = ' ' if self.lang not in ['zh', 'ja'] else ''
        start_idx = 0

        for split_point in split_points:
            if split_point < start_idx : continue # Avoid issues if split_points are not incremental

            fragment_words = words[start_idx : split_point + 1]
            if not fragment_words: continue

            current_word_count = len(fragment_words)
            
            start_time = fragment_words[0].get('start')
            end_time = fragment_words[-1].get('end')

            # Fallback if start/end are not in word dicts (should have been estimated)
            if start_time is None or end_time is None:
                current_duration = (current_word_count / total_word_count) * total_time if total_word_count > 0 else 0
                start_time = elapsed_time
                end_time = elapsed_time + current_duration
                elapsed_time += current_duration
            
            text_to_join = [fw.get('word', '') for fw in fragment_words]
            subtitles.append({
                'start': start_time,
                'end': end_time,
                'text': prefix.join(text_to_join).strip()
            })
            
            start_idx = split_point + 1

        if start_idx < len(words):
            fragment_words = words[start_idx:]
            if not fragment_words: return subtitles # Should not happen if start_idx < len(words)

            current_word_count = len(fragment_words)
            
            start_time = fragment_words[0].get('start')
            end_time = fragment_words[-1].get('end')

            if start_time is None or end_time is None:
                current_duration = (current_word_count / total_word_count) * total_time if total_word_count > 0 else 0
                start_time = elapsed_time
                end_time = elapsed_time + current_duration # This might be segment['end']
            
            if next_start_time and end_time is not None and (next_start_time - end_time) <= 0.8:
                end_time = next_start_time
            elif end_time is None:
                end_time = segment_end_time


            text_to_join = [fw.get('word', '') for fw in fragment_words]
            subtitles.append({
                'start': start_time,
                'end': end_time if end_time is not None else segment_end_time,
                'text': prefix.join(text_to_join).strip()
            })
            
        return subtitles
    
    # --- NEW METHOD for word-by-word subtitles ---
    def get_word_by_word_subtitles(self):
        word_subtitles = []
        for i, segment in enumerate(self.segments):
            next_segment_start_time = self.segments[i + 1]['start'] if i + 1 < len(self.segments) else None
            
            if 'words' not in segment or not segment['words']:
                # If no 'words' array, but 'text' exists, try to split it.
                # This will likely result in poor timing as estimate_timestamp_for_word
                # will have to guess everything.
                if 'text' in segment and segment['text']:
                    print(f"Warning: Segment {i} has no 'words' data, splitting 'text'. Timings will be rough estimates.")
                    # Create basic word dicts from text
                    segment_words = [{'word': w, 'segment_start_time': segment.get('start',0)} for w in segment['text'].split()]
                else:
                    print(f"Warning: Segment {i} has no 'words' or 'text' data. Skipping.")
                    continue
            else:
                segment_words = segment['words']

            # Ensure all words have timestamps, estimating if necessary
            for word_idx, word_data in enumerate(segment_words):
                # Ensure word_data is a dict
                if not isinstance(word_data, dict):
                    segment_words[word_idx] = {'word': str(word_data), 'segment_start_time': segment.get('start',0)}
                    word_data = segment_words[word_idx]

                if 'start' not in word_data or 'end' not in word_data or \
                   (word_data['start'] == word_data['end'] and len(word_data.get('word','')) > 0): # Re-estimate if start==end for non-empty word
                    if 'start' in segment: # Pass segment start time for context if estimator needs it
                        word_data['segment_start_time'] = segment['start']
                    self.estimate_timestamp_for_word(segment_words, word_idx, next_segment_start_time)
            
            # Create subtitle entries for each word
            for word_data in segment_words:
                if isinstance(word_data, dict) and 'word' in word_data and \
                   'start' in word_data and 'end' in word_data and \
                   word_data['start'] < word_data['end']: # Ensure valid timing
                    word_subtitles.append({
                        'start': word_data['start'],
                        'end': word_data['end'],
                        'text': word_data['word']
                    })
                elif isinstance(word_data, dict) and 'word' in word_data : # Log if timing is problematic
                     print(f"Warning: Skipping word '{word_data['word']}' due to invalid/missing timestamps "
                           f"(start: {word_data.get('start')}, end: {word_data.get('end')}).")
        return word_subtitles

    def save(self, filename="subtitles.srt", advanced_splitting=True):
        # (Original save code remains unchanged)
        subtitles = self.process_segments(advanced_splitting)

        def write_subtitle_entry(file, idx, start_time, end_time, text):
            file.write(f"{idx}\n")
            file.write(f"{start_time} --> {end_time}\n")
            file.write(text + "\n\n")

        with open(filename, 'w', encoding='utf-8') as file:
            if self.is_vtt:
                file.write("WEBVTT\n\n")
            
            for idx, subtitle in enumerate(subtitles, 1):
                start_time_str = format_timestamp(subtitle['start'], self.is_vtt)
                end_time_str = format_timestamp(subtitle['end'], self.is_vtt)
                text = subtitle['text'].strip()
                write_subtitle_entry(file, idx, start_time_str, end_time_str, text)

        return len(subtitles)

    # --- NEW METHOD to save word-by-word subtitles ---
    def save_word_by_word(self, filename="subtitles_word_by_word.srt"):
        word_subtitles = self.get_word_by_word_subtitles()

        def write_subtitle_entry(file, idx, start_time_str, end_time_str, text):
            file.write(f"{idx}\n")
            file.write(f"{start_time_str} --> {end_time_str}\n")
            file.write(text + "\n\n")

        with open(filename, 'w', encoding='utf-8') as file:
            if self.is_vtt:
                file.write("WEBVTT\n\n")
            
            for idx, subtitle_item in enumerate(word_subtitles, 1):
                start_time = format_timestamp(subtitle_item['start'], self.is_vtt)
                end_time = format_timestamp(subtitle_item['end'], self.is_vtt)
                text = subtitle_item['text'].strip()
                if not text: # Skip empty words if any
                    continue
                write_subtitle_entry(file, idx, start_time, end_time, text)
        
        print(f"Saved {len(word_subtitles)} word-by-word subtitles to {filename}")
        return len(word_subtitles)


# --- Example Usage ---
if __name__ == '__main__':
    # Mock the conjunctions module for this example
    class MockConjunctions:
        def get_conjunctions(self, lang):
            if lang == 'en':
                return ["and", "or", "but", "so", "for", "yet", "nor"]
            return []

        def get_comma(self, lang):
            return ","

    import sys
    sys.modules['conjunctions'] = MockConjunctions() # Replace the actual module with mock

    # Sample segments data with word-level timings
    sample_segments_data = [
        {
            'start': 0.0, 'end': 3.0, 'text': "Hello world example",
            'words': [
                {'word': 'Hello', 'start': 0.1, 'end': 0.5},
                {'word': 'world', 'start': 0.6, 'end': 1.2},
                {'word': 'example', 'start': 1.5, 'end': 2.5} # Timestamps are good
            ]
        },
        {
            'start': 3.5, 'end': 6.0, 'text': "Another one here, perhaps.",
            'words': [
                {'word': 'Another', 'start': 3.6, 'end': 4.2},
                {'word': 'one'}, # Missing timestamps, will be estimated
                {'word': 'here,', 'start': 4.9, 'end': 5.2},
                {'word': 'perhaps.', 'start': 5.3, 'end': 5.8}
            ]
        },
        {
            'start': 7.0, 'end': 8.0, 'text': "Test", # No 'words' array, will try to split 'text'
        }
    ]

    # For VTT output
    processor_vtt = SubtitlesProcessor(segments=sample_segments_data, lang='en', is_vtt=True)
    num_vtt_word_subs = processor_vtt.save_word_by_word("output_word_by_word.vtt")
    print(f"Generated {num_vtt_word_subs} word-by-word VTT entries.")

    # For SRT output
    processor_srt = SubtitlesProcessor(segments=sample_segments_data, lang='en', is_vtt=False)
    num_srt_word_subs = processor_srt.save_word_by_word("output_word_by_word.srt")
    print(f"Generated {num_srt_word_subs} word-by-word SRT entries.")

    # Example of using the original line-based splitting for comparison
    # processor_srt.save("output_line_based.srt", advanced_splitting=True)
    # print("Generated line-based SRT for comparison.")