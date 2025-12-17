import os
import pandas as pd
import re
import numpy as np

class FencingPhraseLoader:
    def __init__(self, root_dir, fps=15.0):
        self.root_dir = root_dir
        self.fps = fps

    def parse_txt_file(self, txt_path):
        """Parses the text file for timeline events, metadata, and the winner."""
        events = []
        winner = None
        metadata = {
            "lockout_windows": [],
            "start_time": None,
            "end_time": None,
            "raw_lines": 0,
        }

        # Regex patterns
        time_pattern = re.compile(r"^\s*(\d+\.\d+)s\s*\|\s*(.*)")
        lockout_pattern = re.compile(r"Lockout period started \((\d+\.\d+)s window\)")
        winner_pattern = re.compile(r"Confirmed result winner:\s*(.*)")
        manual_winner_pattern = re.compile(r"Manual selection winner:\s*(.*)")

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            metadata["raw_lines"] += 1

            # Parse timeline events with timestamps
            match = time_pattern.match(line)
            if match:
                time_sec = float(match.group(1))
                description = match.group(2).strip()
                frame_idx = int(round(time_sec * self.fps))

                event = {
                    "time": time_sec,
                    "frame": frame_idx,
                    "description": description,
                    "type": "unknown",
                    "attributes": {}
                }

                desc_lower = description.lower()
                if "phrase recording started" in desc_lower:
                    event["type"] = "start"
                    metadata["start_time"] = time_sec
                elif "phrase recording ended" in desc_lower:
                    event["type"] = "end"
                    metadata["end_time"] = time_sec
                elif "blade-to-blade contact" in desc_lower:
                    event["type"] = "blade_contact"
                    event["attributes"]["result"] = "off-target" if "off-target" in desc_lower else None
                elif "hit" in desc_lower:
                    # Determine which fencer scored
                    scorer = None
                    target = None
                    double_hit = "simultaneous" in desc_lower
                    if double_hit:
                        event["type"] = "double_hit"
                    else:
                        event["type"] = "hit"
                        if "left scores" in desc_lower or "fencer 2 scores" in desc_lower:
                            scorer = "left"
                            target = "right"
                        elif "right scores" in desc_lower or "fencer 1 scores" in desc_lower:
                            scorer = "right"
                            target = "left"
                    event["attributes"].update({
                        "scorer": scorer,
                        "target": target,
                        "double_hit": double_hit,
                    })
                elif "lockout period started" in desc_lower:
                    event["type"] = "lockout_start"
                    window_match = lockout_pattern.search(description)
                    if window_match:
                        window = float(window_match.group(1))
                        event["attributes"]["window"] = window
                        metadata["lockout_windows"].append({
                            "time": time_sec,
                            "frame": frame_idx,
                            "window": window,
                        })
                elif "lockout active" in desc_lower:
                    event["type"] = "lockout_notice"

                events.append(event)

            # Parse Winner from textual summaries (no timestamps)
            w_match = winner_pattern.search(line)
            if not w_match:
                w_match = manual_winner_pattern.search(line)
            if not w_match:
                # Simple "Winner: Left" check
                simple_line = line.lower()
                if "winner:" in simple_line:
                    if "winner: left" in simple_line or "winner: fencer 2" in simple_line:
                        winner = "left"
                    elif "winner: right" in simple_line or "winner: fencer 1" in simple_line:
                        winner = "right"
                    elif "winner: simultaneous" in simple_line:
                        winner = "simultaneous"
                continue

            if w_match:
                winner_str = w_match.group(1).lower()
                if "left" in winner_str or "fencer 2" in winner_str:
                    winner = "left"
                elif "right" in winner_str or "fencer 1" in winner_str:
                    winner = "right"
                elif "simultaneous" in winner_str:
                    winner = "simultaneous"

        return events, winner, metadata

    def load_phrase(self, phrase_folder):
        """Loads a single phrase folder."""
        
        # Find required files
        files = os.listdir(phrase_folder)
        excel_file = next((f for f in files if f.endswith('_compressed_keypoints.xlsx')), None)
        txt_file = next((f for f in files if f.endswith('.txt') and not f.startswith('requirements')), None) # Avoid random txts
        
        # Fallback for txt file: usually named same as folder prefix or phraseXX.txt
        if not txt_file:
             # Try finding any txt file that looks like a log
             candidates = [f for f in files if f.endswith('.txt')]
             if candidates:
                 txt_file = candidates[0]

        if not excel_file or not txt_file:
            print(f"Skipping {phrase_folder}: Missing excel or txt")
            return None

        # Load Excel
        excel_path = os.path.join(phrase_folder, excel_file)
        try:
            # Load all sheets
            xls = pd.ExcelFile(excel_path)
            left_x = pd.read_excel(xls, 'left_x')
            left_y = pd.read_excel(xls, 'left_y')
            right_x = pd.read_excel(xls, 'right_x')
            right_y = pd.read_excel(xls, 'right_y')
            
            # Basic validation
            if len(left_x) != len(right_x):
                print(f"Warning {phrase_folder}: Frame count mismatch between sheets.")
                
        except Exception as e:
            print(f"Error reading Excel {phrase_folder}: {e}")
            return None

        # Load Text
        txt_path = os.path.join(phrase_folder, txt_file)
        events, winner, metadata = self.parse_txt_file(txt_path)

        if winner is None:
            print(f"Skipping {phrase_folder}: winner missing in TXT log")
            return None

        return {
            "folder": os.path.basename(phrase_folder),
            "left_x": left_x,
            "left_y": left_y,
            "right_x": right_x,
            "right_y": right_y,
            "events": events,
            "winner": winner,
            "metadata": metadata,
        }

    def load_all(self):
        """Iterates through root_dir and loads all valid phrases."""
        data = []
        subdirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for subdir in subdirs:
            phrase_data = self.load_phrase(subdir)
            if phrase_data:
                data.append(phrase_data)
                
        return data

# Example Usage (for debugging)
if __name__ == "__main__":
    loader = FencingPhraseLoader("/workspace/training_data")
    all_phrases = loader.load_all()
    print(f"Loaded {len(all_phrases)} phrases.")
    if len(all_phrases) > 0:
        print("Sample Phrase Folder:", all_phrases[0]['folder'])
        print("Sample Phrase Winner:", all_phrases[0]['winner'])
        print("Sample Events:", all_phrases[0]['events'][:3])
