# command_matcher.py
import csv
from rapidfuzz import process, fuzz

import csv

def load_commands(filename="commands.csv"):
    commands = {}
    print(f"Loading from {filename} ...")

    with open(filename, "r", encoding="utf-8-sig", newline='') as f:
        reader = csv.DictReader(f)
        print("CSV fieldnames:", reader.fieldnames)  # debug
        for row in reader:
            if not row.get("command") or not row.get("phrase"):
                continue
            cmd = row["command"].strip()
            phrase = row["phrase"].strip().lower()
            commands.setdefault(cmd, []).append(phrase)

    print("Loaded commands:", commands)
    return commands

def find_best_match(text, commands, threshold=70):
    all_phrases = {phrase: cmd for cmd, phrases in commands.items() for phrase in phrases}
    result = process.extractOne(text.lower(), all_phrases.keys(), scorer=fuzz.token_sort_ratio)

    if result:
        best_phrase, score, _ = result
        if score >= threshold:
            return all_phrases[best_phrase], score

    return None  # ensures you can safely check before unpacking
