import json
import os


def load_alphabet() -> tuple[dict[str, int], dict[int, str]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "alphabet.json")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    char_to_index = {char.upper(): int(idx) for char, idx in raw_data.items()}
    index_to_char = {int(idx): char.upper() for char, idx in raw_data.items()}

    return char_to_index, index_to_char