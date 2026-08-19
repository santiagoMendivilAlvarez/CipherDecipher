import json
import os


def load_alphabet() -> tuple[dict[str, int], dict[int, str]]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "alphabet.json")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    char_to_index = {char.upper(): int(idx) for char, idx in raw_data.items()}
    index_to_char = {int(idx): char.upper() for char, idx in raw_data.items()}

    return char_to_index, index_to_char


def caesar_cipher(text: str, shift: int) -> str:
    char_to_index, index_to_char = load_alphabet()
    total_letters = len(char_to_index)
    cipher_text = []

    for char in text:
        upper_char = char.upper()

        if upper_char in char_to_index:
            current_index = char_to_index[upper_char]
            new_index = (current_index + shift) % total_letters
            new_char = index_to_char[new_index]
            cipher_text.append(new_char.lower() if char.islower() else new_char)
        else:
            cipher_text.append(char)

    return "".join(cipher_text)