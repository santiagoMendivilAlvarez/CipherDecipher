from data import load_alphabet


def affine_cipher(text: str, a: int, b: int) -> str:
    char_to_index, index_to_char = load_alphabet()
    total_letters = len(char_to_index)
    cipher_text = []

    for char in text:
        upper_char = char.upper()

        if upper_char in char_to_index:
            current_index = char_to_index[upper_char]
            new_index = (a * current_index + b) % total_letters
            new_char = index_to_char[new_index]
            cipher_text.append(new_char.lower() if char.islower() else new_char)
        else:
            cipher_text.append(char)

    return "".join(cipher_text)