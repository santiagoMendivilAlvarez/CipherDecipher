from data import load_alphabet


def generate_cipher_alphabet(key: str, char_to_index: dict[str, int], index_to_char: dict[int, str]) -> str:

    seen = set()
    key_unique = []

    # 1. Quita los caracteres unicos validos de la clave ingresada
    for char in key.upper():
        if char in char_to_index and char not in seen:
            seen.add(char)
            key_unique.append(char)

    # 2. Completa con los caracteres faltantes del alfabeto base
    total_letters = len(char_to_index)
    for idx in range(total_letters):
        char = index_to_char[idx]
        if char not in seen:
            seen.add(char)
            key_unique.append(char)

    return "".join(key_unique)

def monoalphabetic_cipher(text: str, key: str) -> str:

    char_to_index, index_to_char = load_alphabet()
    cipher_alphabet = generate_cipher_alphabet(key, char_to_index, index_to_char)
    cipher_text = []

    for char in text:
        upper_char = char.upper()

        if upper_char in char_to_index:
            # Tomar el indice del caracter original y mapear al alfabeto permutado
            original_index = char_to_index[upper_char]
            new_char = cipher_alphabet[original_index]
            cipher_text.append(new_char.lower() if char.islower() else new_char)
        else:
            # Espacios, numeros o signos de puntuacion no se modifican
            cipher_text.append(char)

    return "".join(cipher_text)