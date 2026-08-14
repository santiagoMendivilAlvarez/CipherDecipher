"""
Args:
    Text to decipher: str

Returns:
    Method of cipher: str
    New text deciphered: str
"""
from abc import ABC, abstractmethod


class IDecipher(ABC):
    @abstractmethod
    def decipher(self, text: str) -> tuple[str, str]:
        pass


class Decipher:
    def __init__(self, decipher: IDecipher):
        self.decipher = decipher

    def caesar_decipher(self, text: str) -> tuple[str, str]: # Bere
        return self.decipher.decipher(text)

    def monoalphabetic_decipher(self, text: str) -> tuple[str, str]: # Adan
        return self.decipher.decipher(text)

    def afin_decipher(self, text: str) -> tuple[str, str]: # Liz
        return self.decipher.decipher(text)