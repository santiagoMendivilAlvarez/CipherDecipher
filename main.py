"""
Menu - Cifrado y Decifrado: César, Monoalfabética, Afín.
"""

from math import gcd
from afin import affine_cipher
from caesarcipher import caesar_cipher
from monoalphabetic.monoalphabetic import monoalphabetic_cipher
from decipher.decipher import Decipher

_decipher_engine = Decipher()

# -- Auxiliares --
def pedir_entero(mensaje: str) -> int:
    """
    Funcion auxiliar para solicitar un numero entero valido.
    """
    while True:
        valor = input(mensaje).strip()
        try:
            return int(valor)
        except ValueError:
            print("Ingresa un numero valido.")

def pedir_valor_a() -> int:
    """
    Funcion auxiliar unica para cifrado afin para variable 'a' (si no no es decifrable:p).
    """
    VALORES_VALIDOS_A = [x for x in range(1, 26) if gcd(x, 26) == 1]

    while True:
        a = pedir_entero(f"Valor de a (debe ser coprimo con 26): ")
        if gcd(a, 26) == 1:
            return a
        print(f"'{a}' no es valido: no es coprimo con 26. Valores validos: {VALORES_VALIDOS_A}")


# -- Submenus --
def menu_cesar():
    print("\n >> Cifrado Cesar")
    texto = input("Texto a cifrar: ")
    shift = pedir_entero("Desplazamiento (k): ")

    print("Resultado:", caesar_cipher(texto, shift))
    print("--------------------------")

def menu_afin():
    print("\n >> Cifrado Afin")
    texto = input("Texto a cifrar: ")
    a = pedir_valor_a()
    b = pedir_entero("Valor de b: ")

    print("Resultado:", affine_cipher(texto, a, b))
    print("--------------------------")

def menu_monoalfabetico():
    print("\n >> Cifrado Monoalfabetico")
    texto = input("Texto a cifrar: ")
    clave = input("Palabra clave o alfabeto completo: ").strip()

    if not clave:
        print("La clave no puede estar vacia.")
        return

    resultado = monoalphabetic_cipher(texto, clave)
    print("Resultado:", resultado)
    print("--------------------------")

def menu_decifrado():
    print("\n >> Descifrado automatico")
    texto = input("Texto cifrado: ").strip()
    if not texto:
        print("Texto vacio.")
        return

    method, plain, key = _decipher_engine.main(texto)

    print("--------------------------")
    print(f"Cifrado detectado : {method}")
    print(f"Texto descifrado  : {plain}")
    print(f"Clave encontrada  : {key}")
    print("--------------------------")


# -- Menu principal --
def mostrar_menu():
    print("\n << Cifrado y Decifrado >>")
    print("1) Cesar")
    print("2) Afin")
    print("3) Monoalfabetico")
    print("4) Descifrado")
    print("0) Salir")

def main():
    opciones = {
        "1": menu_cesar,
        "2": menu_afin,
        "3": menu_monoalfabetico,
        "4": menu_decifrado,
    }

    while True:
        mostrar_menu()
        opcion = input("Opcion: ").strip()

        if opcion == "0":
            break

        accion = opciones.get(opcion)
        if accion:
            accion()
        else:
            print("Opcion invalida.")


if __name__ == "__main__":
    main()