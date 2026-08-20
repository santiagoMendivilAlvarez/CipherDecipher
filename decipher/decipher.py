"""
Descifrador por ingeniería inversa.

Args:
    text (str): Texto cifrado a analizar.

Returns:
    tuple[str, str, str]: (método_detectado, texto_descifrado, clave_encontrada)
"""

from abc           import ABC, abstractmethod
import math
import random
import unicodedata
from collections   import Counter
from spellchecker  import SpellChecker


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N = len(ALPHABET)  # 26

# Frecuencias esperadas del español (en porcentaje, ordenadas A-Z).
# Fuente: análisis estándar de corpus en castellano.
FREQ_SPANISH = {
    'A': 12.53, 'B': 1.42, 'C': 4.68, 'D': 5.86, 'E': 13.68,
    'F': 0.69,  'G': 1.01, 'H': 0.70, 'I': 6.25,  'J': 0.44,
    'K': 0.02,  'L': 4.97, 'M': 3.15, 'N': 6.71,  'O': 8.68,
    'P': 2.51,  'Q': 0.88, 'R': 6.87, 'S': 7.98,  'T': 4.63,
    'U': 3.93,  'V': 0.90, 'W': 0.01, 'X': 0.22,  'Y': 0.90,
    'Z': 0.52,
}

# IC teórico del español ≈ 0.0745, del inglés ≈ 0.0667, uniforme ≈ 0.0385
IC_SPANISH_THRESHOLD = 0.055   # por encima → sustitución simple (César/Afín)
IC_UNIFORM_THRESHOLD = 0.045   # por debajo → posible Vigenère (no pedido aquí)

# Frecuencias de bigramas del español (por 10 000 bigramas).
# Fuente: corpus CREA / Almela et al.
BIGRAMS_SPANISH = {
    'DE': 343, 'EN': 298, 'ES': 284, 'LA': 276, 'OS': 233,
    'EL': 221, 'AS': 219, 'AR': 214, 'ER': 208, 'AL': 199,
    'AN': 197, 'OR': 192, 'ON': 188, 'SE': 183, 'TE': 175,
    'NT': 169, 'RA': 166, 'RE': 161, 'AD': 158, 'IC': 152,
    'UN': 150, 'IO': 147, 'TA': 144, 'CO': 142, 'IN': 140,
    'ST': 138, 'NA': 136, 'MA': 135, 'DO': 132, 'NO': 130,
    'CI': 128, 'PA': 125, 'LO': 123, 'RS': 121, 'TO': 119,
    'QU': 117, 'UE': 115, 'CA': 112, 'TI': 110, 'LI': 108,
    'LE': 106, 'RO': 105, 'SI': 102, 'ME': 100, 'NE':  98,
    'SA':  97, 'DA':  95, 'AC':  94, 'RI':  92, 'EM':  90,
    'PR':  89, 'MO':  88, 'OL':  86, 'AM':  84, 'IA':  83,
    'SO':  82, 'ND':  81, 'ID':  78, 'IL':  77, 'EC':  76,
    'OM':  75, 'OC':  74, 'UC':  73, 'TR':  72, 'ED':  71,
    'IS':  70, 'IE':  65, 'IR':  64, 'UL':  63, 'OT':  62,
    'AT':  60, 'AB':  59, 'IM':  56, 'OB':  55, 'RR':  30,
}

# Log-probabilidades precalculadas para scoring eficiente
_BIGRAM_TOTAL = sum(BIGRAMS_SPANISH.values())
_BIGRAM_LOG: dict[str, float] = {
    bg: math.log(cnt / _BIGRAM_TOTAL) for bg, cnt in BIGRAMS_SPANISH.items()
}
# Floor = log-probabilidad promedio: bigramas no vistos son tratados como "promedio",
# no se penaliza texto técnico por tener combinaciones poco comunes.
_BIGRAM_AVG   = _BIGRAM_TOTAL / len(BIGRAMS_SPANISH)
_BIGRAM_FLOOR = math.log(_BIGRAM_AVG / _BIGRAM_TOTAL)


def _bigram_log_score(text: str) -> float:
    """Log-probabilidad del texto bajo el modelo de bigramas del español. Mayor = más español."""
    return sum(_BIGRAM_LOG.get(text[i:i+2], _BIGRAM_FLOOR) for i in range(len(text) - 1))


# ─────────────────────────────────────────────
#  VALIDACIÓN NLP (DICCIONARIO ESPAÑOL)
# ─────────────────────────────────────────────

# Instancia única; carga el diccionario de español una sola vez al importar.
_es_checker = SpellChecker(language='es')


def _strip_accents(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Conjunto de palabras españolas sin acentos, calculado una vez al cargar el módulo.
# Permite reconocer 'dias' como forma de 'días', 'tambien' de 'también', etc.
_es_words_no_accent: frozenset = frozenset(
    _strip_accents(w) for w in _es_checker.word_frequency.keys()
)


def _is_spanish_word(word: str) -> bool:
    """
    True si la palabra es español válido. Maneja:
      1. Búsqueda directa en el diccionario.
      2. Formas sin tilde (dias → días, tambien → también).
      3. Plurales simples: elimina sufijo -s / -es y reintenta (buenos → bueno).
         El stem debe tener al menos 4 letras para evitar falsos positivos
         con conjunciones y artículos cortos (e.g., 'aun', 'nu').
    """
    if word in _es_checker or word in _es_words_no_accent:
        return True
    for suffix in ('es', 's'):
        stem = word[:-len(suffix)] if word.endswith(suffix) else None
        if stem and len(stem) >= 4:
            if stem in _es_checker or stem in _es_words_no_accent:
                return True
    return False


def _shift_text(text: str, k: int) -> str:
    """Aplica descifrado César (−k) conservando espacios y puntuación."""
    result = []
    for c in text.upper():
        if c in ALPHABET:
            result.append(ALPHABET[(ALPHABET.index(c) - k) % N])
        else:
            result.append(c)
    return ''.join(result)


def _afin_text(text: str, a_inv: int, b: int) -> str:
    """Aplica descifrado Afín conservando espacios y puntuación."""
    result = []
    for c in text.upper():
        if c in ALPHABET:
            result.append(ALPHABET[(a_inv * (ALPHABET.index(c) - b)) % N])
        else:
            result.append(c)
    return ''.join(result)


def word_validity_score(text: str) -> float:
    """
    Fracción de tokens separados por espacio que son palabras españolas válidas.
    Retorna 0.0 si el texto no tiene espacios (no se puede segmentar en palabras).
    """
    words = [w.lower() for w in text.split() if w.isalpha() and len(w) > 1]
    if not words:
        return 0.0
    return sum(1 for w in words if _is_spanish_word(w)) / len(words)


# ─────────────────────────────────────────────
#  UTILIDADES ESTADÍSTICAS
# ─────────────────────────────────────────────

def clean(text: str) -> str:
    """Devuelve solo letras mayúsculas del texto."""
    return "".join(c for c in text.upper() if c in ALPHABET)


def index_of_coincidence(text: str) -> float:
    """
    Calcula el Índice de Coincidencia (IC).

    El IC mide qué tan cerca está la distribución de frecuencias
    de una distribución no uniforme (idioma natural).

    IC = Σ f_i * (f_i - 1) / (n * (n - 1))

    - IC español  ≈ 0.0745
    - IC inglés   ≈ 0.0667
    - IC uniforme ≈ 0.0385  (texto con letras al azar)

    César y Afín NO cambian la distribución de frecuencias relativas
    (solo las desplazan/mezclan), por lo que conservan un IC alto.
    """
    letters = clean(text)
    n = len(letters)
    if n < 2:
        return 0.0
    counts = Counter(letters)
    numerator = sum(f * (f - 1) for f in counts.values())
    return numerator / (n * (n - 1))


def chi_squared(observed_freq: dict[str, float], expected_freq: dict[str, float]) -> float:
    """
    Chi-cuadrado entre la distribución observada y la esperada.
    Cuanto menor, más parecido al idioma de referencia.
    """
    total = sum(observed_freq.values())
    if total == 0:
        return float('inf')
    chi = 0.0
    for letter in ALPHABET:
        observed = observed_freq.get(letter, 0) / total * 100
        expected = expected_freq.get(letter, 0)
        if expected > 0:
            chi += (observed - expected) ** 2 / expected
    return chi


def letter_frequencies(text: str) -> dict[str, float]:
    """Devuelve el conteo de letras del texto limpio."""
    letters = clean(text)
    return Counter(letters)


# ─────────────────────────────────────────────
#  INTERFAZ ABSTRACTA
# ─────────────────────────────────────────────

class IDecipher(ABC):
    @abstractmethod
    def decipher(self, text: str) -> tuple[str, str]:
        """Descifra el texto y devuelve (texto_descifrado, clave_como_str)."""
        pass


# ─────────────────────────────────────────────
#  DESCIFRADOR CÉSAR
# ─────────────────────────────────────────────

class CaesarDecipher(IDecipher):
    """
    Ataque por fuerza bruta sobre los 26 desplazamientos posibles.

    Estrategia:
        Para cada k en [0..25]:
            - Aplicar C_orig = (C_cifrado - k) mod 26
            - Calcular chi² contra las frecuencias del español
        Elegir el k con menor chi².
    """

    def decipher(self, text: str) -> tuple[str, str]:
        best_ws = -1.0
        best_chi = float('inf')
        best_shift = 0
        best_plain = ""

        for k in range(N):
            # Descifrar conservando espacios para evaluar palabras reales
            plain = _shift_text(text, k)
            decoded = clean(plain)
            freq = letter_frequencies(decoded)
            chi = chi_squared(freq, FREQ_SPANISH)
            ws = word_validity_score(plain)
            # Prioridad: mayor porcentaje de palabras válidas; en empate, menor chi².
            if ws > best_ws or (ws == best_ws and chi < best_chi):
                best_ws = ws
                best_chi = chi
                best_shift = k
                best_plain = plain  # conservar espacios originales

        return best_plain, f"desplazamiento={best_shift}"


# ─────────────────────────────────────────────
#  DESCIFRADOR AFÍN
# ─────────────────────────────────────────────

class AfinDecipher(IDecipher):
    """
    El cifrado Afín usa E(x) = (a·x + b) mod 26.
    Para descifrar: D(y) = a⁻¹ · (y - b) mod 26

    Condición: 'a' debe ser coprimo con 26.
    Valores válidos de a: {1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25}

    Estrategia:
        - Fuerza bruta sobre todos los pares (a, b) válidos: 12 × 26 = 312 combinaciones.
        - Para cada par, descifrar y calcular chi².
        - Elegir el par con menor chi².
    """

    # Valores de 'a' coprimos con 26
    VALID_A = [a for a in range(1, 26) if math.gcd(a, 26) == 1]

    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """Inverso modular de a mod m usando el algoritmo extendido de Euclides."""
        for x in range(1, m):
            if (a * x) % m == 1:
                return x
        raise ValueError(f"No existe inverso de {a} mod {m}")

    def decipher(self, text: str) -> tuple[str, str]:
        best_ws = -1.0
        best_chi = float('inf')
        best_a, best_b = 1, 0
        best_plain = ""

        for a in self.VALID_A:
            a_inv = self.mod_inverse(a, N)
            for b in range(N):
                plain = _afin_text(text, a_inv, b)
                decoded = clean(plain)
                freq = letter_frequencies(decoded)
                chi = chi_squared(freq, FREQ_SPANISH)
                ws = word_validity_score(plain)
                if ws > best_ws or (ws == best_ws and chi < best_chi):
                    best_ws = ws
                    best_chi = chi
                    best_a, best_b = a, b
                    best_plain = plain  # conservar espacios originales

        return best_plain, f"a={best_a}, b={best_b}"


# ─────────────────────────────────────────────
#  DESCIFRADOR MONOALFABÉTICO
# ─────────────────────────────────────────────

class MonoalphabeticDecipher(IDecipher):
    """
    La sustitución monoalfabética tiene 26! ≈ 4×10²⁶ claves posibles.
    La fuerza bruta es inviable; usamos búsqueda local iterada.

    Estrategia (Iterated Local Search con bigramas):
        1. Mapeo inicial por frecuencia de letras.
        2. Hill-climbing con scoring de bigramas (log-probabilidad):
           los bigramas ofrecen un paisaje mucho más discriminante que los monogramas.
        3. Perturbación + re-escala (20 reinicios): aplicar 4 swaps aleatorios
           al mejor mapeo encontrado y repetir el hill-climbing para escapar
           de mínimos locales.
    """

    _RESTARTS = 60
    _PERTURB  = 8   # swaps aleatorios por reinicio

    def decipher(self, text: str) -> tuple[str, str]:
        letters = clean(text)
        freq = Counter(letters)

        spanish_by_freq = sorted(FREQ_SPANISH, key=FREQ_SPANISH.get, reverse=True)
        cipher_by_freq  = [pair[0] for pair in freq.most_common()]

        # Mapeo inicial por frecuencia de letras
        mapping: dict[str, str] = {
            cipher_by_freq[i]: spanish_by_freq[i]
            for i in range(min(len(cipher_by_freq), len(spanish_by_freq)))
        }
        for letter in ALPHABET:
            if letter not in mapping:
                mapping[letter] = letter

        def apply_mapping(m: dict[str, str]) -> str:
            return "".join(m.get(c, c) for c in letters)

        def score(m: dict[str, str]) -> float:
            # Señal primaria: palabras españolas válidas (cuando el texto tiene espacios).
            # Señal secundaria: bigramas (útil cuando no hay espacios).
            decoded_full = "".join(m.get(c, c) if c in ALPHABET else c for c in text.upper())
            ws = word_validity_score(decoded_full) * 200
            bs = _bigram_log_score(apply_mapping(m))
            return ws + bs

        def hill_climb(init: dict[str, str]) -> tuple[dict[str, str], float]:
            m = dict(init)
            improved = True
            while improved:
                improved = False
                s = score(m)
                for i in range(len(ALPHABET)):
                    for j in range(i + 1, len(ALPHABET)):
                        a, b = ALPHABET[i], ALPHABET[j]
                        m[a], m[b] = m[b], m[a]
                        ns = score(m)
                        if ns > s:   # mayor log-prob = más español
                            s = ns
                            improved = True
                        else:
                            m[a], m[b] = m[b], m[a]
            return m, s

        best_mapping, best_score = hill_climb(mapping)

        # Búsqueda local iterada: perturbar el mejor mapeo y re-escalar
        rng = random.Random()  # semilla aleatoria para mayor diversidad
        for _ in range(self._RESTARTS):
            candidate = dict(best_mapping)
            for _ in range(self._PERTURB):
                a, b = rng.sample(list(ALPHABET), 2)
                candidate[a], candidate[b] = candidate[b], candidate[a]
            m, s = hill_climb(candidate)
            if s > best_score:
                best_score = s
                best_mapping = m

        best_plain = "".join(best_mapping.get(c, c) if c in ALPHABET else c for c in text.upper())
        key_str = " ".join(f"{c}→{best_mapping[c]}" for c in sorted(best_mapping))
        return best_plain, f"tabla: {key_str}"


# ─────────────────────────────────────────────
#  CLASE PRINCIPAL: DETECCIÓN + DESPACHO
# ─────────────────────────────────────────────

class Decipher:
    """
    Orquesta la detección del cifrado y llama al descifrador correcto.

    Flujo:
        1. Calcular el Índice de Coincidencia del texto.
        2. Si IC ≥ 0.060 → sustitución simple (César o Afín).
           a. Intentar descifrar como César (k puro, a=1).
           b. Intentar descifrar como Afín  (a≠1).
           c. Comparar chi² de ambos; el menor gana.
              Si el ganador tiene a=1 → César; si a≠1 → Afín.
        3. Si IC < 0.060 → sustitución monoalfabética general.
    """

    def __init__(self):
        self._caesar = CaesarDecipher()
        self._afin   = AfinDecipher()
        self._mono   = MonoalphabeticDecipher()

    # ── API pública ──────────────────────────

    def main(self, text: str) -> tuple[str, str, str]:
        """
        Returns:
            (método, texto_descifrado, clave)
        """
        method = self._detect_pattern(text)

        if method == "Caesar":
            plain, key = self._caesar.decipher(text)
        elif method == "Afin":
            plain, key = self._afin.decipher(text)
        else:
            plain, key = self._mono.decipher(text)

        return method, plain, key

    # ── Detección ────────────────────────────

    def _detect_pattern(self, text: str) -> str:
        """
        Detecta el tipo de cifrado combinando validación NLP y estadísticas.

        Flujo:
          1. Búsqueda por diccionario: prueba todos los pares clave César/Afín
             y verifica si el texto descifrado contiene palabras españolas reales.
             Fiable incluso para palabras sueltas o textos cortos.
          2. Fallback estadístico (textos sin espacios o vocabulario no estándar):
             - IC alto  → César o Afín (permutación lineal).
             - IC bajo  → Monoalfabético general.
        """
        # 1. NLP: verificación por diccionario (prioridad sobre estadísticas).
        word_method = self._detect_by_word_lookup(text)
        if word_method is not None:
            return word_method

        # 2. Fallback estadístico.
        ic = index_of_coincidence(text)
        if ic >= IC_SPANISH_THRESHOLD:
            return self._distinguish_caesar_afin(text)
        else:
            return "Monoalphabetic"

    def _detect_by_word_lookup(self, text: str) -> str | None:
        """
        Prueba los 26 desplazamientos César y los 312 pares (a,b) Afín.
        Para cada clave descifrada calcula el porcentaje de palabras españolas
        válidas. Si el mejor supera el umbral (0.5), devuelve el tipo de cifrado;
        si no, devuelve None para ceder el turno al análisis estadístico.

        Requiere que el texto cifrado conserve los espacios originales.
        """
        best_c_ws, best_c_chi = 0.0, float('inf')
        for k in range(N):
            plain = _shift_text(text, k)
            ws = word_validity_score(plain)
            chi = chi_squared(letter_frequencies(clean(plain)), FREQ_SPANISH)
            if ws > best_c_ws or (ws == best_c_ws and chi < best_c_chi):
                best_c_ws, best_c_chi = ws, chi

        best_a_ws, best_a_chi, best_a_a = 0.0, float('inf'), 1
        for a in AfinDecipher.VALID_A:
            a_inv = AfinDecipher.mod_inverse(a, N)
            for b in range(N):
                plain = _afin_text(text, a_inv, b)
                ws = word_validity_score(plain)
                chi = chi_squared(letter_frequencies(clean(plain)), FREQ_SPANISH)
                if ws > best_a_ws or (ws == best_a_ws and chi < best_a_chi):
                    best_a_ws, best_a_chi, best_a_a = ws, chi, a

        if best_c_ws == 0.0 and best_a_ws == 0.0:
            return None  # Sin palabras reconocibles → ceder al método estadístico

        # Elegir el cifrado con mayor porcentaje de palabras válidas.
        # Si ambos tienen un score muy bajo (< 0.3) con un texto largo,
        # probablemente sea un falso positivo; ceder a estadísticas.
        if max(best_c_ws, best_a_ws) < 0.3 and len(clean(text)) > 30:
            return None

        if best_a_ws > best_c_ws:
            return "Afin"
        if best_c_ws > best_a_ws:
            return "Caesar"

        # Empate en word score.
        # Para un solo token (sin espacios) preferir César: Afín tiene 312 claves vs 26
        # de César, por lo que es ~12x más probable encontrar un falso positivo.
        # Para frases con espacios la ambigüedad es mucho menor; usar chi².
        has_spaces = ' ' in text.upper()
        if not has_spaces:
            return "Caesar"
        if best_a_a == 1:
            return "Caesar"
        return "Afin" if best_a_chi < best_c_chi * 0.97 else "Caesar"

    def _distinguish_caesar_afin(self, text: str) -> str:
        """
        Descifra con ambos métodos y elige el que produce menor chi².
        Si el mejor descifrado afín tiene a=1, es César (caso especial).

        Pero ANTES de decidir entre César/Afín, verificamos si la sustitución
        es realmente matemática o es monoalfabética arbitraria:

        Prueba de linealidad:
            Si el cifrado es Afín (incluyendo César), entonces para cualquier
            par de letras (x1, y1) y (x2, y2) se debe cumplir:
                y2 - y1 ≡ a · (x2 - x1)  (mod 26)
            Es decir, las diferencias de posición entre letras cifradas y
            originales son constantes bajo una función lineal.
            Si esta consistencia falla para muchos pares del texto, es monoalfabético.
        """
        letters = clean(text)

        # ─── Prueba de linealidad con las letras más frecuentes ───────────────
        # Tomar las 6 letras más frecuentes del español y del texto cifrado.
        # Intentar inferir 'a' y 'b' desde dos pares y verificar en los demás.
        is_affine = self._test_affine_structure(letters)
        if not is_affine:
            return "Monoalphabetic"

        # ─── Distinguir César vs Afín ────────────────────────────────────────
        best_chi_caesar = float('inf')
        for k in range(N):
            decoded = "".join(ALPHABET[(ALPHABET.index(c) - k) % N] for c in letters)
            chi = chi_squared(letter_frequencies(decoded), FREQ_SPANISH)
            if chi < best_chi_caesar:
                best_chi_caesar = chi

        best_chi_afin = float('inf')
        best_a_afin = 1
        for a in AfinDecipher.VALID_A:
            a_inv = AfinDecipher.mod_inverse(a, N)
            for b in range(N):
                decoded = "".join(
                    ALPHABET[(a_inv * (ALPHABET.index(c) - b)) % N] for c in letters
                )
                chi = chi_squared(letter_frequencies(decoded), FREQ_SPANISH)
                if chi < best_chi_afin:
                    best_chi_afin = chi
                    best_a_afin = a

        if best_a_afin == 1:
            return "Caesar"
        if best_chi_afin < best_chi_caesar * 0.97:
            return "Afin"
        return "Caesar"

    def _test_affine_structure(self, letters: str) -> bool:
        """
        Prueba si el texto tiene estructura de cifrado afín verificando
        el texto COMPLETO, no solo las letras más frecuentes.

        Un cifrado afín E(x) = (a·x + b) mod 26 implica que la misma letra
        de entrada SIEMPRE produce la misma letra de salida Y que esa función
        es lineal: conocidos dos pares (x1,y1) y (x2,y2), podemos derivar
        el (a, b) único y verificar que todo el texto lo respete.

        Estrategia:
            Para cada par (a, b) candidato (312 combinaciones):
                1. Descifrar el texto completo con E⁻¹.
                2. Contar letras distintas que NO encajan → si es 0, es afín/César.
            Si ningún (a,b) lo explica perfectamente → es monoalfabético.

        Nota: César es el caso a=1, que también pasará este test.
        """
        for a in AfinDecipher.VALID_A:
            a_inv = AfinDecipher.mod_inverse(a, N)
            for b in range(N):
                # Descifrar y verificar que el chi² sea muy bajo
                decoded = "".join(
                    ALPHABET[(a_inv * (ALPHABET.index(c) - b)) % N] for c in letters
                )
                chi = chi_squared(letter_frequencies(decoded), FREQ_SPANISH)
                # Un texto bien descifrado tendrá chi² muy bajo (< 30 para texto largo)
                # Un texto monoalfabético tendrá chi² alto para todos los (a,b)
                n = len(letters)
                # Umbral dinámico: textos largos permiten menos tolerancia
                chi_threshold = max(20.0, 150.0 / math.sqrt(n))
                if chi < chi_threshold:
                    return True
        return False



if __name__ == "__main__":
    engine = Decipher()

    # # Texto en español suficientemente largo para que el IC sea estadísticamente válido.
    # # Para textos muy cortos (<100 letras) el IC es poco confiable.
    # sample = (
    #     "EL ANALISIS DE FRECUENCIA ES UNA TECNICA FUNDAMENTAL EN CRIPTOGRAFIA "
    #     "QUE PERMITE IDENTIFICAR PATRONES EN TEXTOS CIFRADOS MEDIANTE LA "
    #     "OBSERVACION DE LA DISTRIBUCION ESTADISTICA DE LAS LETRAS EN UN IDIOMA "
    #     "EL ESPANOL TIENE UNA DISTRIBUCION CARACTERISTICA DONDE LA E LA A Y LA "
    #     "O SON LAS LETRAS MAS FRECUENTES LO QUE FACILITA EL PROCESO DE DESCIFRADO"
    # )
    # sample_clean = clean(sample)

    # # ── 1. Cifrado César con desplazamiento 3 ──
    # k = 3
    # caesar_text = "".join(ALPHABET[(ALPHABET.index(c) + k) % N] for c in sample_clean)
    # print("=" * 60)
    # print(f"[CESAR] Cifrado (k={k}): {caesar_text}")
    # method, plain, key = engine.main(caesar_text)
    # print(f"  Detectado : {method}")
    # print(f"  Descifrado: {plain}")
    # print(f"  Clave     : {key}")
    # print(f"  IC        : {index_of_coincidence(caesar_text):.4f}")

    # # ── 2. Cifrado Afín con a=5, b=8 ──
    # a_key, b_key = 5, 8
    # afin_text = "".join(
    #     ALPHABET[(a_key * ALPHABET.index(c) + b_key) % N] for c in sample_clean
    # )
    # print("=" * 60)
    # print(f"[AFIN] Cifrado (a={a_key}, b={b_key}): {afin_text}")
    # method, plain, key = engine.main(afin_text)
    # print(f"  Detectado : {method}")
    # print(f"  Descifrado: {plain}")
    # print(f"  Clave     : {key}")
    # print(f"  IC        : {index_of_coincidence(afin_text):.4f}")

    # # ── 3. Sustitución monoalfabética arbitraria ──
    # mono_map = dict(zip(ALPHABET, "QWERTYUIOPASDFGHJKLZXCVBNM"))
    # mono_text = "".join(mono_map.get(c, c) for c in sample_clean)
    # print("=" * 60)
    # print(f"[MONO] Cifrado: {mono_text}")
    # method, plain, key = engine.main(mono_text)
    # print(f"  Detectado : {method}")
    # print(f"  Descifrado (aprox): {plain[:40]}...")
    # print(f"  IC        : {index_of_coincidence(mono_text):.4f}")
    # print("=" * 60)
    cifrado = "TS QFQSOLOL RT YKTEXTFEOQ TL XFQ ZTEFOEQ YXFRQDTFZQS TF EKOHZGUKQYOQ JXT HTKDOZT ORTFZOYOEQK HQZKGFTL TF ZTBZGL EOYKQRGL"                   # "HOLA MUNDO" con César k=3
    method, plain, key = engine.main(cifrado)

    print(method)   # Caesar
    print(plain)    # HOLA MUNDO
    print(key)      # desplazamiento=3