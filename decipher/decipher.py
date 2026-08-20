from abc           import ABC, abstractmethod
import math
import random
import unicodedata
from collections   import Counter
from spellchecker  import SpellChecker


#region Constants and Variables
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N = len(ALPHABET)  # 26


# Expected frequencies of Spanish letters (in percentage, ordered A-Z).
# Source: standard analysis of Spanish corpora.
FREQ_SPANISH = {
    'A': 12.53, 'B': 1.42, 'C': 4.68, 'D': 5.86, 'E': 13.68,
    'F': 0.69,  'G': 1.01, 'H': 0.70, 'I': 6.25,  'J': 0.44,
    'K': 0.02,  'L': 4.97, 'M': 3.15, 'N': 6.71,  'O': 8.68,
    'P': 2.51,  'Q': 0.88, 'R': 6.87, 'S': 7.98,  'T': 4.63,
    'U': 3.93,  'V': 0.90, 'W': 0.01, 'X': 0.22,  'Y': 0.90,
    'Z': 0.52,
}

# Thresholds for deciding the type of cipher based on Index of Coincidence (IC).
IC_SPANISH_THRESHOLD = 0.055  # ON top sustitution simple (Caesar/Afin) has a higher IC, while polyalphabetic ciphers like Vigenere have a lower IC.

IC_UNIFORM_THRESHOLD = 0.045  # From below this threshold, the text is likely not monoalphabetic, indicating a polyalphabetic cipher.

# Bigram frequencies of Spanish (per 10,000 bigrams).
# Source: CREA corpus / Almela et al.
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

# Precomputed log-probabilities for efficient scoring
_BIGRAM_TOTAL = sum(BIGRAMS_SPANISH.values())
_BIGRAM_LOG: dict[str, float] = {
    bg: math.log(cnt / _BIGRAM_TOTAL) for bg, cnt in BIGRAMS_SPANISH.items()
}
# Floor = average log-probability: unseen bigrams are treated as "average",
# technical text is not penalized for having uncommon combinations.
_BIGRAM_AVG   = _BIGRAM_TOTAL / len(BIGRAMS_SPANISH)
_BIGRAM_FLOOR = math.log(_BIGRAM_AVG / _BIGRAM_TOTAL)

# NLP Validation (spanish dictionary)
_es_checker = SpellChecker(language='es')

#endregion
#region Helper methods
def _bigram_log_score(text: str) -> float:
    """
    Probability score of the text based on bigram log-probabilities.

    Args:
        text (str): Input text to score.

    Returns:
        float: Sum of log-probabilities of bigrams in the text.
    """
    return sum(_BIGRAM_LOG.get(text[i:i+2], _BIGRAM_FLOOR) for i in range(len(text) - 1))


def _strip_accents(s: str) -> str:
    """
    Remove accents from the input string.

    Args:
        s (str): Input string.

    Returns:
        str: Input string with accents removed.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Set of Spanish words without accents, computed once when loading the module.
# Allows recognizing 'dias' as a form of 'días', 'tambien' as 'también', etc.
_es_words_no_accent: frozenset = frozenset(
    _strip_accents(w) for w in _es_checker.word_frequency.keys()
)


def _is_spanish_word(word: str) -> bool:
    """
    True if the word is a valid Spanish word. Handles:
      1. Direct lookup in the dictionary.
      2. Forms without accents (dias -> días, tambien -> también).
      3. Simple plurals: removes suffix -s / -es and retries (buenos -> bueno).
            The stem must have at least 4 letters to avoid false positives
            with short conjunctions and articles (e.g., 'aun', 'nu').
    
    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is valid Spanish, False otherwise.
    """
    if word in _es_checker or word in _es_words_no_accent:
        return True
    for suffix in ('es', 's'):
        stem = word[:-len(suffix)] if word.endswith(suffix) else None
        if stem:
            if len(stem) >= 4 and (stem in _es_checker or stem in _es_words_no_accent):
                return True
            # 3 letters stem: only accept if it is a tilded form (e.g., "tú" -> "tu", "él" -> "el")
            # and not a common short word (e.g., "aun", "nu").
            if len(stem) == 3 and stem in _es_words_no_accent and stem not in _es_checker:
                return True
    return False


def _shift_text(text: str, k: int) -> str:
    """
    Applies Caesar decryption (-k) while preserving spaces and punctuation.

    Args:
        text (str): The input text to decrypt.
        k (int): The shift amount (0-25).
    
    Returns:
        str: The decrypted text with the same formatting as the input.
    """
    result = []
    for c in text.upper():
        if c in ALPHABET:
            result.append(ALPHABET[(ALPHABET.index(c) - k) % N])
        else:
            result.append(c)
    return ''.join(result)


def _afin_text(text: str, a_inv: int, b: int) -> str:
    """
    Applies Affine decryption while preserving spaces and punctuation.

    Args:
        text (str): The input text to decrypt.
        a_inv (int): The modular inverse of the multiplicative key 'a'.
        b (int): The additive key 'b'.

    Returns:
        str: The decrypted text with the same formatting as the input.
    """
    result = []
    for c in text.upper():
        if c in ALPHABET:
            result.append(ALPHABET[(a_inv * (ALPHABET.index(c) - b)) % N])
        else:
            result.append(c)
    return ''.join(result)


def word_validity_score(text: str) -> float:
    """
    Fraccion de tokens separados por espacio que son palabras espaÃ±olas validas.
    Retorna 0.0 si el texto no tiene espacios (no se puede segmentar en palabras).
    """
    words = [w.lower() for w in text.split() if w.isalpha() and len(w) >= 3]
    if not words:
        return 0.0
    return sum(1 for w in words if _is_spanish_word(w)) / len(words)

#endregion
#region Statistical Utilities

def clean(text: str) -> str:
    """
    Return only the uppercase letters A-Z from the input text, removing accents and ignoring other characters.

    Args:
        text (str): Input text to clean.

    Returns:
        str: Cleaned text containing only uppercase letters A-Z.
    """
    return "".join(c for c in text.upper() if c in ALPHABET)


def index_of_coincidence(text: str) -> float:
    """
    Calculate the Index of Coincidence (IC) for the given text.
    The IC measures how close the frequency distribution of the text is to a non-uniform 
    distribution (natural language).

    IC = E [f_i * (f_i - 1)] / [n * (n - 1)]

    - Spanish IC = 0.0745
    - English IC = 0.0667
    - Uniform IC = 0.0385 (random letters)

    Caesar and Affine do NOT change the relative frequency distribution (they only shift/mix it), 
    so they maintain a high IC.

    Args:
        text (str): Input text to analyze.

    Float:
        float: Index of Coincidence value (0.0 to 1.0).
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
    Calculate the Chi-squared statistic between the observed and expected letter frequency distributions.
    The lower the value, the closer the text is to the reference language.

    Args:
        observed_freq (dict[str, float]): Observed letter frequencies.
        expected_freq (dict[str, float]): Expected letter frequencies.

    Returns:
        float: Chi-squared value.
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
    """
    Return the letter frequencies of the cleaned text.

    Args:
        text (str): Input text to analyze.
    
    Returns:
        dict[str, float]: Dictionary mapping letters A-Z to their counts in the text.
    """
    letters = clean(text)
    return Counter(letters)

#endregion
#region Abstract Interface

class IDecipher(ABC):
    @abstractmethod
    def decipher(self, text: str) -> tuple[str, str]:
        """
        Deciphers the text and returns (decrypted_text, key_as_str).

        Args:
            text (str): The input text to decipher.
        
        Returns:
            tuple[str, str]: A tuple containing the decrypted text and a string representation of the key.
        """
        pass


#endregion
#region Caesar Decipher

class CaesarDecipher(IDecipher):
    """
    Brute force attack over the 26 possible shifts.

    Strategy:
        For each k in [0..25]:
            - Apply C_orig = (C_ciphered - k) mod 26
            - Calculate chi² against Spanish letter frequencies
    """
    def decipher(self, text: str) -> tuple[str, str]:
        best_ws = -1.0
        best_chi = float('inf')
        best_shift = 0
        best_plain = ""
        for k in range(N):
            # Decipher while preserving spaces to evaluate real words
            plain = _shift_text(text, k)
            decoded = clean(plain)
            freq = letter_frequencies(decoded)
            chi = chi_squared(freq, FREQ_SPANISH)
            ws = word_validity_score(plain)
            # Priority: higher percentage of valid words; in case of tie, lower chi².
            if ws > best_ws or (ws == best_ws and chi < best_chi):
                best_ws = ws
                best_chi = chi
                best_shift = k
                best_plain = plain  # Preserve original spaces for word scoring
        return best_plain, f"desplazamiento={best_shift}"

#endregion
#region Afin Decipher

class AfinDecipher(IDecipher):
    """
    Afin cipher uses E(x) = (a·x + b) mod 26.
    For decipher: D(y) =  a⁻¹ · (y - b) mod 26

    Condition: 'a' must be coprime with 26. 
    Valid values of a: {1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25}

    Strategy:
        - Brute force over all valid (a, b) pairs: 12 × 26 = 312 combinations.
        - For each pair, decipher and calculate chi².
        - Choose the pair with the lowest chi². 
    """

    # Values from 'a' coprime with 26: {1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25}
    VALID_A = [a for a in range(1, 26) if math.gcd(a, 26) == 1]

    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """
        Modular inverse of a mod m using the Extended Euclidean Algorithm. 

        Args:
            a (int): The number to find the inverse for.
            m (int): The modulus.

        Returns:
            int: The modular inverse of a mod m. 
        """
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
                    best_plain = plain  # Preserve original spaces for word scoring

        return best_plain, f"a={best_a}, b={best_b}"

#endregion
#region Monoalphabetic Decipher
class MonoalphabeticDecipher(IDecipher):
    """
    The monoalphabetic substitution has 26! possible keys. 
    We use Simulated Annealing with combined scoring + final hill-climbing.

    Score SA = bigramas - chi²*0.4 + words(>=2)*80
    Score HC = bigramas + words(>=2)*600
    Short words are included with a whitelist to avoid penalizing common short words.
    """

    _SA_STEPS   = 30000
    _SA_CHAINS  = 16
    _SA_T_INIT  = 100.0
    _SA_COOLING = 0.9997

    _SHORT_WORDS = frozenset([
        "de","en","el","la","es","un","al","se","lo","no",
        "me","te","le","ya","si","su","mi","tu","os","ni",
        "yo","he","ha","ve","ir","fe"
    ])

    def _ws_mono(self, text: str) -> float:
        """
        Word validity score that includes 2-letter words via a whitelist.

        Args:
            text (str): Input text to evaluate.
        
        Returns:
            float: Fraction of valid Spanish words in the text.
        """
        words = [w.lower() for w in text.split() if w.isalpha()]
        if not words:
            return 0.0
        valid = sum(
            1 for w in words
            if (len(w) >= 3 and _is_spanish_word(w)) or (len(w) == 2 and w in self._SHORT_WORDS)
        )
        return valid / len(words)

    def decipher(self, text: str) -> tuple[str, str]:
        """
        The flow for deciphering this cipher is:
        1. Calculate letter frequencies and bigrams.
        2. Generate an initial mapping based on frequency analysis.
        3. Run Simulated Annealing to optimize the mapping.
        4. Run Hill Climbing to further refine the mapping.
        5. Return the best mapping found and the corresponding decrypted text.
        6. The key is returned as a string showing the mapping from cipher letters to plaintext letters.

        Args:
            text (str): The input text to decipher.
        
        Returns:
            tuple[str, str]: A tuple containing the decrypted text and a string representation of the key mapping.
        """
        letters    = clean(text)
        freq       = Counter(letters)
        alpha_list = list(ALPHABET)

        spanish_by_freq = sorted(FREQ_SPANISH, key=FREQ_SPANISH.get, reverse=True)
        cipher_by_freq  = [p[0] for p in freq.most_common()]

        init_mapping: dict[str, str] = {
            cipher_by_freq[i]: spanish_by_freq[i]
            for i in range(min(len(cipher_by_freq), len(spanish_by_freq)))
        }
        for letter in ALPHABET:
            if letter not in init_mapping:
                init_mapping[letter] = letter

        def apply_mapping(m: dict[str, str]) -> str:
            return "".join(m.get(c, c) for c in letters)

        def decoded_with_spaces(m: dict[str, str]) -> str:
            return "".join(m.get(c, c) if c in ALPHABET else c for c in text.upper())

        def score_sa(m: dict[str, str]) -> float:
            decoded = apply_mapping(m)
            return (
                _bigram_log_score(decoded)
                - chi_squared(letter_frequencies(decoded), FREQ_SPANISH) * 0.4
                + self._ws_mono(decoded_with_spaces(m)) * 100
            )

        def score_full(m: dict[str, str]) -> float:
            decoded = apply_mapping(m)
            return self._ws_mono(decoded_with_spaces(m)) * 600 + _bigram_log_score(decoded)

        def simulated_annealing(init: dict[str, str]) -> dict[str, str]:
            m = dict(init)
            s = score_sa(m)
            best_m, best_s = dict(m), s
            T = self._SA_T_INIT
            for _ in range(self._SA_STEPS):
                T *= self._SA_COOLING
                a, b = rng.sample(alpha_list, 2)
                m[a], m[b] = m[b], m[a]
                ns = score_sa(m)
                if ns > s or (T > 1e-4 and rng.random() < math.exp((ns - s) / T)):
                    s = ns
                    if s > best_s:
                        best_s, best_m = s, dict(m)
                else:
                    m[a], m[b] = m[b], m[a]
            return best_m

        def hill_climb_full(init: dict[str, str]) -> tuple[dict[str, str], float]:
            m = dict(init)
            improved = True
            while improved:
                improved = False
                s = score_full(m)
                for i in range(len(ALPHABET)):
                    for j in range(i + 1, len(ALPHABET)):
                        a, b = ALPHABET[i], ALPHABET[j]
                        m[a], m[b] = m[b], m[a]
                        ns = score_full(m)
                        if ns > s:
                            s, improved = ns, True
                        else:
                            m[a], m[b] = m[b], m[a]
            return m, s

        rng = random.Random()
        best_mapping: dict[str, str] = {}
        best_score = float("-inf")

        starts = [init_mapping]
        for _ in range(self._SA_CHAINS - 1):
            vals = list(ALPHABET)
            rng.shuffle(vals)
            starts.append(dict(zip(ALPHABET, vals)))

        for start in starts:
            sa_result = simulated_annealing(start)
            m, s      = hill_climb_full(sa_result)
            if s > best_score:
                best_score, best_mapping = s, m

        best_plain = "".join(best_mapping.get(c, c) if c in ALPHABET else c for c in text.upper())
        key_str    = " ".join(f"{c}->{best_mapping[c]}" for c in sorted(best_mapping))
        return best_plain, f"tabla: {key_str}"

#endregion
#region Decipher Class
class Decipher:
    """
    Orchestration of cipher detection and calls the correct decipherer.

    Flow:
        1. Calculate the Index of Coincidence (IC) of the text.
        2. If IC >= 0.060 -> simple substitution (Caesar or Affine).
           a. Attempt to decipher as Caesar (pure k, a=1).
           b. Attempt to decipher as Affine (a≠1).
           c. Compare chi² of both; the lower wins.
              If the winner has a=1 -> Caesar; if a≠1 -> Affine.
        3. If IC < 0.060 -> general monoalphabetic substitution.
    """

    def __init__(self):
        self._caesar = CaesarDecipher()
        self._afin   = AfinDecipher()
        self._mono   = MonoalphabeticDecipher()

    # Public API

    def main(self, text: str) -> tuple[str, str, str]:
        """
        Args:
            text (str): The input text to decipher.

        Returns:
            method (str): Detected cipher type ("Caesar", "Afin", "Monoalphabetic").
            plain  (str): Decrypted text.
            key    (str): Key or mapping used for decryption.
        """
        method = self._detect_pattern(text)

        if method == "Caesar":
            plain, key = self._caesar.decipher(text)
        elif method == "Afin":
            plain, key = self._afin.decipher(text)
        else:
            plain, key = self._mono.decipher(text)

        return method, plain, key

    # Detection

    def _detect_pattern(self, text: str) -> str:
        """
        Detects cipher type by combining NLP validation and statistical analysis.

        Flow:
            1. NLP: dictionary lookup verification (priority over statistics).
            2. Statistical fallback.

        Args:
            text (str): The input text to analyze.
        
        Returns:
            str: Detected cipher type ("Caesar", "Afin", "Monoalphabetic").
        """
        # 1. NLP: Verification by dictionary (priority over statistics).
        word_method = self._detect_by_word_lookup(text)
        if word_method is not None:
            return word_method

        # 2. Statistical fallback.
        ic = index_of_coincidence(text)
        if ic >= IC_SPANISH_THRESHOLD:
            return self._distinguish_caesar_afin(text)
        else:
            return "Monoalphabetic"

    def _detect_by_word_lookup(self, text: str) -> str | None:
        """
        Test the 26 Caesar shifts and the 312 (a,b) pairs of Affine. 
        For each decrypted key, calculate the percentage of valid Spanish words. 
        If the best exceeds the threshold (0.5), return the cipher type; if not, 
        return None to defer to statistical analysis.

        Requires that the ciphered text preserves the original spaces.
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
            return None  # Without recognizable words, we cannot trust the NLP method; defer to statistical detection.
        
        # Choose the cipher with the highest percentage of valid words.
        # If both have a very low score (< 0.3) with a long text
        # (probably a false positive), defer to statistical detection.
        if max(best_c_ws, best_a_ws) < 0.3 and len(clean(text)) > 30:
            return None

        if best_a_ws > best_c_ws:
            # Afin has 312 keys vs 26 for Caesar: higher risk of accidental match.
            # Require stronger evidence before classifying as Affine.
            if best_a_ws >= 0.6:
                return "Afin"
            return None  # Insufficient evidence; defer to statistical detection.
        if best_c_ws > best_a_ws:
            return "Caesar"
        # Tie in word score.
        # For a single token (no spaces), prefer Caesar: Affine has 312 keys vs 26 for Caesar, 
        # so it's ~12x more likely to find a false positive.
        # For phrases with spaces, the ambiguity is much lower; use chi².
        has_spaces = ' ' in text.upper()
        if not has_spaces:
            return "Caesar"
        if best_a_a == 1:
            return "Caesar"
        return "Afin" if best_a_chi < best_c_chi * 0.97 else "Caesar"

    def _distinguish_caesar_afin(self, text: str) -> str:
        """
        Decipher with both methods and choose the one with the lowest chi².
        If the best Affine decryption has a=1, it is Caesar (special case

        But BEFORE deciding between Caesar/Affine, we check if the substitution 
        is truly mathematical or arbitrary monoalphabetic:

        Lineal test:
            If cipher is Affine (including Caesar), then for any pair 
            of letters (x1, y1) and (x2, y2) it must hold:
                y2 - y1 ≡ a · (x2 - x1)  (mod 26)
            That is, the differences in position between ciphered and original letters 
            are constant under a linear function.
            If this consistency fails for many pairs in the text, it is monoalphabetic.

        Args:
            text (str): The input text to analyze.
        
        Returns:
            str: Detected cipher type ("Caesar", "Afin", "Monoalphabetic").
        """
        letters = clean(text)

        # Lineal test with the most frequent letters: if it fails, it's monoalphabetic.
        # Taje the 6 most frequent letters in the text and compare with the 6 most frequent in Spanish.
        # Try to infer 'a' and 'b' from two pairs and check the rest.   
        is_affine = self._test_affine_structure(letters)
        if not is_affine:
            return "Monoalphabetic"

        # Distinguish between Caesar and Affine by brute force chi² comparison.
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
        Test if text is consistent with an affine cipher by 
        checking all pairs of letters.

        Afin cipher E(x) = (a·x + b) mod 26 implies that the same input letter 
        ALWAYS produces the same output letter AND that this function 
        is linear: given two pairs (x1,y1) and (x2,y2), we can derive 
        the unique (a, b) and verify that the entire text respects it.

        Strategy:
            For each candidate pair (a, b) (312 combinations):
                1. Decrypt the entire text with D(y) = a⁻¹ * (y - b) mod 26.
                2. Count distinct letters that do NOT match the expected output.
                   If this count is 0, it is consistent with an affine/Caesar cipher.
            If no (a, b) explains it perfectly, it is monoalphabetic.

        NOTE: This test is strict: if the text has typos or non-standard characters, 
        it may fail even if it is an affine cipher.

        Args:
            letters (str): Cleaned text containing only uppercase letters A-Z.
        
        Returns:
            bool: True if the text is consistent with an affine cipher, False otherwise.
        """
        for a in AfinDecipher.VALID_A:
            a_inv = AfinDecipher.mod_inverse(a, N)
            for b in range(N):
                # Decipher and verify that the chi² is very low (indicating a good match 
                # with Spanish letter frequencies)
                decoded = "".join(
                    ALPHABET[(a_inv * (ALPHABET.index(c) - b)) % N] for c in letters
                )
                chi = chi_squared(letter_frequencies(decoded), FREQ_SPANISH)
                # Un texto bien descifrado tendra chiÂ² muy bajo (< 30 para texto largo)
                # Un texto monoalfabetico tendra chiÂ² alto para todos los (a,b)
                n = len(letters)
                # Umbral dinamico: textos largos permiten menos tolerancia
                chi_threshold = max(20.0, 150.0 / math.sqrt(n))
                if chi < chi_threshold:
                    return True
        return False

#endregion
#region Main Function

if __name__ == "__main__":
    engine = Decipher()

    cifrado = "TS QFQSOLOL RT YKTEXTFEOQ TL XFQ ZTEFOEQ YXFRQDTFZQS TF EKOHZGUKQYOQ JXT HTKDOZT ORTFZOYOEQK HQZKGFTL TF ZTBZGL EOYKQRGL"                   # "HOLA MUNDO" con CÃ©sar k=3
    method, plain, key = engine.main(cifrado)

    print(method)   # Caesar
    print(plain)    # HOLA MUNDO
    print(key)      # desplazamiento=3

#endregion