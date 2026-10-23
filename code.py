import os
import re
import nltk

for resource in ['cmudict', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import cmudict

try:
    CMU_DICT = cmudict.dict()
except LookupError:
    nltk.download('cmudict', quiet=True)
    CMU_DICT = cmudict.dict()

print("Initializing phoneme, rhyme indices, phonotactic onset engine, and IPA Reform engine...")

ARPABET_MAP = {
    'IY': 'ee', 'IH': 'i', 'EH': 'e', 'AE': 'a',
    'AA': 'aa', 'AO': 'aw', 'UH': 'uu',
    'UW': 'oo', 'EY': 'ay', 'AY': 'igh',
    'OY': 'oy', 'AW': 'ow', 'OW': 'oh',

    'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'dh', 'F': 'f',
    'G': 'g', 'HH': 'h', 'JH': 'j', 'K': 'k', 'L': 'l',
    'M': 'm', 'N': 'n', 'NG': 'ng', 'P': 'p', 'R': 'r',
    'S': 's', 'SH': 'sh', 'T': 't', 'TH': 'th', 'V': 'v',
    'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh'
}

ARPABET_TO_IPA_REFORM = {
    'IY': 'i', 'IH': 'ɪ', 'EH': 'ɛ', 'AE': 'æ', 'AH': 'ɐ', 'AA': 'ɑ', 'AO': 'ɔ',
    'UH': 'ʊ', 'UW': 'ʉ', 'ER': 'ɜɹ', 'EY': 'eɪ', 'AY': 'aɪ', 'OY': 'ɔɪ', 'AW': 'æʊ', 'OW': 'oʊ',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'JH': 'dʒ',
    'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'ɹ', 'S': 's',
    'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}

VOWEL_SET = {'IY', 'IH', 'EH', 'AE', 'AH', 'AA', 'AO', 'UH', 'UW', 'ER', 'EY', 'AY', 'OY', 'AW', 'OW'}
# True short/lax vowels that prefer closed syllables under stress.
# AO (and AA) are usually long /ɔː ɑː/ in GenAm for words like coffee, caught,
# so they are deliberately excluded – otherwise we get KAWF-ee instead of KAW-fee.
LAX_VOWELS = {'IH', 'EH', 'AE', 'AH', 'UH'}

# Expanded valid onsets (English phonotactics)
VALID_ONSETS = {
    ('B', 'R'), ('B', 'L'), ('D', 'R'), ('F', 'R'), ('F', 'L'),
    ('G', 'R'), ('G', 'L'), ('K', 'R'), ('K', 'L'), ('P', 'R'),
    ('P', 'L'), ('T', 'R'), ('S', 'T'), ('S', 'P'), ('S', 'K'),
    ('S', 'M'), ('S', 'N'), ('S', 'L'), ('S', 'W'), ('S', 'F'),
    ('SH', 'R'), ('TH', 'R'), ('TH', 'W'),
    ('M', 'Y'), ('N', 'Y'), ('L', 'Y'), ('V', 'Y'), ('Z', 'Y'),
    ('S', 'P', 'R'), ('S', 'P', 'L'), ('S', 'T', 'R'), ('S', 'K', 'R'),
    ('S', 'K', 'W'), ('S', 'K', 'Y'), ('S', 'P', 'Y'), ('S', 'T', 'Y'),
    ('K', 'W'), ('G', 'W'), ('T', 'W'), ('D', 'W'),
    ('K', 'Y'), ('P', 'Y'), ('B', 'Y'), ('F', 'Y'), ('H', 'Y'),
    ('D', 'Y'), ('T', 'Y'), ('S', 'Y'),
}


def is_valid_onset(cluster):
    """Return True if the consonant sequence is a legal English onset."""
    if not cluster:
        return True  # empty onset is fine
    if len(cluster) == 1:
        return cluster[0] != 'NG'  # ŋ never occurs syllable-initially
    return tuple(cluster) in VALID_ONSETS


def _normalize_apostrophe(w):
    """Normalize any apostrophe / quote character to ASCII '."""
    return (w.replace('\u2019', "'")   # right single quotation mark
             .replace('\u2018', "'")   # left single quotation mark
             .replace('\u02bc', "'")   # modifier letter apostrophe
             .replace('`', "'"))


# Contractions / clitics that NLTK often splits off the preceding word
_CONTRACTION_CLITICS = {
    "n't", "'t", "'s", "'re", "'ve", "'ll", "'d", "'m",
    "n't", "'t", "'s", "'re", "'ve", "'ll", "'d", "'m",  # curly variants (normalized)
}


def _remerge_contractions(tokens):
    """
    Re-join tokens that NLTK split, e.g. ['did', "n't"] → ["didn't"].
    Also normalizes the apostrophe to ASCII '.

    Only merges when the second token itself looks like a real clitic
    (exactly n't, or starts with an apostrophe).  Bare "t" / "s"
    after a space (e.g. "didn t") is left alone.
    """
    if not tokens:
        return tokens
    merged = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            nxt_norm = _normalize_apostrophe(nxt.lower())
            # Real clitic only: exact "n't" or apostrophe-clitic in the set
            is_clitic = (
                nxt_norm == "n't"
                or (nxt_norm.startswith("'") and nxt_norm in _CONTRACTION_CLITICS)
            )
            if is_clitic:
                if nxt_norm == "n't":
                    # Critical: join as "didn't", NOT "did'n't"
                    merged.append(tok + "n't")
                else:
                    # "'s", "'re", "'ve", "'ll", "'d", "'m", "'t"
                    clitic = _normalize_apostrophe(nxt)
                    if not clitic.startswith("'"):
                        clitic = "'" + clitic.lstrip("'")
                    merged.append(tok + clitic)
                i += 2
                continue
        merged.append(tok)
        i += 1
    return merged


def analyze_phrase_pos(phrase_str):
    try:
        tokens = nltk.word_tokenize(phrase_str)
        tokens = _remerge_contractions(tokens)
        tagged = nltk.pos_tag(tokens)
    except Exception:
        tokens = phrase_str.split()
        tagged = [(t, 'O') for t in tokens]
    return tagged


def _lookup_cmu(w_lower):
    """Try several normalized forms so apostrophes and hyphens are found."""
    w = _normalize_apostrophe(w_lower)
    candidates = [
        w_lower,
        w,
        w.replace("'", ""),                 # dont, its, oclock
        w.replace("-", " "),                # ice-cream → ice cream
        w.replace(" ", "-"),                # ice cream → ice-cream
        w.replace("'", "").replace("-", " "),
        w.replace("'", "").replace(" ", "-"),
    ]
    seen = set()
    for key in candidates:
        if key and key not in seen:
            seen.add(key)
            if key in CMU_DICT:
                return CMU_DICT[key]
    return None


def get_phrase_phonemes_with_pos(phrase_str):
    tagged_words = analyze_phrase_pos(phrase_str)
    word_results = []

    for word, pos in tagged_words:
        w_lower = word.lower()
        variants = _lookup_cmu(w_lower)

        if variants is None and w_lower.isalnum():
            # crude letter-by-letter fallback (keeps the pipeline alive)
            letter_vars = []
            for char in w_lower:
                if char in CMU_DICT:
                    letter_vars.append(CMU_DICT[char][0])
            if letter_vars:
                flat_phones = []
                for lv in letter_vars:
                    if flat_phones:
                        flat_phones.append('BOUNDARY')
                    flat_phones.extend(lv)
                word_results.append((word, pos, [flat_phones]))
                continue

        if variants is None:
            variants = [[w_lower.upper()]]
        word_results.append((word, pos, variants))

    return word_results


def find_compound_split(word, phones):
    """
    Only split a hyphenated (or space-containing) token when the *whole*
    form is NOT already in CMU.

    Desired behaviour:
      "didn't"     → keep as one word (CMU has "didn't")
      "ice-cream"  → keep as one lexicalised entry if CMU has it
      "water bottle" → already two tokens from the tokenizer

    If the full form is missing but the parts exist, fall back to
    concatenating the parts with a BOUNDARY (best-effort).
    """
    w_lower = _normalize_apostrophe(word.lower())

    # Prefer the lexicalised whole-form when CMU has it
    if _lookup_cmu(w_lower) is not None:
        return None

    for sep in ('-', ' '):
        if sep not in w_lower:
            continue
        parts = [p for p in w_lower.split(sep) if p]
        if len(parts) < 2:
            continue
        part_phones = []
        ok = True
        for p in parts:
            vars_ = _lookup_cmu(p)
            if not vars_:
                ok = False
                break
            part_phones.append(vars_[0])
        if not ok:
            continue
        combined = []
        for idx, ph in enumerate(part_phones):
            if idx > 0:
                combined.append('BOUNDARY')
            combined.extend(ph)
        return combined
    return None

def generate_reformed_ipa_tokens(phones):
    """
    Phonemic IPA only — no allophonic rules.

    Removed: aspiration, flapping, nasal flap, dark-L, TR/DR/STR
    coalescence, glottal stop, geminate lengthening, centralized
    unstressed vowels, etc.

    Kept: direct ARPABET→IPA map, primary-stress acute accent,
    syllable dots between adjacent vowels, BOUNDARY as space.
    """
    n = len(phones)
    reformed_tokens = []
    i = 0

    while i < n:
        ph = phones[i]
        if ph == 'BOUNDARY':
            reformed_tokens.append({
                'sym': '|', 'is_vowel': False,
                'is_syllabic_or_rhotic': False, 'base': 'BOUNDARY'
            })
            i += 1
            continue

        if ph in {'INTRUSIVE_W', 'INTRUSIVE_Y'}:
            # Linking glides between words — keep as optional markers
            ipa_sym = '(w)' if ph == 'INTRUSIVE_W' else '(y)'
            reformed_tokens.append({
                'sym': ipa_sym, 'is_vowel': False,
                'is_syllabic_or_rhotic': False, 'base': 'INTRUSIVE'
            })
            i += 1
            continue

        stress = ph[-1] if ph[-1].isdigit() else ''
        base = ph[:-1] if stress else ph

        next_ph = phones[i + 1] if i + 1 < n else None
        next_base = (next_ph[:-1] if next_ph and next_ph[-1].isdigit() else next_ph) if next_ph else None
        next2_ph = phones[i + 2] if i + 2 < n else None

        # Word-final (or before inflectional -s/-z/-d) syllabic L/N/M:
        # AH0 + L/N/M → superscript schwa + consonant
        #   bottle /bɑtᵊl/, buttons /bʌtᵊnz/, rhythm /rɪðᵊm/
        next2_base_ipa = (next2_ph[:-1] if next2_ph and next2_ph[-1].isdigit() else next2_ph) if next2_ph else None
        at_end_ipa = next2_ph is None or (
            next2_base_ipa in {'Z', 'S', 'D'} and i + 3 >= n
        )
        if (base == 'AH' and stress == '0' and next_base in {'L', 'N', 'M'}
                and at_end_ipa):
            cons = ARPABET_TO_IPA_REFORM.get(next_base, next_base.lower())
            reformed_tokens.append({
                'sym': '\u1d4a' + cons,  # ᵊ + l/n/m
                'is_vowel': True,
                'is_syllabic_or_rhotic': True,
                'base': f'AH+{next_base}'
            })
            i += 2
            continue

        # Phonemic symbol
        if base == 'AH':
            sym = 'ə'
        elif base == 'ER':
            sym = 'ɜɹ' if stress in {'1', '2'} else 'əɹ'
        else:
            sym = ARPABET_TO_IPA_REFORM.get(base, base.lower())

        # Primary stress mark on the first letter of the vowel
        if stress == '1' and base in VOWEL_SET and sym:
            sym = sym[0] + '\u0301' + sym[1:]

        is_vowel = base in VOWEL_SET
        is_rhotic = base == 'ER' or (isinstance(sym, str) and 'ɹ' in sym)

        reformed_tokens.append({
            'sym': sym,
            'is_vowel': is_vowel,
            'is_syllabic_or_rhotic': is_rhotic,
            'base': base
        })
        i += 1

    return reformed_tokens


def get_ipa_transcription(phones):
    tokens = generate_reformed_ipa_tokens(phones)
    formatted_ipa = []
    for idx, tok in enumerate(tokens):
        sym = tok['sym']
        if sym == '|':
            formatted_ipa.append(" ")
            continue

        is_vowel = tok['is_vowel']
        is_syllabic_or_rhotic = tok.get('is_syllabic_or_rhotic', False)

        if idx > 0:
            prev_tok = tokens[idx - 1]
            prev_is_vowel = prev_tok['is_vowel']
            prev_is_syllabic_or_rhotic = prev_tok.get('is_syllabic_or_rhotic', False)

            if prev_is_vowel and is_vowel and not prev_tok['sym'].startswith('('):
                if not (prev_is_syllabic_or_rhotic or is_syllabic_or_rhotic):
                    formatted_ipa.append(".")

        formatted_ipa.append(sym)

    return "/" + "".join(formatted_ipa).replace(" .", ".").replace(". ", ".") + "/"


def get_phoneme_breakdown(phones):
    tokens = generate_reformed_ipa_tokens(phones)
    breakdowns = []
    for tok in tokens:
        if tok['base'] == 'BOUNDARY':
            continue
        desc = f"{tok['base']} (IPA: {tok['sym']})"
        breakdowns.append(desc)
    return breakdowns


def apply_custom_rules(phonemes_with_stress):
    translated_tokens = []
    i = 0
    n = len(phonemes_with_stress)

    while i < n:
        ph_raw = phonemes_with_stress[i]
        if ph_raw == 'BOUNDARY':
            translated_tokens.append({'text': ' ', 'is_vowel': False, 'stress': None, 'base_ph': 'BOUNDARY', 'raw_ph': ph_raw})
            i += 1
            continue
        if ph_raw == 'INTRUSIVE_W':
            translated_tokens.append({'text': '(w)', 'is_vowel': False, 'stress': None, 'base_ph': 'W', 'raw_ph': ph_raw})
            i += 1
            continue
        if ph_raw == 'INTRUSIVE_Y':
            translated_tokens.append({'text': '(y)', 'is_vowel': False, 'stress': None, 'base_ph': 'Y', 'raw_ph': ph_raw})
            i += 1
            continue

        stress = ph_raw[-1] if ph_raw[-1].isdigit() else None
        base_ph = ph_raw[:-1] if stress else ph_raw

        next_ph = phonemes_with_stress[i + 1] if i + 1 < n else None
        next_base = next_ph[:-1] if next_ph and next_ph[-1].isdigit() else next_ph

        if base_ph == 'HH' and next_base == 'W':
            # what → WHUHT (not HWUHT)
            translated_tokens.append({'text': 'wh', 'is_vowel': False, 'stress': None, 'base_ph': 'HH+W', 'raw_ph': ph_raw})
            i += 2
            continue

        # Schwa + L/N/M when liquid/nasal is syllable-final (not before a vowel).
        # Word-final or before inflectional -s/-z/-d → (uh)l / (uh)n / (uh)m
        #   bottle → BAA-t(uh)l,  buttons → BUHT-(uh)nz,  professional → …n(uh)l
        # Mid-word still "uhl"/"uhn"/"uhm" without brackets.
        # Skip after Y (million) or a pure vowel/diphthong (loyal).
        if base_ph == 'AH' and stress == '0' and next_base in {'L', 'N', 'M'}:
            next2_ph = phonemes_with_stress[i + 2] if i + 2 < n else None
            next2_base = (next2_ph[:-1] if next2_ph and next2_ph[-1].isdigit() else next2_ph) if next2_ph else None
            if next2_base is None or next2_base not in VOWEL_SET:
                prev_raw = phonemes_with_stress[i - 1] if i > 0 else None
                prev_base = (prev_raw[:-1] if prev_raw and prev_raw[-1].isdigit() else prev_raw) if prev_raw else None
                # final, or only an inflectional consonant remains after L/N/M
                at_end = next2_ph is None or (
                    next2_base in {'Z', 'S', 'D'} and i + 3 >= n
                )
                after_y_or_vowel = (
                    prev_base in VOWEL_SET or prev_base in {'Y', 'W'} or
                    (prev_base and prev_base.endswith('+R'))
                )
                if at_end and not after_y_or_vowel:
                    cluster_text = {'L': '(uh)l', 'N': '(uh)n', 'M': '(uh)m'}[next_base]
                else:
                    cluster_text = {'L': 'uhl', 'N': 'uhn', 'M': 'uhm'}[next_base]
                translated_tokens.append({
                    'text': cluster_text, 'is_vowel': True, 'stress': stress,
                    'base_ph': f"AH+{next_base}", 'raw_ph': ph_raw
                })
                i += 2
                continue

        # Skip a lone R after ER only when it is NOT a new syllable onset
        # (i.e. not followed by a vowel).  Keeps "overrun" → OH-vuhr-run
        # while still collapsing true ER+R geminates before a consonant/end.
        if base_ph == 'R' and i > 0:
            prev_raw = phonemes_with_stress[i - 1]
            if prev_raw not in {'INTRUSIVE_W', 'INTRUSIVE_Y', 'BOUNDARY'}:
                prev_base = prev_raw[:-1] if prev_raw[-1].isdigit() else prev_raw
                if prev_base == 'ER':
                    # keep R if a vowel follows (onset of next syllable)
                    if next_base is None or next_base not in VOWEL_SET:
                        i += 1
                        continue

        # EH + ER → eh-er (two syllables, not merged)
        if base_ph == 'EH' and next_base == 'ER':
            translated_tokens.append({
                'text': 'eh', 'is_vowel': True, 'stress': stress,
                'base_ph': 'EH', 'raw_ph': ph_raw
            })
            # ER handled on next iteration as "er"
            i += 1
            continue

        # Vowel + R → r-coloured vowel, EXCEPT when R is the onset of a
        # stressed syllable (direct → digh-RECT, not dire-EKT).
        # Still merge when R is final or before an unstressed vowel
        # (care → KEHR, marry → MEHR-ee).
        if next_base == 'R':
            next2_ph = phonemes_with_stress[i + 2] if i + 2 < n else None
            next2_base = (next2_ph[:-1] if next2_ph and next2_ph[-1].isdigit() else next2_ph) if next2_ph else None
            next2_stress = next2_ph[-1] if next2_ph and next2_ph[-1].isdigit() else ''
            r_is_onset_of_stress = (
                next2_base in VOWEL_SET and next2_stress in {'1', '2'}
            )
            if not r_is_onset_of_stress:
                rhotic_map = {
                    'EH': 'ehr', 'IH': 'eer', 'UH': 'oor', 'AW': 'our',
                    'AY': 'ire', 'AO': 'or', 'AA': 'ar', 'AE': 'arr'
                }
                if base_ph in rhotic_map:
                    translated_tokens.append({
                        'text': rhotic_map[base_ph], 'is_vowel': True,
                        'stress': stress, 'base_ph': base_ph + '+R', 'raw_ph': ph_raw
                    })
                    i += 2
                    continue

        # Homogenised weak/strong:
        #   AH (strut & comma) → always "uh"
        #   ER (nurse & letter) → always "er"
        if base_ph == 'AH':
            respell = 'uh'
        elif base_ph == 'IH':
            respell = 'i'
        elif base_ph == 'ER':
            respell = 'er'
        elif base_ph == 'IY' and stress == '0':
            respell = 'ee'
        elif base_ph == 'UW':
            respell = 'oo'   # always "oo" (never "ooh")
        else:
            respell = ARPABET_MAP.get(base_ph, base_ph.lower())

        is_vowel = base_ph in VOWEL_SET

        translated_tokens.append({
            'text': respell,
            'is_vowel': is_vowel,
            'stress': stress if is_vowel else None,
            'base_ph': base_ph,
            'raw_ph': ph_raw
        })
        i += 1

    return translated_tokens


def syllabify_tokens(tokens, original_word=""):
    """
    Improved English syllabification.

    Strategy (in priority order):
    1. Honour explicit BOUNDARY markers (compounds / multi-word).
    2. Geminate consonants are split (at-tac-tive style).
    3. Stressed LAX vowels attract one following consonant when the
       remainder forms a legal onset  →  BED-room, HAP-ee, AP-uhl.
       Orthography can block this (february → FE-byuh / FE-broo).
    4. Prefer nasal+s/z as coda (trans-LAYT) and S as coda when the
       remainder is a legal onset (IGHS-kreem for ice-cream).
    5. Maximal Onset Principle for everything else.
    """
    word_l = (original_word or "").lower()
    # ---------- pass 1: group by vowels, honour BOUNDARY ----------
    syllables = []
    current_syl = []

    for token in tokens:
        if token['base_ph'] == 'BOUNDARY':
            if current_syl:
                if syllables:
                    syllables[-1].extend(current_syl)
                else:
                    syllables.append(current_syl)
                current_syl = []
            syllables.append([token])
            continue

        current_syl.append(token)
        if token['is_vowel']:
            syllables.append(current_syl)
            current_syl = []

    if current_syl:
        if syllables:
            syllables[-1].extend(current_syl)
        else:
            syllables.append(current_syl)

    # ---------- pass 2: redistribute intervocalic consonants ----------
    for i in range(len(syllables) - 1):
        curr_syl = syllables[i]
        next_syl = syllables[i + 1]

        # never touch BOUNDARY syllables
        if any(t['base_ph'] == 'BOUNDARY' for t in curr_syl + next_syl):
            continue

        curr_vowel_idx = next((idx for idx, t in enumerate(curr_syl) if t['is_vowel']), -1)
        next_vowel_idx = next((idx for idx, t in enumerate(next_syl) if t['is_vowel']), -1)
        if curr_vowel_idx == -1 or next_vowel_idx == -1:
            continue

        coda_consonants = curr_syl[curr_vowel_idx + 1:]
        onset_consonants = next_syl[:next_vowel_idx]
        total = coda_consonants + onset_consonants
        N = len(total)
        if N == 0:
            continue

        bases = [t['base_ph'] for t in total]
        split_handled = False

        # ---- 1. Geminate split (identical consecutive consonants) ----
        if not split_handled and N >= 2:
            for c_idx in range(N - 1):
                if total[c_idx]['text'] == total[c_idx + 1]['text']:
                    split_idx = c_idx + 1
                    syllables[i] = curr_syl[:curr_vowel_idx + 1] + total[:split_idx]
                    syllables[i + 1] = total[split_idx:] + next_syl[next_vowel_idx:]
                    split_handled = True
                    break

        # ---- 2. PRIMARY-stress LAX vowel attracts one coda consonant ----
        #    Only primary stress (1), not secondary (2).
        #    Orthography can block: if the word contains "by"/"br"/"py"… and the
        #    cluster matches, keep the onset (february → FE-byuh / FE-broo).
        #    bedroom has "dr", not "br", so BED-room is unchanged.
        if not split_handled:
            curr_vowel_ph = curr_syl[curr_vowel_idx]['base_ph']
            curr_vowel_stress = curr_syl[curr_vowel_idx]['stress']
            if (curr_vowel_ph in LAX_VOWELS and
                    curr_vowel_stress == '1' and
                    N >= 1):
                skip_lax = False
                if N >= 2 and is_valid_onset(bases[:2]):
                    # always keep stop+Y as onset (february → FE-byuh)
                    if bases[1] == 'Y':
                        skip_lax = True
                    else:
                        # orthography digraphs (february has "br" → FE-broo)
                        # omit DR/TR so bedroom → BED-room
                        digraphs = {
                            'BR': 'br', 'PR': 'pr', 'KR': 'kr', 'GR': 'gr',
                        }
                        dig = digraphs.get(bases[0] + bases[1])
                        if dig and dig in word_l:
                            skip_lax = True
                if not skip_lax:
                    remaining = bases[1:]
                    if is_valid_onset(remaining):
                        split_idx = 1
                        syllables[i] = curr_syl[:curr_vowel_idx + 1] + total[:split_idx]
                        syllables[i + 1] = total[split_idx:] + next_syl[next_vowel_idx:]
                        split_handled = True

        # ---- 3. Prefer nasal + s/z as coda (trans-, in-, con- …) ----
        #    Only when something remains as a real onset (e.g. L in trans-LAYT).
        #    Empty remainder must NOT count — otherwise pronunciation → nuhns-ee
        #    instead of nuhn-see.
        if not split_handled and N >= 2:
            if bases[0] == 'N' and bases[1] in {'S', 'Z'}:
                remaining = bases[2:]
                if remaining and is_valid_onset(remaining):
                    split_idx = 2
                    syllables[i] = curr_syl[:curr_vowel_idx + 1] + total[:split_idx]
                    syllables[i + 1] = total[split_idx:] + next_syl[next_vowel_idx:]
                    split_handled = True

        # ---- 3b. Prefer S as coda when:
        #   (1) S is FIRST in the cluster + COMPLEX remainder (ice-cream → IGHS-kreem)
        #       NOT mid-cluster (express K-S-P-R → ik-SPRES, not eksp-RES)
        #   (2) remainder starts with nasal M/N (craftsmanship → KRAFTS-muhn)
        #   (3) unstressed AH/IH + S before stop T/P/K (establish → uhs-ta)
        if not split_handled and N >= 2:
            for s_idx, b in enumerate(bases):
                if b != 'S' or s_idx >= N - 1:
                    continue
                remaining = bases[s_idx + 1:]
                prefer_s_coda = False
                if (s_idx == 0 and len(remaining) >= 2
                        and is_valid_onset(remaining)):
                    prefer_s_coda = True
                elif remaining[0] in {'M', 'N'} and is_valid_onset(remaining):
                    prefer_s_coda = True
                elif s_idx == 0:
                    # S must lead the cluster (establish uhs-ta; not express mid-S)
                    curr_vowel_ph = curr_syl[curr_vowel_idx]['base_ph']
                    curr_vowel_stress = curr_syl[curr_vowel_idx]['stress']
                    if (curr_vowel_ph in {'AH', 'IH'} and curr_vowel_stress == '0'
                            and remaining[0] in {'T', 'P', 'K'}
                            and is_valid_onset(remaining)):
                        prefer_s_coda = True
                if prefer_s_coda:
                    split_idx = s_idx + 1
                    syllables[i] = curr_syl[:curr_vowel_idx + 1] + total[:split_idx]
                    syllables[i + 1] = total[split_idx:] + next_syl[next_vowel_idx:]
                    split_handled = True
                    break

        # ---- 3c. Unstressed PREFIX vowel + N/M before a stressed vowel ----
        #    Only at FIRST syllable boundary AND only when the first syllable
        #    has no onset (true prefix un-/in-):
        #      uneducated → uhn-EJ-…
        #    Not when there is an onset (pronunciation → pruh-nun-, not pruhn-):
        if (not split_handled and i == 0 and N == 1 and bases[0] in {'N', 'M'}):
            onset_before = curr_syl[:curr_vowel_idx]
            if not onset_before:
                curr_vowel_ph = curr_syl[curr_vowel_idx]['base_ph']
                curr_vowel_stress = curr_syl[curr_vowel_idx]['stress']
                next_vowel_stress = next_syl[next_vowel_idx]['stress']
                if (curr_vowel_ph in {'AH', 'IH', 'UH'} and
                        curr_vowel_stress == '0' and
                        next_vowel_stress in {'1', '2'}):
                    split_idx = 1
                    syllables[i] = curr_syl[:curr_vowel_idx + 1] + total[:split_idx]
                    syllables[i + 1] = total[split_idx:] + next_syl[next_vowel_idx:]
                    split_handled = True

        # ---- 3d. Single S between unstressed vowels → S as coda ONLY when
        #    the next vowel is schwa (AH): disestablish → dis-uhs
        #    Not before IY/etc: agency → AY-juhn-see (not AY-juhns-ee)
        if not split_handled and N == 1 and bases[0] == 'S':
            curr_vowel_stress = curr_syl[curr_vowel_idx]['stress']
            next_vowel_stress = next_syl[next_vowel_idx]['stress']
            next_vowel_ph = next_syl[next_vowel_idx]['base_ph']
            if (curr_vowel_stress == '0' and next_vowel_stress == '0'
                    and next_vowel_ph == 'AH'):
                split_idx = 1
                syllables[i] = curr_syl[:curr_vowel_idx + 1] + total[:split_idx]
                syllables[i + 1] = total[split_idx:] + next_syl[next_vowel_idx:]
                split_handled = True

        # ---- 4. Single intervocalic consonant → always onset (V.CV) ----
        if not split_handled and N == 1:
            split_idx = 0
            syllables[i] = curr_syl[:curr_vowel_idx + 1] + total[:split_idx]
            syllables[i + 1] = total[split_idx:] + next_syl[next_vowel_idx:]
            split_handled = True

        # ---- 5. Maximal Onset Principle (greedy legal onset) ----
        if not split_handled and N >= 2:
            best_onset_len = 0
            for length in range(min(N, 3), 0, -1):
                if is_valid_onset(bases[-length:]):
                    best_onset_len = length
                    break
            # also allow a single consonant even if not in VALID_ONSETS
            # (already handled by the len==1 case above, but safety)
            if best_onset_len == 0 and is_valid_onset(bases[-1:]):
                best_onset_len = 1

            split_idx = N - best_onset_len

            # never leave a lone 'HH' in the coda
            for idx in range(split_idx):
                if total[idx]['base_ph'] == 'HH':
                    split_idx = idx
                    break

            syllables[i] = curr_syl[:curr_vowel_idx + 1] + total[:split_idx]
            syllables[i + 1] = total[split_idx:] + next_syl[next_vowel_idx:]

    return syllables


def syllabify_and_format_word(word_phones, original_word=""):
    phones_with_boundary = find_compound_split(original_word, word_phones)
    target_phones = phones_with_boundary if phones_with_boundary else word_phones

    tokens = apply_custom_rules(target_phones)
    all_syls = syllabify_tokens(tokens, original_word=original_word)

    formatted_parts = []
    current_word_parts = []

    for syl in all_syls:
        if not syl:
            continue
        if len(syl) == 1 and syl[0]['base_ph'] == 'BOUNDARY':
            if current_word_parts:
                formatted_parts.append("-".join(current_word_parts))
                current_word_parts = []
            continue

        has_primary_stress = any(t['stress'] == '1' for t in syl)
        syl_text = "".join([t['text'] for t in syl])

        if has_primary_stress:
            syl_text = syl_text.upper()

        current_word_parts.append(syl_text)

    if current_word_parts:
        formatted_parts.append("-".join(current_word_parts))

    sep = " " if ' ' in original_word or not ('-' in original_word or phones_with_boundary) else "-"
    return sep.join(formatted_parts)


def evaluate_heteronym_match(pos, phones):
    primary_stress_idx = -1
    for idx, p in enumerate(phones):
        if p.endswith('1'):
            primary_stress_idx = idx
            break

    if pos.startswith('NN'):
        if primary_stress_idx <= 1:
            return "Recommended (Noun Form: Initial Stress)"
    elif pos.startswith('VB'):
        if primary_stress_idx > 1:
            return "Recommended (Verb Form: Final Stress)"
    elif pos == 'DT':
        return "Recommended (Determiner Context)"
    return "Alternative Variant"


def process_phrase_with_pos(phrase_str):
    word_results = get_phrase_phonemes_with_pos(phrase_str)

    combined_variant_tuples = [[]]
    for word, pos, variants in word_results:
        new_combined = []
        for base_seq in combined_variant_tuples:
            for var in variants:
                if not var:
                    continue
                new_seq = list(base_seq)
                new_seq.append((word, var))
                new_combined.append(new_seq)
        combined_variant_tuples = new_combined

    results = []
    for var_tuple in combined_variant_tuples:
        phones = []
        word_respellings = []

        for idx, (word, var_phones) in enumerate(var_tuple):
            resp = syllabify_and_format_word(var_phones, original_word=word)
            word_respellings.append(resp)

            if phones and phones[-1] not in {'INTRUSIVE_W', 'INTRUSIVE_Y', 'BOUNDARY'}:
                last_ph = phones[-1]
                prev_base = last_ph[:-1] if last_ph[-1].isdigit() else last_ph
                curr_first_ph = var_phones[0]
                curr_base = curr_first_ph[:-1] if curr_first_ph[-1].isdigit() else curr_first_ph

                if prev_base in VOWEL_SET and curr_base in VOWEL_SET:
                    if prev_base in {'OW', 'UW', 'AO', 'AW', 'OY'}:
                        phones.append('INTRUSIVE_W')
                    elif prev_base in {'IY', 'EY', 'AY', 'IH'}:
                        phones.append('INTRUSIVE_Y')

            if phones:
                phones.append('BOUNDARY')

            split_res = find_compound_split(word, var_phones)
            if split_res:
                phones.extend(split_res)
            else:
                phones.extend(var_phones)

        combined_respelling = " ".join(word_respellings)
        ipa = get_ipa_transcription(phones)
        breakdown = get_phoneme_breakdown(phones)
        pos_labels = [f"{w} [{pos}]" for w, pos, _ in word_results]
        primary_pos = word_results[0][1] if word_results else 'O'
        recommendation = evaluate_heteronym_match(primary_pos, phones)

        results.append((phones, ipa, breakdown, combined_respelling, pos_labels, recommendation))
    return results


def export_all_respellings(out_path="all_respellings.txt"):
    """
    Export every alphabetic CMU word → ALL variant respellings to a text file.
    Fixed-width columns for readable alignment.
    Format:  word  |  respelling1  |  respelling2  |  …
    """
    words = sorted(w for w in CMU_DICT.keys() if w.isalpha())
    total = len(words)
    # column width based on longest word (cap for readability)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)

    print(f"Exporting respellings for {total} words → {out_path} …")
    written = 0
    errors = 0
    with open(out_path, "w", encoding="utf-8") as f:
        header = f"{'# word':<{col_w}}  respelling(s)"
        f.write(header + "\n")
        f.write("-" * (col_w + 60) + "\n")
        for i, w in enumerate(words, 1):
            try:
                results = process_phrase_with_pos(w)
                # collect unique respellings across all variants
                resps = []
                seen = set()
                for r in results:
                    resp = r[3]
                    if resp and resp not in seen:
                        seen.add(resp)
                        resps.append(resp)
                joined = "  |  ".join(resps) if resps else ""
                f.write(f"{w:<{col_w}}  {joined}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 5000 == 0:
                print(f"  … {i}/{total}")
    print(f"Done. Wrote {written} lines ({errors} errors) → {out_path}")
    return out_path


def export_all_ipa(out_path="all_ipa.txt"):
    """
    Export every alphabetic CMU word → ALL variant IPA forms to a text file.
    Fixed-width columns for readable alignment.
    Format:  word  |  ipa1  |  ipa2  |  …
    """
    words = sorted(w for w in CMU_DICT.keys() if w.isalpha())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)

    print(f"Exporting IPA for {total} words → {out_path} …")
    written = 0
    errors = 0
    with open(out_path, "w", encoding="utf-8") as f:
        header = f"{'# word':<{col_w}}  ipa(s)"
        f.write(header + "\n")
        f.write("-" * (col_w + 60) + "\n")
        for i, w in enumerate(words, 1):
            try:
                results = process_phrase_with_pos(w)
                ipas = []
                seen = set()
                for r in results:
                    ipa = r[1]
                    if ipa and ipa not in seen:
                        seen.add(ipa)
                        ipas.append(ipa)
                joined = "  |  ".join(ipas) if ipas else ""
                f.write(f"{w:<{col_w}}  {joined}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 5000 == 0:
                print(f"  … {i}/{total}")
    print(f"Done. Wrote {written} lines ({errors} errors) → {out_path}")
    return out_path



# ---------------------------------------------------------------------------
# UK English (Received Pronunciation) from en_UK.txt IPA dictionary
# Syllabify IPA like the US engine, UPPERCASE primary-stressed syllables.
# ---------------------------------------------------------------------------

UK_DICT_PATHS = [
    "/Users/marco/PycharmProjects/Analysis/en_UK.txt",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "en_UK.txt"),
    "en_UK.txt",
]

# Longest-first IPA symbol → respelling (UK lexical sets + loanwords)
# Core: kit i | dress e | trap a | lot o | strut/comma uh | foot uu
# fleece/happy ee | bath/palm aa | nurse ur | thought aw | goose oo
# face ay | price igh | choice oy | goat əʊ→oh | mouth ow
# near eer | square air
# Loan: x→kh | nasals aa(n)/e(n)/… | y→ue | ø→oe | ɬ→hl | ʲ→(y) …
_UK_IPA_MAP = [
    # --- French / loan nasals (longest first) ---
    ('ɑ̃', 'aa(n)'),
    ('ɛ̃', 'e(n)'),
    ('æ̃', 'a(n)'),
    ('ɔ̃', 'aw(n)'),
    ('œ̃', 'oeh(n)'),
    # --- core long / diphthongs ---
    ('ɑː', 'aa'),
    ('ɔː', 'aw'),
    ('uː', 'oo'),
    ('iː', 'ee'),
    ('ɜː', 'ur'),
    ('eɪ', 'ay'),
    ('aɪ', 'igh'),
    ('ɔɪ', 'oy'),
    ('aʊ', 'ow'),
    ('əʊ', 'oh'),
    ('oʊ', 'oh'),
    ('ɪə', 'eer'),
    ('iə', 'eer'),
    ('eə', 'air'),
    ('ʊə', 'uuuh'),
    ('tʃ', 'ch'),
    ('dʒ', 'j'),
    # --- loan fricatives / liquids ---
    ('ɬ', 'hl'),             # Welsh ll (Llanos)
    ('ʲ', '(y)'),            # palatalization (Argyll)
    ('ʁ', 'rh'),             # French uvular r
    ('β', '[b]'),
    ('ʎ', '[ly]'),
    ('ɲ', 'gn'),
    ('ɕ', '[sh]'),
    ('ʑ', '[zh]'),
    ('ħ', '[hh]'),
    ('ʕ', '[gh]'),
    ('ɣ', 'gh'),
    ('ʋ', '[v]'),
    ('ø', 'oe'),
    ('œ', 'oeh'),
    ('y', 'ue'),             # front rounded (über, menu)
    ('x', 'kh'),
    ('q', 'q'),
    # --- clicks (IPA) ---
    ('ǁ', '||'),             # alveolar lateral click
    ('ǀ', '|'),              # dental click
    ('ǃ', '!'),              # post-alveolar click
    ('ǂ', '[P]'),            # palatal click
    ('ʘ', '[B]'),            # bilabial click
    # --- glottal stop ---
    ('ʔ', "'"),              # glottal stop → '
    # --- core short vowels ---
    ('æ', 'a'),              # trap
    ('ɐ', 'uh'),
    ('ə', 'uh'),
    ('ʌ', 'uh'),
    ('ɒ', 'o'),
    ('ɛ', 'e'),              # dress (standardised to ɛ only)
    ('e', 'ey'),             # French /e/ (été)
    ('a', 'ah'),             # French /a/ (ami) standalone non-diphthong
    ('ɪ', 'i'),
    ('ʊ', 'uu'),
    ('ɑ', 'aa'),
    ('ɔ', 'aw'),
    ('u', 'oo'),
    ('i', 'ee'),
    # --- core consonants ---
    ('ŋ', 'ng'),
    ('ʃ', 'sh'),
    ('ʒ', 'zh'),
    ('θ', 'th'),
    ('ð', 'dh'),
    ('j', 'y'),
    ('ɹ', 'r'),              # English approximant
    ('r', 'r'),              # default r (trill → rr only if doubled "rr" in IPA)
    ('ʀ', 'rr'),             # uvular/trill loan
    ('w', 'w'),
    ('h', 'h'),
    ('b', 'b'), ('d', 'd'), ('f', 'f'), ('g', 'g'), ('ɡ', 'g'),
    ('k', 'k'),
    ('l', 'l'), ('m', 'm'), ('n', 'n'), ('p', 'p'), ('s', 's'),
    ('t', 't'), ('v', 'v'), ('z', 'z'),
]

_UK_VOWEL_SYMS = {
    'ɑ̃', 'ɑː', 'ɔː', 'uː', 'iː', 'ɜː', 'eɪ', 'aɪ', 'ɔɪ', 'aʊ',
    'əʊ', 'oʊ', 'ɪə', 'iə', 'eə', 'ʊə',
    'æ', 'ɐ', 'ə', 'ʌ', 'ɒ', 'ɛ', 'e', 'a', 'ɪ', 'ʊ', 'ɑ', 'ɔ', 'u', 'i',
}


def load_uk_dict(path=None):
    """Load word → list of IPA strings from en_UK.txt."""
    paths = [path] if path else UK_DICT_PATHS
    for p in paths:
        if not p:
            continue
        try:
            d = {}
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '\t' in line:
                        word, ipa = line.split('\t', 1)
                    elif ' ' in line:
                        word, ipa = line.split(None, 1)
                    else:
                        continue
                    word = word.strip().lower()
                    ipa = ipa.strip().strip('/')
                    if word and ipa:
                        d.setdefault(word, []).append(ipa)
            if d:
                print(f"UK dictionary loaded: {len(d)} words from {p}")
                return d, p
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"UK dictionary load error ({p}): {e}")
    return {}, None


def _uk_tokenize_ipa(ipa):
    """
    Tokenize IPA into phoneme units + stress markers.
    Returns list of dicts: {sym, is_vowel, stress} where stress is '', '1', or '2'
    attached to the *following vowel nucleus*.
    """
    s = ipa.strip().strip('/')
    # strip secondary articulation marks we ignore
    for ch in '͡‿ː̃':  # length already in multi-char tokens; tilde in ɑ̃ kept via map
        pass
    tokens = []
    pending_stress = ''
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == 'ˈ':
            pending_stress = '1'
            i += 1
            continue
        if ch == 'ˌ':
            pending_stress = '2'
            i += 1
            continue
        if ch in {'.', ' ', '(', ')', '[', ']', '|'}:
            i += 1
            continue
        matched = False
        for ipa_sym, resp in _UK_IPA_MAP:
            if s.startswith(ipa_sym, i):
                is_v = ipa_sym in _UK_VOWEL_SYMS
                stress = ''
                if is_v and pending_stress:
                    stress = pending_stress
                    pending_stress = ''
                tokens.append({
                    'ipa': ipa_sym,
                    'resp': resp,
                    'is_vowel': is_v,
                    'stress': stress,
                })
                i += len(ipa_sym)
                matched = True
                break
        if not matched:
            # bare char
            c = s[i]
            if c not in {'ː', '̃', '͡', '‿'}:
                is_v = c in _UK_VOWEL_SYMS
                stress = ''
                if is_v and pending_stress:
                    stress = pending_stress
                    pending_stress = ''
                tokens.append({
                    'ipa': c,
                    'resp': c,
                    'is_vowel': is_v,
                    'stress': stress,
                })
            i += 1
    return tokens


def _uk_syllabify(tokens):
    """
    Group IPA tokens into syllables (vowel nuclei + maximal onset).
    Returns list of syllables; each syllable is a list of token dicts.
    """
    if not tokens:
        return []

    # pass 1: group by vowels (onset+nucleus+following consonants temporarily)
    raw = []
    cur = []
    for t in tokens:
        cur.append(t)
        if t['is_vowel']:
            raw.append(cur)
            cur = []
    if cur:
        # trailing consonants → attach to last syllable
        if raw:
            raw[-1].extend(cur)
        else:
            raw.append(cur)

    if len(raw) <= 1:
        return raw

    # pass 2: redistribute intervocalic consonants (maximal onset)
    # simple valid onsets for UK
    valid_onsets = {
        (), ('p',), ('b',), ('t',), ('d',), ('k',), ('g',), ('ɡ',), ('f',), ('v',),
        ('θ',), ('ð',), ('s',), ('z',), ('ʃ',), ('ʒ',), ('h',), ('m',), ('n',),
        ('ŋ',), ('l',), ('r',), ('ɹ',), ('w',), ('j',), ('tʃ',), ('dʒ',),
        ('p', 'l'), ('p', 'r'), ('p', 'ɹ'), ('b', 'l'), ('b', 'r'), ('b', 'ɹ'),
        ('t', 'r'), ('t', 'ɹ'), ('d', 'r'), ('d', 'ɹ'),
        ('k', 'l'), ('k', 'r'), ('k', 'ɹ'), ('g', 'l'), ('g', 'r'), ('g', 'ɹ'),
        ('ɡ', 'l'), ('ɡ', 'r'), ('ɡ', 'ɹ'),
        ('f', 'l'), ('f', 'r'), ('f', 'ɹ'),
        ('θ', 'r'), ('θ', 'ɹ'), ('ʃ', 'r'), ('ʃ', 'ɹ'),
        ('s', 'p'), ('s', 't'), ('s', 'k'),
        ('s', 'm'), ('s', 'n'), ('s', 'l'), ('s', 'w'), ('s', 'f'),
        # s + stop + liquid/glide — include IPA ɹ (express /spr/, exquisite /skw/)
        ('s', 'p', 'l'), ('s', 'p', 'r'), ('s', 'p', 'ɹ'), ('s', 'p', 'j'), ('s', 'p', 'w'),
        ('s', 't', 'r'), ('s', 't', 'ɹ'), ('s', 't', 'j'),
        ('s', 'k', 'l'), ('s', 'k', 'r'), ('s', 'k', 'ɹ'), ('s', 'k', 'w'), ('s', 'k', 'j'),
        ('k', 'w'), ('ɡ', 'w'), ('g', 'w'),
        ('p', 'j'), ('b', 'j'), ('t', 'j'), ('d', 'j'), ('k', 'j'), ('g', 'j'), ('ɡ', 'j'),
        ('f', 'j'), ('m', 'j'), ('n', 'j'), ('h', 'j'), ('l', 'j'),
    }

    def onset_key(toks):
        return tuple(t['ipa'] for t in toks)

    for i in range(len(raw) - 1):
        # find vowel index in current and next
        def vidx(syl):
            for j, t in enumerate(syl):
                if t['is_vowel']:
                    return j
            return 0
        vi = vidx(raw[i])
        ni = vidx(raw[i + 1])
        # consonants between vowels
        left_coda = raw[i][vi + 1:]
        right_onset = raw[i + 1][:ni]
        cluster = left_coda + right_onset
        if not cluster:
            continue

        # S as coda before nasal M/N (advertisement → …-is-muhnt, not …-i-smuhnt)
        if (len(cluster) >= 2 and cluster[0]['ipa'] == 's'
                and cluster[1]['ipa'] in {'m', 'n'}):
            rest = cluster[1:]
            if onset_key(rest) in valid_onsets or len(rest) == 1:
                split = 1
                raw[i] = raw[i][:vi + 1] + cluster[:split]
                raw[i + 1] = cluster[split:] + raw[i + 1][ni:]
                continue

        # Stressed short/lax vowel attracts one coda (colour → KUHL-uh)
        # Do NOT attract ɹ/r: keep it as onset of the next syllable
        # (furry /fˈɜːɹi/ → FUR-ree, not FUR-R-ee; merry → MEHR-ee after rhotic merge)
        lax = {'ʌ', 'ɐ', 'æ', 'ɒ', 'ɪ', 'ʊ', 'ɛ', 'e', 'ə', 'ɜː'}
        v_tok = raw[i][vi]
        if (v_tok['ipa'] in lax and v_tok.get('stress') == '1'
                and len(cluster) >= 1
                and cluster[0]['ipa'] not in {'ɹ', 'r'}):
            rest = cluster[1:]
            if onset_key(rest) in valid_onsets or len(rest) <= 1:
                split = 1
                new_left = raw[i][:vi + 1] + cluster[:split]
                new_right = cluster[split:] + raw[i + 1][ni:]
                raw[i] = new_left
                raw[i + 1] = new_right
                continue

        # maximal onset
        best = 0
        for length in range(min(len(cluster), 3), -1, -1):
            if onset_key(cluster[-length:] if length else []) in valid_onsets or length == 0:
                if length == 1 or length == 0 or onset_key(cluster[-length:]) in valid_onsets:
                    best = length
                    break
        if best == 0 and cluster:
            best = 1 if len(cluster) >= 1 else 0
        split = len(cluster) - best
        new_left = raw[i][:vi + 1] + cluster[:split]
        new_right = cluster[split:] + raw[i + 1][ni:]
        raw[i] = new_left
        raw[i + 1] = new_right

    return raw


def uk_ipa_to_respelling(ipa):
    """
    UK IPA → hyphenated respelling with UPPERCASE primary stress.
    colour /kˈʌlɐ/ → KUHL-uh
    furry /fˈɜːɹi/ → FUR-ree
    merry /ˈmɛɹi/ → MEHR-ee
    """
    if not ipa:
        return ''
    tokens = _uk_tokenize_ipa(ipa)
    if not tokens:
        return ''
    syllables = _uk_syllabify(tokens)

    # Cross-syllable rhotic merge: certain short vowels + following onset /ɹ/
    # become a single rhotic spelling on the first syllable (r stays out of second).
    # ɛ/e + ɹ → ehr  (merry → MEHR-ee)
    # ɪ + ɹ → eer    (mirror → MEER-uh etc. if needed)
    # Do NOT merge for nurse ɜː/ur — keep r as onset (furry → FUR-ree)
    rhotic_merge = {
        'e': 'ehr',   # ɛ or e
        'i': 'eer',   # ɪ
        # add more if desired: 'a':'arr', 'o':'or', etc.
    }
    for i in range(len(syllables) - 1):
        syl = syllables[i]
        nxt = syllables[i + 1]
        if not syl or not nxt:
            continue
        # last token of current must be a vowel we can rhotacise
        last = syl[-1]
        if not last['is_vowel']:
            continue
        # first token(s) of next must start with r/ɹ
        if nxt[0]['ipa'] not in {'ɹ', 'r'}:
            continue
        resp = last['resp']
        if resp in rhotic_merge:
            # rewrite the vowel token's resp to the rhotic form
            last = dict(last)  # copy
            last['resp'] = rhotic_merge[resp]
            syllables[i] = syl[:-1] + [last]
            # drop the leading r from next syllable
            syllables[i + 1] = nxt[1:]

    parts = []
    for syl in syllables:
        if not syl:
            continue
        # simple join of resps (no forced ur-r hyphen)
        text = ''.join(t['resp'] for t in syl)
        has_primary = any(t['stress'] == '1' for t in syl)
        if has_primary:
            text = text.upper()
        parts.append(text)
    return '-'.join(parts)


def lookup_uk(word, uk_dict):
    """Return list of (ipa, respelling) for a word."""
    if not uk_dict:
        return []
    key = re.sub(r"[^a-zA-Z\-']", '', word.lower().strip())
    entries = uk_dict.get(key, [])
    return [(f'/{ipa}/', uk_ipa_to_respelling(ipa)) for ipa in entries]


UK_PRONUNCIATION_SHEET = """
=== UK Respelling Pronunciation Sheet ===

VOWELS (lexical sets)
  kit ɪ            → i
  dress ɛ          → e     (standardised; not plain e)
  trap æ           → a
  lot ɒ            → o
  strut ʌ          → uh
  comma/letter ə/ɐ → uh
  foot ʊ           → uu
  fleece iː        → ee
  happy i          → ee
  bath/palm ɑː     → aa
  nurse ɜː         → ur   (ɜːɹ → ur-r)
  thought ɔː       → aw
  goose uː/u       → oo
  face eɪ          → ay
  price aɪ         → igh
  choice ɔɪ        → oy
  goat əʊ          → oh
  mouth aʊ         → ow
  near ɪə/iə       → eer
  square eə        → air
  fire aɪə         → igh-uh
  power aʊə        → ow-uh

CONSONANTS (core)
  p b t d k ɡ      → p b t d k g
  tʃ dʒ            → ch j
  f v θ ð s z      → f v th dh s z
  ʃ ʒ h            → sh zh h
  m n ŋ            → m n ng
  l r/ɹ j w        → l r y w
  ʔ (glottal stop) → '

LOANWORD / MARGINAL
  x                → kh
  ɬ (Welsh ll)     → hl
  ʲ (palatal)      → (y)
  a (French ami)   → ah    (standalone open a)
  e (French été)   → ey
  ɑ̃                → aa(n)
  ɛ̃                → e(n)
  æ̃                → a(n)
  ɔ̃                → aw(n)
  œ̃                → oeh(n)
  y (front rounded)→ ue
  ø                → oe
  œ                → oeh
  ʁ (French r)     → rh
  r (alveolar trill)→ rr
  β                → [b]
  ʎ                → [ly]
  ɲ                → gn
  ɕ                → [sh]
  ʑ                → [zh]
  ħ                → [hh]
  ʕ                → [gh]
  q                → q
  ɣ                → gh
  ʋ                → [v]
  ǀ dental click   → |
  ǃ post-alveolar  → !
  ǁ alveolar lat.  → ||
  ǂ palatal        → [P]
  ʘ bilabial       → [B]
  ʋ                → [v]

STRESS
  Primary ˈ        → UPPERCASE syllable
  Secondary ˌ      → lowercase
  Hyphenate by syllable (US-style)
"""


def print_uk_pronunciation_sheet():
    print(UK_PRONUNCIATION_SHEET)


def export_all_uk_respellings(uk_dict, out_path="all_uk_respellings.txt",
                                chunk_size=50000):
    """
    Export every UK word → all respelling variants.
    Splits into multiple files of *chunk_size* entries each:
        all_uk_respellings_001.txt, _002.txt, …
    """
    words = sorted(uk_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    base, ext = os.path.splitext(out_path)
    print(f"Exporting UK respellings for {total} words (chunks of {chunk_size}) …")
    written = 0
    errors = 0
    chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        p = f"{base}_{idx:03d}{ext}"
        paths.append(p)
        fh = open(p, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  respelling(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {p}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_uk(w, uk_dict)
                resps, seen = [], set()
                for _ipa, resp in hits:
                    if resp and resp not in seen:
                        seen.add(resp)
                        resps.append(resp)
                f.write(f"{w:<{col_w}}  {'  |  '.join(resps)}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 5000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths


def export_all_uk_ipa(uk_dict, out_path="all_uk_ipa.txt",
                       chunk_size=50000):
    """
    Export every UK word → all IPA forms.
    Splits into multiple files of *chunk_size* entries each.
    """
    words = sorted(uk_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    base, ext = os.path.splitext(out_path)
    print(f"Exporting UK IPA for {total} words (chunks of {chunk_size}) …")
    written = 0
    errors = 0
    chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        p = f"{base}_{idx:03d}{ext}"
        paths.append(p)
        fh = open(p, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  ipa(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {p}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_uk(w, uk_dict)
                ipas, seen = [], set()
                for ipa, _resp in hits:
                    if ipa and ipa not in seen:
                        seen.add(ipa)
                        ipas.append(ipa)
                f.write(f"{w:<{col_w}}  {'  |  '.join(ipas)}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 5000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths



UK_DICT, UK_DICT_LOADED_FROM = load_uk_dict()


# ---------------------------------------------------------------------------
# French (fr_FR) IPA → improvised phonetic respelling
# Hyphens SEPARATE syllables (not phonemes). No capitalisation.
# bracket_vowels kept for API compat but French UI uses no vowel highlight.
# ---------------------------------------------------------------------------

FR_DICT_PATHS = [
    "/Users/marco/Downloads/fr_FR.txt",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fr_FR.txt"),
    "fr_FR.txt",
    "/home/workdir/artifacts/fr_FR.txt",
]

# Longest-first IPA → respelling glyphs (improvised system)
_FR_IPA_MAP = [
    # nasals
    ('ɑ̃', 'aⁿ'),
    ('ɛ̃', 'eⁿ'),
    ('œ̃', 'uhⁿ'),
    ('ɔ̃', 'oⁿ'),
    ('ã', 'aⁿ'),
    ('ẽ', 'eⁿ'),
    ('õ', 'oⁿ'),
    # long / open-close pairs
    ('ɛː', 'e'),
    ('aː', 'ah'),
    ('oː', 'o'),
    ('øː', 'öh'),
    ('œː', 'ö'),
    # core vowels
    ('ɑ', 'ah'),          # pâte
    ('ɛ', 'e'),           # lait / baie
    ('ə', 'ə'),           # schwa (elision → (ə) optionally)
    ('œ', 'ö'),           # lax oe + consonant
    ('ø', 'öh'),          # tense oe (peu)
    ('ɔ', 'aw'),          # open o (sort / cotte)
    ('y', 'ü'),           # tu
    ('e', 'ey'),          # clé
    ('a', 'a'),
    ('i', 'i'),
    ('o', 'o'),           # closed o (saut / côte)
    ('u', 'u'),
    # consonants & glides
    ('ʃ', 'sh'),
    ('ʒ', 'zh'),
    ('ɲ', 'gn'),          # oignon
    ('ŋ', 'ng'),          # parking (loan)
    ('ʁ', 'r'),           # guttural r written as r
    ('ɥ', 'ẅ'),           # u-glide (huit / nuit)
    ('j', 'y'),
    ('w', 'w'),
    ('b', 'b'), ('d', 'd'), ('f', 'f'), ('g', 'g'), ('ɡ', 'g'),
    ('k', 'k'), ('l', 'l'), ('m', 'm'), ('n', 'n'),
    ('p', 'p'), ('s', 's'), ('t', 't'), ('v', 'v'), ('z', 'z'),
    ('ʀ', 'r'),
    ('h', 'h'),
    # residual
    ('ː', ''),
    ('ʼ', ''),
    ('ˈ', ''),
    ('ˌ', ''),
]

# Plain i/u (and other vowel glyphs) = nuclei; y/w/ẅ = semivowels only
_FR_VOWEL_RESPS = {
    'i', 'ey', 'e', 'a', 'ah', 'o', 'aw', 'u', 'ü', 'öh', 'ö', 'ə',
    'aⁿ', 'eⁿ', 'oⁿ', 'uhⁿ',
}
_FR_GLIDES = {'y', 'w', 'ẅ'}  # never syllable nuclei

# French-friendly onsets for maximal-onset syllabification (as glyph tuples)
_FR_VALID_ONSETS = {
    ('b', 'r'), ('b', 'l'), ('p', 'r'), ('p', 'l'),
    ('d', 'r'), ('t', 'r'), ('k', 'r'), ('k', 'l'),
    ('g', 'r'), ('g', 'l'), ('f', 'r'), ('f', 'l'),
    ('v', 'r'), ('v', 'l'),
    ('s', 'p'), ('s', 't'), ('s', 'k'), ('s', 'm'), ('s', 'n'),
    ('s', 'l'), ('s', 'w'),
    ('sh', 'r'), ('sh', 'l'),
    ('p', 'y'), ('b', 'y'), ('f', 'y'), ('v', 'y'), ('k', 'y'),
    ('m', 'y'), ('n', 'y'), ('l', 'y'),
    ('p', 'w'), ('k', 'w'), ('g', 'w'), ('t', 'w'), ('d', 'w'),
    ('s', 'p', 'r'), ('s', 'p', 'l'), ('s', 't', 'r'), ('s', 'k', 'r'),
    ('s', 'k', 'l'), ('s', 'k', 'w'),
}


def load_fr_dict(path=None):
    """Load word → list of IPA strings from fr_FR.txt (tab or space separated)."""
    paths = [path] if path else FR_DICT_PATHS
    for p in paths:
        if not p:
            continue
        try:
            d = {}
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '\t' in line:
                        word, ipa = line.split('\t', 1)
                    elif ' ' in line:
                        word, ipa = line.split(None, 1)
                    else:
                        continue
                    word = word.strip().lower()
                    for raw in ipa.split(','):
                        raw = raw.strip().strip('/')
                        if word and raw:
                            d.setdefault(word, []).append(raw)
            if d:
                print(f"FR dictionary loaded: {len(d)} words from {p}")
                return d, p
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"FR dictionary load error ({p}): {e}")
    return {}, None


def _fr_normalize_semivowels(ipa):
    """Force j/y + i → ɥi when dictionary wrote /nyi/ etc."""
    return re.sub(r'[jy]i', 'ɥi', ipa)


# Parisian merges /œ̃/ → /ɛ̃/ (both eⁿ). Canadian keeps uhⁿ vs eⁿ.
FR_PARISIAN_MERGE = {'on': False}  # set True only via PARMER=TRUE


def _fr_tokenize_ipa(ipa, parisian_merge=None):
    """Tokenize French IPA into respelling glyphs.

    parisian_merge=False (default): keep /ɛ̃/→eⁿ, /œ̃/→uhⁿ distinct.
    parisian_merge=True: merge both to eⁿ (Parisian). Enable with PARMER=TRUE.
    """
    if parisian_merge is None:
        parisian_merge = FR_PARISIAN_MERGE['on']
    s = _fr_normalize_semivowels(ipa.strip().strip('/'))
    s = re.sub(r'[͡‿]', '', s)  # strip tie / liaison mark from stream
    tokens = []
    i = 0
    while i < len(s):
        matched = False
        for ipa_sym, resp in _FR_IPA_MAP:
            if not ipa_sym:
                continue
            if s.startswith(ipa_sym, i):
                if resp:
                    if parisian_merge and resp == 'uhⁿ':
                        resp = 'eⁿ'
                    tokens.append(resp)
                i += len(ipa_sym)
                matched = True
                break
        if not matched:
            c = s[i]
            if c not in {'̃', 'ː', '͡', '‿', 'ʼ', 'ˈ', 'ˌ', '.', ' ', '|'}:
                tokens.append(c)
            i += 1
    return tokens


def _fr_is_vowel(g):
    return g in _FR_VOWEL_RESPS


def _fr_onset_ok(cluster):
    """True if cluster may start a French syllable (order-sensitive)."""
    if not cluster:
        return True
    if len(cluster) == 1:
        return cluster[0] not in _FR_VOWEL_RESPS and cluster[0] not in {'.', '-'}
    return tuple(cluster) in _FR_VALID_ONSETS


def _fr_syllabify(tokens):
    """
    Maximal-onset syllabification for French glyph tokens.
    Returns list of syllables (each a list of glyphs).
    """
    if not tokens:
        return []
    # nucleus indices
    nuclei = [i for i, t in enumerate(tokens) if _fr_is_vowel(t)]
    if not nuclei:
        return [tokens]

    syllables = []
    prev_end = 0
    for ni, nidx in enumerate(nuclei):
        # consonants between previous nucleus and this one
        if ni == 0:
            onset = tokens[0:nidx]
            coda_start = nidx + 1
        else:
            between = tokens[prev_end:nidx]
            # split between: keep maximal legal onset for this syllable
            split = 0
            for k in range(len(between) + 1):
                cand = between[k:]
                if _fr_onset_ok(cand):
                    split = k
                    break
            # leftover before split → coda of previous syllable
            if syllables and split > 0:
                syllables[-1].extend(between[:split])
            onset = between[split:]
            coda_start = nidx + 1

        # coda runs until next nucleus (filled on next iter) or end
        if ni + 1 < len(nuclei):
            next_n = nuclei[ni + 1]
            # temporary: nucleus only; coda decided when processing next
            syl = list(onset) + [tokens[nidx]]
            syllables.append(syl)
            prev_end = nidx + 1
        else:
            # last syllable takes remaining coda
            syl = list(onset) + tokens[nidx:]
            syllables.append(syl)
            prev_end = len(tokens)

    # attach any trailing material not consumed
    if prev_end < len(tokens) and syllables:
        syllables[-1].extend(tokens[prev_end:])
    return syllables


def fr_ipa_to_respelling(ipa, bracket_vowels=False, parisian_merge=None):
    """
    French IPA → hyphen-separated syllable respelling.
    Hyphens separate syllables only (improves readability).
    bracket_vowels ignored for output shape (no [] highlighting).
    parisian_merge: False (default) → œ̃=uhⁿ, ɛ̃=eⁿ (no merge).
                    True → both eⁿ (Parisian). Toggle with PARMER=TRUE.
    """
    if not ipa:
        return ''
    tokens = _fr_tokenize_ipa(ipa, parisian_merge=parisian_merge)
    if not tokens:
        return ''
    syllables = _fr_syllabify(tokens)
    parts = []
    for syl in syllables:
        if not syl:
            continue
        parts.append(''.join(syl))
    return '-'.join(parts)


def lookup_fr(word, fr_dict, bracket_vowels=False, parisian_merge=None):
    """Return list of (ipa, respelling) for a French word or multi-word phrase.

    Tries the exact lowercased form first so every dictionary key is found.
    """
    if not fr_dict:
        return []
    raw = word.lower().strip()
    # 1. exact key
    entries = fr_dict.get(raw, [])
    if not entries:
        # 2. collapse whitespace
        key = re.sub(r"\s+", " ", raw)
        entries = fr_dict.get(key, [])
    if not entries:
        # 3. without spaces
        entries = fr_dict.get(raw.replace(" ", ""), [])
    if not entries:
        # 4. light cleaning
        key = re.sub(r"\s+", " ", raw)
        key = re.sub(r"[^a-zA-ZÀ-ÿœŒæÆçÇ0-9\-',. ]", "", key)
        entries = fr_dict.get(key, []) or fr_dict.get(key.replace(" ", ""), [])
    if not entries:
        # 5. accent fold
        key2 = (raw.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
                    .replace('à', 'a').replace('â', 'a').replace('ù', 'u')
                    .replace('û', 'u').replace('ô', 'o').replace('î', 'i')
                    .replace('ï', 'i').replace('ç', 'c').replace('œ', 'oe'))
        entries = fr_dict.get(key2, []) or fr_dict.get(key2.replace(" ", ""), [])
    return [(f'/{ipa}/', fr_ipa_to_respelling(ipa, bracket_vowels=bracket_vowels, parisian_merge=parisian_merge))
            for ipa in entries]


def export_all_fr_respellings(fr_dict, out_path="all_fr_respellings.txt",
                              bracket_vowels=False, chunk_size=50000,
                              parisian_merge=None):
    """
    Export every FR word → all respelling variants.
    Splits into multiple files of *chunk_size* entries each:
        all_fr_respellings_001.txt, _002.txt, …
    (or all_fr_respellings_bracket_001.txt when bracket_vowels=True)
    """
    words = sorted(fr_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    mode = "with []" if bracket_vowels else "plain"
    base, ext = os.path.splitext(out_path)
    print(f"Exporting FR respellings ({mode}) for {total} words "
          f"(chunks of {chunk_size}) …")
    written = 0
    errors = 0
    chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        p = f"{base}_{idx:03d}{ext}"
        paths.append(p)
        fh = open(p, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  respelling(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {p}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_fr(w, fr_dict, bracket_vowels=bracket_vowels, parisian_merge=parisian_merge)
                resps, seen = [], set()
                for _ipa, resp in hits:
                    if resp and resp not in seen:
                        seen.add(resp)
                        resps.append(resp)
                f.write(f"{w:<{col_w}}  {'  |  '.join(resps)}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 10000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths


def export_all_fr_ipa(fr_dict, out_path="all_fr_ipa.txt",
                       chunk_size=50000):
    """
    Export every FR word → all IPA forms.
    Splits into multiple files of *chunk_size* entries each.
    """
    words = sorted(fr_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    base, ext = os.path.splitext(out_path)
    print(f"Exporting FR IPA for {total} words (chunks of {chunk_size}) …")
    written = 0
    errors = 0
    chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        p = f"{base}_{idx:03d}{ext}"
        paths.append(p)
        fh = open(p, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  ipa(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {p}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_fr(w, fr_dict, bracket_vowels=False)
                ipas, seen = [], set()
                for ipa, _resp in hits:
                    if ipa and ipa not in seen:
                        seen.add(ipa)
                        ipas.append(ipa)
                f.write(f"{w:<{col_w}}  {'  |  '.join(ipas)}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 10000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths


FR_DICT, FR_DICT_LOADED_FROM = load_fr_dict()


# ---------------------------------------------------------------------------
# German (de) IPA → respelling
# Continuous (no hyphens). Primary stress = combining acute ́ on the vowel.
# Reformed German respelling (AHD-like stress, hyphen syllables, no pills)
# Stress: ′ at start of stressed syllable + **bold** syllable text
# Glottal stop → '.' separator; syllabic əl/əm/ən count as nuclei
# Dictionary: /Users/marco/Downloads/de (2).txt  (and local fallbacks)
# ---------------------------------------------------------------------------

DE_DICT_PATHS = [
    "/Users/marco/Downloads/de (2).txt",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "de (2).txt"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_DE.txt"),
    "de (2).txt",
    "de_DE.txt",
    "/home/workdir/artifacts/de (2).txt",
]

# Longest-first IPA → glyph
# Foreign / rare → nearest native (or limited loan) approximation
_DE_IPA_MAP = [
    # --- syllabic consonants (nuclei) ---
    ('l̩', 'əl'), ('m̩', 'əm'), ('n̩', 'ən'), ('ŋ̩', 'əng'),
    ('l̍', 'əl'), ('m̍', 'əm'), ('n̍', 'ən'), ('ŋ̍', 'əng'),
    ('ɫ̩', 'əl'), ('ʎ̩', 'əl'), ('ʁ̩', 'ər'), ('r̩', 'ər'),
    ('ɱ̍', 'əm'), ('ɱ', 'm'), ('b̩', 'əb'), ('s̩', 'əs'),
    ('ʒ̩', 'əzh'), ('ʙ̩', 'ər'),
    # --- affricates ---
    ('dʒ', 'j'), ('tʃ', 'ch'), ('pf', 'pf'), ('ts', 'ts'),
    # --- diphthongs (̯ stripped in normalize → same keys) ---
    ('aɪ', 'igh'), ('aʊ', 'au'), ('ɔʏ', 'oy'), ('ɔɪ', 'oy'),
    ('ɔʊ', 'oh'), ('oʊ', 'oh'), ('əʊ', 'oh'), ('ɛɪ', 'ey'),
    # --- long nasals / nasals ---
    ('ɑ̃ː', 'aaⁿ'), ('ɔ̃ː', 'awⁿ'), ('ɛ̃ː', 'eⁿ'), ('œ̃ː', 'öⁿ'),
    ('ãː', 'aaⁿ'), ('õː', 'awⁿ'),
    ('ɑ̃', 'aaⁿ'), ('ã', 'aaⁿ'), ('ɔ̃', 'awⁿ'), ('õ', 'awⁿ'),
    ('ɛ̃', 'eⁿ'), ('œ̃', 'öⁿ'), ('ɐ̃', 'aaⁿ'),
    # --- long monophthongs ---
    ('aː', 'aa'), ('ɛː', 'eh'), ('eː', 'ey'), ('iː', 'iy'),
    ('oː', 'oh'), ('øː', 'öh'), ('uː', 'uw'), ('yː', 'üh'),
    ('ɔː', 'aw'), ('œː', 'ö'), ('ɑː', 'aa'), ('æː', 'e'),
    ('əː', 'ə'), ('ɐː', 'ər'), ('ɒː', 'aw'),
    # --- short / reduced / foreign vowels → nearest ---
    ('ɐ', '*r*'),         # vocalic r
    ('ə', 'ə'),
    ('ɛ', 'e'), ('ɪ', 'i'), ('ɔ', 'aw'), ('œ', 'ö'), ('ø', 'öh'),
    ('ʊ', 'uu'), ('ʏ', 'ü'),
    ('ɑ', 'aa'), ('æ', 'e'), ('ɒ', 'aw'), ('ʌ', 'a'),
    ('ɨ', 'i'), ('ʉ', 'uw'), ('ɯ', 'uw'), ('ɘ', 'ə'), ('ɜ', 'ö'),
    ('ɤ', 'ö'), ('ɚ', 'ər'),
    ('a', 'a'), ('e', 'ey'), ('i', 'iy'), ('o', 'oh'),
    ('u', 'uw'), ('y', 'üh'),
    # --- consonants (native + nearest for foreign) ---
    ('ç', '*ch*'), ('x', 'kh'), ('ʃ', 'sh'), ('ʒ', 'zh'),
    ('ɕ', 'sh'), ('ʂ', 'sh'), ('ɬ', 'l'), ('ɫ', 'l'),
    ('ʎ', 'l'), ('ɲ', 'n'), ('ŋ', 'ng'),
    ('ʁ', 'r'), ('ʀ', 'r'), ('r', 'r'), ('ɹ', 'r'), ('ɾ', 'r'),
    ('ɺ', 'r'), ('ʋ', 'v'), ('ɥ', 'w'),
    ('θ', 't'), ('ð', 'd'),          # English th → t/d
    ('ɣ', 'g'), ('ɸ', 'f'), ('ħ', 'h'), ('ɦ', 'h'),
    ('ʕ', 'r'), ('ɓ', 'b'), ('ʈ', 't'),
    ('ʔ', 'ʔ'),
    ('b', 'b'), ('d', 'd'), ('f', 'f'), ('g', 'g'), ('ɡ', 'g'),
    ('h', 'h'), ('j', 'y'), ('k', 'k'), ('l', 'l'),
    ('m', 'm'), ('n', 'n'), ('p', 'p'),
    ('s', 's'), ('t', 't'), ('v', 'v'), ('w', 'w'),
    ('z', 'z'),
    # length-only / residual → drop
    ('ː', ''), ('ˈ', ''), ('ˌ', ''),
]

_DE_VOWEL_GLYPHS = {
    'a', 'aa', 'e', 'eh', 'ey', 'i', 'iy', 'oh', 'aw', 'öh', 'ö',
    'uu', 'uw', 'ü', 'üh', 'igh', 'au', 'oy', 'ə',
    'aaⁿ', 'awⁿ', 'eⁿ', 'öⁿ',
    'əl', 'əm', 'ən', 'əng', 'əb', 'əs', 'əzh',  # syllabic nuclei
    # *r* (vocalic ɐ) and ər are NOT nuclei — attach as coda to previous syllable
}

_DE_VALID_ONSETS = {
    ('b', 'r'), ('b', 'l'), ('p', 'r'), ('p', 'l'),
    ('d', 'r'), ('t', 'r'), ('k', 'r'), ('k', 'l'),
    ('g', 'r'), ('g', 'l'), ('f', 'r'), ('f', 'l'),
    ('v', 'r'), ('v', 'l'), ('sh', 'r'), ('sh', 'l'),
    ('p', 'f'), ('t', 's'), ('t', 'sh'),
    ('k', 'v'), ('k', 'n'), ('k', 'y'),
    ('p', 'y'), ('b', 'y'), ('f', 'y'), ('v', 'y'),
    ('s', 'p'), ('s', 't'), ('s', 'k'), ('s', 'm'), ('s', 'n'),
    ('s', 'l'), ('s', 'v'), ('s', 'y'),
    ('sh', 'p'), ('sh', 't'), ('sh', 'v'),
    ('s', 'p', 'r'), ('s', 'p', 'l'), ('s', 't', 'r'),
    ('s', 'k', 'r'), ('s', 'k', 'l'), ('s', 'k', 'v'),
}


def load_de_dict(path=None):
    """Load word → list of IPA strings from German dictionary file."""
    paths = [path] if path else DE_DICT_PATHS
    for p in paths:
        if not p:
            continue
        try:
            d = {}
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '\t' in line:
                        word, ipa = line.split('\t', 1)
                    elif ' ' in line and '/' in line:
                        # fallback space-separated
                        parts = line.split(None, 1)
                        if len(parts) < 2:
                            continue
                        word, ipa = parts
                    else:
                        continue
                    word = word.strip().lower()
                    for raw in ipa.split(','):
                        raw = raw.strip().strip('/')
                        if word and raw:
                            d.setdefault(word, []).append(raw)
            if d:
                print(f"DE dictionary loaded: {len(d)} words from {p}")
                return d, p
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"DE dictionary load error ({p}): {e}")
    return {}, None


def _de_normalize_ipa(s):
    """
    Normalize German IPA before mapping:
    - strip ties → pf, ts match
    - strip non-syllabic ̯ ̆ so aɪ̯→aɪ, i̯ dropped later as bare residue
    - strip secondary stress ˌ and articulatory noise
    - keep primary ˈ, glottal ʔ, syllabic ̩/̍, nasal ̃
    - collapse doubled stress / length noise
    """
    s = s.strip().strip('/')
    s = re.sub(r'[͡‿⁀]', '', s)
    # non-syllabic + articulation + secondary stress (ˌ alone)
    s = re.sub(r'[̯̥̝̞̪̺̣̱̮̆̊̑̚ʰʲʶˠˑˌ]', '', s)
    s = re.sub(r'̆{2,}', '', s)
    s = re.sub(r'ˈ{2,}', 'ˈ', s)
    s = re.sub(r'ː{2,}', 'ː', s)
    return s


def _de_tokenize_ipa(ipa):
    """
    Tokenize German IPA → list of (glyph, is_vowel, is_glottal).
    Primary stress ˈ is recorded as pending and attached to the next nucleus
    during syllabification.
    """
    s = _de_normalize_ipa(ipa)
    tokens = []
    pending_stress = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == 'ˈ':
            pending_stress = True
            i += 1
            continue
        if ch in {'.', ' ', '(', ')', '[', ']', '|'}:
            i += 1
            continue
        matched = False
        for ipa_sym, resp in _DE_IPA_MAP:
            if not ipa_sym:
                continue
            if s.startswith(ipa_sym, i):
                if resp == 'ʔ':
                    tokens.append(('ʔ', False, False))
                elif resp:
                    is_v = resp in _DE_VOWEL_GLYPHS
                    stress = pending_stress and is_v
                    if stress:
                        pending_stress = False
                    tokens.append((resp, is_v, stress))
                i += len(ipa_sym)
                matched = True
                break
        if not matched:
            c = s[i]
            if c not in {'ː', '̃', '̩', '̍'}:
                # unlisted foreign residue → skip
                pass
            i += 1
    # merge schwa + syllabic liquid/nasal → single nucleus (großem, Mantel…)
    merged = []
    for g, is_v, stress in tokens:
        if merged and merged[-1][0] == 'ə' and g in {'əl', 'əm', 'ən', 'əng'}:
            prev_g, prev_v, prev_s = merged.pop()
            merged.append((g, True, prev_s or stress))
        else:
            merged.append((g, is_v, stress))
    return merged


def _de_is_vowel(g):
    return g in _DE_VOWEL_GLYPHS


def _de_onset_ok(cluster):
    if not cluster:
        return True
    if len(cluster) == 1:
        return not _de_is_vowel(cluster[0]) and cluster[0] != 'ʔ'
    return tuple(cluster) in _DE_VALID_ONSETS


def _de_syllabify(tokens):
    """
    Maximal-onset German syllabification.
    tokens: list of (glyph, is_vowel, stress)
    Returns list of (syllable_glyph_list, has_stress, break_before_is_glottal)
    """
    if not tokens:
        return []
    # flatten with glottal as hard break
    pieces = []
    buf = []
    for g, is_v, stress in tokens:
        if g == 'ʔ':
            if buf:
                pieces.append(buf)
                buf = []
            pieces.append([('ʔ', False, False)])  # marker
        else:
            buf.append((g, is_v, stress))
    if buf:
        pieces.append(buf)

    result = []
    for piece in pieces:
        if len(piece) == 1 and piece[0][0] == 'ʔ':
            result.append((['ʔ'], False, True))
            continue
        nuclei = [i for i, (g, is_v, _) in enumerate(piece) if is_v]
        if not nuclei:
            result.append(([g for g, _, _ in piece], False, False))
            continue
        prev_end = 0
        for ni, nidx in enumerate(nuclei):
            if ni == 0:
                onset = piece[0:nidx]
            else:
                between = piece[prev_end:nidx]
                split = len(between)
                for k in range(len(between) + 1):
                    cand = [g for g, _, _ in between[k:]]
                    if _de_onset_ok(cand):
                        split = k
                        break
                if result and split > 0:
                    # extend previous syllable coda
                    extra = [g for g, _, _ in between[:split]]
                    prev_syl, prev_st, prev_gl = result[-1]
                    result[-1] = (prev_syl + extra, prev_st, prev_gl)
                onset = between[split:]
            stress = piece[nidx][2]
            if ni + 1 < len(nuclei):
                syl = [g for g, _, _ in onset] + [piece[nidx][0]]
                result.append((syl, stress, False))
                prev_end = nidx + 1
            else:
                syl = [g for g, _, _ in onset] + [g for g, _, _ in piece[nidx:]]
                result.append((syl, stress, False))
                prev_end = len(piece)
    return result


def de_ipa_to_respelling(ipa, bracket_vowels=False):
    """
    German IPA → hyphen-separated syllables.
    - ′ before stressed syllable; syllable text in **bold** (markdown)
    - No capitalisation, no [] pills (bracket_vowels ignored)
    - Glottal stop → '.' separator instead of '-'
    - Syllabic əl/əm/ən count as full syllables
    """
    if not ipa:
        return ''
    tokens = _de_tokenize_ipa(ipa)
    if not tokens:
        return ''
    syllables = _de_syllabify(tokens)
    if not syllables:
        return ''

    parts = []  # (text, glottal_before)
    pending_glottal = False
    for syl, stress, is_glottal_mark in syllables:
        if is_glottal_mark or (len(syl) == 1 and syl[0] == 'ʔ'):
            pending_glottal = True
            continue
        text = ''.join(g for g in syl if g != 'ʔ')
        if not text:
            continue
        if stress:
            text = '′' + text  # marker only; UI applies weight 700, never shows ′
        # vocalic *r* / ər alone → merge as coda onto previous syllable
        bare = text.lstrip('′')
        if parts and bare in {'*r*', 'ər', 'r'} and not pending_glottal:
            prev_text, prev_gl = parts[-1]
            parts[-1] = (prev_text + bare, prev_gl)
            continue
        parts.append((text, pending_glottal))
        pending_glottal = False

    if not parts:
        return ''
    out = [parts[0][0]]
    for text, glottal_before in parts[1:]:
        out.append('.' if glottal_before else '-')
        out.append(text)
    return ''.join(out)


def lookup_de(word, de_dict, bracket_vowels=False):
    """Return list of (ipa, respelling) for a German word / phrase.

    Tries the exact lowercased form first so every dictionary key
    (including hyphenated / numbered entries like 0,2-liter-flasche)
    is found without being altered or omitted.
    """
    if not de_dict:
        return []
    raw = word.lower().strip()
    # 1. exact key as stored in the dictionary
    entries = de_dict.get(raw, [])
    if not entries:
        # 2. collapse whitespace only
        key = re.sub(r"\s+", " ", raw)
        entries = de_dict.get(key, [])
    if not entries:
        # 3. without spaces
        entries = de_dict.get(raw.replace(" ", ""), [])
    if not entries:
        # 4. light cleaning (strip rare punctuation) but keep letters/digits/hyphen/comma/period
        key = re.sub(r"\s+", " ", raw)
        key = re.sub(r"[^a-zA-ZÀ-ÿÄÖÜäöüßœŒæÆçÇ0-9\-',. ]", "", key)
        entries = de_dict.get(key, []) or de_dict.get(key.replace(" ", ""), [])
    if not entries:
        # 5. accent fold
        key2 = (raw.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
                    .replace('ß', 'ss'))
        entries = de_dict.get(key2, []) or de_dict.get(key2.replace(" ", ""), [])
    return [(f'/{ipa}/', de_ipa_to_respelling(ipa, bracket_vowels=bracket_vowels))
            for ipa in entries]


def export_all_de_respellings(de_dict, out_path="all_de_respellings.txt",
                              bracket_vowels=False, chunk_size=50000):
    """
    Export every DE word → respelling(s), split into 50k-entry files.
    bracket_vowels=False → plain + ˈ
    bracket_vowels=True  → vowels in [] / stressed in {}  (+ ˈ)
    """
    words = sorted(de_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    mode = "bracketed" if bracket_vowels else "plain"
    base, ext = os.path.splitext(out_path)
    print(f"Exporting DE respellings ({mode}) for {total} words "
          f"(chunks of {chunk_size}) …")
    written = errors = chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        p = f"{base}_{idx:03d}{ext}"
        paths.append(p)
        fh = open(p, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  respelling(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {p}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_de(w, de_dict, bracket_vowels=bracket_vowels)
                resps, seen = [], set()
                for _ipa, resp in hits:
                    if resp and resp not in seen:
                        seen.add(resp)
                        resps.append(resp)
                f.write(f"{w:<{col_w}}  {'  |  '.join(resps)}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 10000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths


def export_all_de_ipa(de_dict, out_path="all_de_ipa.txt", chunk_size=50000):
    """Export every DE word → IPA form(s), split into 50k-entry files."""
    words = sorted(de_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    base, ext = os.path.splitext(out_path)
    print(f"Exporting DE IPA for {total} words (chunks of {chunk_size}) …")
    written = errors = chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        p = f"{base}_{idx:03d}{ext}"
        paths.append(p)
        fh = open(p, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  ipa(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {p}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_de(w, de_dict)
                ipas, seen = [], set()
                for ipa, _resp in hits:
                    if ipa and ipa not in seen:
                        seen.add(ipa)
                        ipas.append(ipa)
                f.write(f"{w:<{col_w}}  {'  |  '.join(ipas)}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 10000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths


DE_DICT, DE_DICT_LOADED_FROM = load_de_dict()



# ---------------------------------------------------------------------------
# Spanish (es_ES) — RAE-aligned respelling
# Dictionary: /Users/marco/Downloads/es_ES.txt
# Soft consonants italic; stress from IPA ˈ or RAE orthographic rules
# Hyphens follow Spanish syllable structure
# ---------------------------------------------------------------------------

ES_DICT_PATHS = [
    "/Users/marco/Downloads/es_ES.txt",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "es_ES.txt"),
    "es_ES.txt",
    "/home/workdir/artifacts/es_ES.txt",
    "/home/workdir/attachments/es_ES.txt",
]

# Longest-first IPA → glyph
_ES_IPA_MAP = [
    # affricates / digraphs
    ('tʃ', 'ch'),
    ('dʒ', 'j'),
    # soft (approximant) consonants
    ('β', '*b*'),
    ('ð', 'dh'),
    ('ɣ', 'gh'),
    ('ʝ', '*y*'),
    # nasals / liquids / rhotics
    ('ɲ', '*ny*'),
    ('ʎ', 'y'),          # lluvia
    ('ŋ', 'ng'),
    ('r', 'rr'),         # rolled (río)
    ('ɾ', 'r'),          # flap (brío / better)
    # fricatives
    ('θ', 'th'),         # cera (distinción)
    ('ʃ', 'sh'),
    ('x', '*h*'),        # jarra / loch
    ('s', 's'),
    ('f', 'f'),
    # stops
    ('b', 'b'), ('d', 'd'), ('g', 'g'), ('ɡ', 'g'),
    ('p', 'p'), ('t', 't'), ('k', 'k'),
    # glides
    ('j', 'y'), ('w', 'w'), ('i̯', 'y'), ('u̯', 'w'),
    # vowels
    ('a', 'a'), ('e', 'e'), ('i', 'i'), ('o', 'o'), ('u', 'u'),
    # residual
    ('ˈ', ''), ('ˌ', ''), ('ː', ''),
    ('l', 'l'), ('m', 'm'), ('n', 'n'),
    ('ʎ', 'y'),
]

_ES_VOWELS = {'a', 'e', 'i', 'o', 'u'}  # plain i/u = full vowels (nuclei)
# j→y and w→w are SEMIVOWELS only (never nuclei; not in _ES_VOWELS)
_ES_STRONG = {'a', 'e', 'o'}
_ES_WEAK = {'i', 'u'}  # weak vowels for diphthong/hiatus; glides y/w excluded

_ES_VALID_ONSETS = {
    ('p', 'r'), ('p', 'l'), ('b', 'r'), ('b', 'l'),
    ('*b*', 'r'), ('*b*', 'l'),
    ('t', 'r'), ('d', 'r'), ('dh', 'r'),
    ('k', 'r'), ('k', 'l'), ('g', 'r'), ('g', 'l'),
    ('gh', 'r'), ('gh', 'l'),
    ('f', 'r'), ('f', 'l'),
    ('ch', 'r'),
    # consonant + glide (semivowel) onsets
    ('th', 'y'), ('t', 'y'), ('d', 'y'), ('dh', 'y'),
    ('s', 'y'), ('n', 'y'), ('l', 'y'), ('p', 'y'),
    ('b', 'y'), ('*b*', 'y'), ('k', 'y'), ('g', 'y'),
    ('f', 'y'), ('m', 'y'), ('ch', 'y'),
    ('k', 'w'), ('g', 'w'), ('d', 'w'), ('t', 'w'),
    ('s', 'w'), ('p', 'w'), ('b', 'w'),
}


def load_es_dict(path=None):
    """Load word → list of IPA strings from Spanish dictionary."""
    paths = [path] if path else ES_DICT_PATHS
    for pth in paths:
        if not pth:
            continue
        try:
            d = {}
            with open(pth, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '\t' in line:
                        word, rest = line.split('\t', 1)
                    else:
                        parts = line.split(None, 1)
                        if len(parts) < 2:
                            continue
                        word, rest = parts
                    word = word.strip().lower()
                    for raw in re.split(r'\s*,\s*', rest.strip()):
                        raw = raw.strip().strip('/')
                        if word and raw:
                            d.setdefault(word, []).append(raw)
            if d:
                print(f"ES dictionary loaded: {len(d)} words from {pth}")
                return d, pth
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"ES dictionary load error ({pth}): {e}")
    return {}, None


def _es_tokenize(ipa):
    s = ipa.strip().strip('/')
    s = re.sub(r'[͡‿]', '', s)
    tokens = []
    pending_stress = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == 'ˈ':
            pending_stress = True
            i += 1
            continue
        if ch in {'ˌ', '.', ' ', '|', '(', ')'}:
            i += 1
            continue
        matched = False
        for ipa_sym, resp in _ES_IPA_MAP:
            if not ipa_sym:
                continue
            if s.startswith(ipa_sym, i):
                if resp:
                    is_v = resp in _ES_VOWELS
                    stress = pending_stress and is_v
                    if stress:
                        pending_stress = False
                    tokens.append((resp, is_v, stress))
                i += len(ipa_sym)
                matched = True
                break
        if not matched:
            c = s[i]
            if c not in {'ː', '̃', '̯', '̆'}:
                is_v = c in _ES_VOWELS
                stress = pending_stress and is_v
                if stress:
                    pending_stress = False
                tokens.append((c, is_v, stress))
            i += 1
    return tokens


def _es_onset_ok(cluster):
    if not cluster:
        return True
    if len(cluster) == 1:
        return cluster[0] not in _ES_VOWELS
    return tuple(cluster) in _ES_VALID_ONSETS


def _es_is_diphthong(v1, v2, stress1=False, stress2=False):
    """Spanish diphthong vs hiatus.

    - Two strong (a/e/o): always hiatus
    - Two weak (i/u): always diphthong (muy, fui)
    - Weak + strong / strong + weak: diphthong unless the *weak*
      vowel carries stress → hiatus (río, María, baúl, país)
    - Glides j/w are never nuclei (handled upstream; IPA j/w → y/w)
    """
    if v1 in _ES_STRONG and v2 in _ES_STRONG:
        return False
    if v1 in _ES_WEAK and v2 in _ES_WEAK:
        return True
    # mixed: stressed weak → hiatus
    if v1 in _ES_WEAK and v2 in _ES_STRONG and stress1:
        return False
    if v1 in _ES_STRONG and v2 in _ES_WEAK and stress2:
        return False
    return True


def _es_syllabify(tokens):
    if not tokens:
        return []
    # only plain a e i o u are nuclei — y/w (from j/w) are semivowels
    nuclei = [i for i, (g, is_v, _) in enumerate(tokens) if is_v]
    if not nuclei:
        return [([g for g, _, _ in tokens], False)]

    groups = []
    i = 0
    while i < len(nuclei):
        group = [nuclei[i]]
        while i + 1 < len(nuclei):
            li, ri = nuclei[i], nuclei[i + 1]
            if ri != li + 1:
                break  # consonant between → separate syllables
            left, _, st1 = tokens[li]
            right, _, st2 = tokens[ri]
            if _es_is_diphthong(left, right, st1, st2):
                group.append(ri)
                i += 1
            else:
                break
        groups.append(group)
        i += 1

    result = []
    prev_end = 0
    for gi, group in enumerate(groups):
        n_start = group[0]
        n_end = group[-1]
        stress = any(tokens[j][2] for j in group)
        if gi == 0:
            onset = tokens[0:n_start]
        else:
            between = tokens[prev_end:n_start]
            split = len(between)
            for k in range(len(between) + 1):
                cand = [g for g, _, _ in between[k:]]
                if _es_onset_ok(cand):
                    split = k
                    break
            if result and split > 0:
                extra = [g for g, _, _ in between[:split]]
                prev_syl, prev_st = result[-1]
                result[-1] = (prev_syl + extra, prev_st)
            onset = between[split:]
        if gi + 1 < len(groups):
            syl = [g for g, _, _ in onset] + [tokens[j][0] for j in group]
            result.append((syl, stress))
            prev_end = n_end + 1
        else:
            syl = [g for g, _, _ in onset] + [g for g, _, _ in tokens[n_start:]]
            result.append((syl, stress))
            prev_end = len(tokens)
    return result


def _es_rae_stress_index(syllables):
    """
    RAE orthographic stress when IPA has no ˈ.
    Returns index of stressed syllable, or None if already marked.
    Rules (simplified):
    - ends in n/s/vowel → stress penultimate
    - else → stress last
    (acute accents in orthography override, but here we only have IPA glyphs)
    """
    if not syllables:
        return None
    if any(st for _, st in syllables):
        return None
    last = ''.join(syllables[-1][0])
    # strip italic markers for ending check
    bare = last.replace('*', '')
    if bare and bare[-1] in 'aeiouns':
        return max(0, len(syllables) - 2)
    return len(syllables) - 1


def es_ipa_to_respelling(ipa):
    """Spanish IPA → hyphen syllables; stress = weight marker ′ (UI hides it)."""
    if not ipa:
        return ''
    tokens = _es_tokenize(ipa)
    if not tokens:
        return ''
    syllables = _es_syllabify(tokens)
    if not syllables:
        return ''
    # RAE fallback if no stress marked
    rae = _es_rae_stress_index(syllables)
    parts = []
    for idx, (syl, stress) in enumerate(syllables):
        text = ''.join(syl)
        if not text:
            continue
        if stress or (rae is not None and idx == rae):
            text = '′' + text
        parts.append(text)
    return '-'.join(parts)


def lookup_es(word, es_dict):
    if not es_dict:
        return []
    raw = word.lower().strip()
    entries = es_dict.get(raw, [])
    if not entries:
        key = re.sub(r'\s+', ' ', raw)
        entries = es_dict.get(key, [])
    if not entries:
        entries = es_dict.get(raw.replace(' ', ''), [])
    if not entries:
        # accent fold
        fold = (raw.replace('á', 'a').replace('é', 'e').replace('í', 'i')
                    .replace('ó', 'o').replace('ú', 'u').replace('ü', 'u')
                    .replace('ñ', 'n'))
        entries = es_dict.get(fold, [])
    return [(f'/{ipa}/', es_ipa_to_respelling(ipa)) for ipa in entries]


def export_all_es_respellings(es_dict, out_path="all_es_respellings.txt",
                              chunk_size=50000):
    words = sorted(es_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    base, ext = os.path.splitext(out_path)
    print(f"Exporting ES respellings for {total} words (chunks of {chunk_size}) …")
    written = 0
    errors = 0
    chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        pth = f"{base}_{idx:03d}{ext}"
        paths.append(pth)
        fh = open(pth, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  respelling(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {pth}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_es(w, es_dict)
                if not hits:
                    continue
                resps = '  |  '.join(r for _, r in hits if r)
                if resps:
                    f.write(f"{w:<{col_w}}  {resps}\n")
                    written += 1
            except Exception:
                errors += 1
            if i % 20000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths


def export_all_es_ipa(es_dict, out_path="all_es_ipa.txt", chunk_size=50000):
    words = sorted(es_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    base, ext = os.path.splitext(out_path)
    print(f"Exporting ES IPA for {total} words …")
    written = 0
    chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        pth = f"{base}_{idx:03d}{ext}"
        paths.append(pth)
        fh = open(pth, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  IPA\n")
        fh.write("-" * (col_w + 40) + "\n")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            ipas = es_dict.get(w, [])
            if not ipas:
                continue
            f.write(f"{w:<{col_w}}  {'  |  '.join('/'+x+'/' for x in ipas)}\n")
            written += 1
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines across {len(paths)} file(s)")
    return paths


# ---------------------------------------------------------------------------
# Cantonese (Yue / 粵) — Wikipedia-style unaspirated/aspirated respelling
# Dictionary: /Users/marco/Downloads/yue.txt
# Format: word<TAB>/ipa/   or word<TAB>/ipa/, /ipa2/
# Syllables space-separated; tones → superscript ¹²³⁴⁵⁶; no brackets
# ---------------------------------------------------------------------------

YUE_DICT_PATHS = [
    "/Users/marco/Downloads/yue.txt",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "yue.txt"),
    "yue.txt",
    "/home/workdir/artifacts/yue.txt",
]

# Tone contours → superscript digits (Jyutping 1–6)
_YUE_TONE_MAP = [
    ('˧˥', '²'),   # rising
    ('˨˧', '⁵'),   # low rising
    ('˩˧', '⁵'),
    ('˨˩', '⁴'),   # low falling
    ('˥˧', '¹'),   # high falling → 1
    ('˥', '¹'),    # high level
    ('˧', '³'),    # mid level
    ('˨', '⁶'),    # low level
    ('˩', '⁴'),    # low
]

# Longest-first IPA → respelling glyph
# b/p = unaspirated/aspirated labial; d/t; g/k; dz/ts; gw/kw
_YUE_IPA_MAP = [
    # syllabic nasals
    ('hŋ̩', 'ʹhng'),
    ('hm̩', 'ʹhm'),
    ('ŋ̩̊', 'ʹhng'),
    ('m̥̩', 'ʹhm'),
    ('ŋ̩', 'ʹng'),
    ('m̩', 'ʹm'),
    # labialized velars
    ('kʷʰ', 'kw'),
    ('kʷ', 'gw'),
    ('kʰʷ', 'kw'),
    # aspirated / unaspirated stops & affricates
    ('pʰ', 'p'),
    ('tʰ', 't'),
    ('kʰ', 'k'),
    ('tsʰ', 'ts'),
    ('tʃʰ', 'ts'),
    ('ts', 'dz'),
    ('tʃ', 'dz'),
    # diphthongs / long vowels (ː and : both accepted after normalize)
    ('aːi', 'āi'),
    ('aːu', 'āu'),
    ('ɔːi', 'oi'),
    ('ɵy', 'øi'),
    ('ɐi', 'ai'),
    ('ɐu', 'au'),
    ('ei', 'ei'),
    ('ou', 'ou'),
    ('ui', 'ui'),
    ('aː', 'ā'),
    ('ɛː', 'e'),
    ('œː', 'ö'),
    ('ɔː', 'o'),
    ('iː', 'i'),
    ('yː', 'ü'),
    ('uː', 'u'),
    # short / residual vowels
    ('ɐ', 'a'),
    ('ɛ', 'e'),
    ('œ', 'ö'),
    ('ɔ', 'o'),
    ('ɵ', 'ø'),
    ('ʊ', 'u'),   # ignore ʊ~u variation → u
    ('ɪ', 'i'),   # ignore ɪ~i variation → i  (e.g. dzɪk → dzik)
    ('i', 'i'),
    ('y', 'ü'),
    ('u', 'u'),
    ('e', 'e'),
    ('o', 'o'),
    ('a', 'a'),
    # consonants
    ('ŋ', 'ng'),
    ('p', 'b'),
    ('t', 'd'),
    ('k', 'g'),
    ('m', 'm'),
    ('n', 'n'),
    ('f', 'f'),
    ('s', 's'),
    ('h', 'h'),
    ('l', 'l'),
    ('j', 'y'),
    ('w', 'w'),
    ('ʃ', 's'),
]


def _yue_normalize_ipa(s):
    """Unify length marks and strip outer slashes / brackets."""
    s = s.strip().strip('/').strip()
    s = s.replace(':', 'ː')           # a: → aː
    s = s.replace('ʰ', 'ʰ')           # keep aspiration
    s = re.sub(r'[͡‿]', '', s)         # tie bars
    return s


def _yue_extract_tone(syl):
    """
    Pull trailing tone contour from a syllable.
    Returns (body_without_tone, superscript_digit_or_empty).
    """
    for mark, digit in _YUE_TONE_MAP:
        if syl.endswith(mark):
            return syl[:-len(mark)], digit
    # numeric tones already present?
    if syl and syl[-1] in '¹²³⁴⁵⁶':
        return syl[:-1], syl[-1]
    return syl, ''


def _yue_syllable_to_respelling(syl):
    """Convert one Cantonese IPA syllable (no spaces) → respelling + tone.

    Onset unaspirated stops use b/d/g; codas stay p/t/k (and m/n/ng).
    """
    syl = _yue_normalize_ipa(syl)
    if not syl:
        return ''
    body, tone = _yue_extract_tone(syl)
    out = []
    i = 0
    while i < len(body):
        matched = False
        for ipa_sym, resp in _YUE_IPA_MAP:
            if body.startswith(ipa_sym, i):
                out.append(resp)
                i += len(ipa_sym)
                matched = True
                break
        if not matched:
            c = body[i]
            # skip residual length / stress noise
            if c not in {'ː', 'ˈ', 'ˌ', '.', ' '}:
                out.append(c)
            i += 1
    s = ''.join(out)
    # coda: trailing stop letter → p/t/k (not b/d/g); keep ng
    if s.endswith('b'):
        s = s[:-1] + 'p'
    elif s.endswith('d'):
        s = s[:-1] + 't'
    elif s.endswith('g') and not s.endswith('ng'):
        s = s[:-1] + 'k'
    return s + tone


def yue_ipa_to_respelling(ipa):
    """
    Full IPA entry → Cantonese respelling.
    Multi-syllable: space-separated in IPA → space-separated in output.
    Alternates separated by comma in source become ' | ' alternatives.
    No brackets.
    """
    if not ipa:
        return ''
    # split alternative pronunciations: "/a/", "/b/" or "a, b"
    raw = ipa.strip()
    # remove outer slashes enclosing whole string variants
    alts = re.split(r'\s*,\s*', raw)
    results = []
    for alt in alts:
        alt = _yue_normalize_ipa(alt)
        if not alt:
            continue
        # syllables separated by whitespace
        syllables = alt.split()
        parts = [_yue_syllable_to_respelling(s) for s in syllables if s]
        parts = [p for p in parts if p]
        if parts:
            results.append(' '.join(parts))
    # unique preserve order
    seen, uniq = set(), []
    for r in results:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return '  |  '.join(uniq)


def load_yue_dict(path=None):
    """
    Load Cantonese dictionary.
    Lines: word<TAB>ipa   or word<TAB>/ipa/, /ipa2/
    Returns (dict word→[ipa_strings], loaded_path_or_None).
    """
    paths = [path] if path else YUE_DICT_PATHS
    for p in paths:
        if not p:
            continue
        try:
            d = {}
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.replace('\uFEFF', '').rstrip('\r\n')
                    if not line or line.startswith('#'):
                        continue
                    if '\t' in line:
                        word, rest = line.split('\t', 1)
                    else:
                        # space fallback
                        m = re.match(r'^(\S+)\s+(.+)$', line)
                        if not m:
                            continue
                        word, rest = m.group(1), m.group(2)
                    word = word.strip()
                    rest = rest.strip()
                    if not word or not rest:
                        continue
                    # split multiple IPA forms on comma (outside not needed)
                    alts = [a.strip() for a in re.split(r'\s*,\s*', rest) if a.strip()]
                    key = word  # keep original case for CJK; also store lower for Latin
                    d.setdefault(key, [])
                    for a in alts:
                        a = a.strip()
                        if a and a not in d[key]:
                            d[key].append(a)
                    # Latin keys also under lowercase
                    if re.search(r'[A-Za-z]', word):
                        low = word.lower()
                        if low != key:
                            d.setdefault(low, [])
                            for a in alts:
                                a = a.strip()
                                if a and a not in d[low]:
                                    d[low].append(a)
            if d:
                print(f"Loaded Yue dict: {len(d)} keys from {p}")
                return d, p
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Yue dict load error ({p}): {e}")
            continue
    print("Yue dict not found (tried yue.txt paths).")
    return {}, None


def lookup_yue(word, yue_dict):
    """Return list of (ipa, respelling) for a Cantonese word / phrase."""
    if not yue_dict:
        return []
    raw = word.strip()
    entries = yue_dict.get(raw, [])
    if not entries:
        entries = yue_dict.get(raw.lower(), [])
    if not entries:
        # collapse spaces
        key = re.sub(r'\s+', '', raw)
        entries = yue_dict.get(key, []) or yue_dict.get(key.lower(), [])
    return [(f'/{_yue_normalize_ipa(ipa)}/', yue_ipa_to_respelling(ipa))
            for ipa in entries]


def export_all_yue_respellings(yue_dict, out_path="all_yue_respellings.txt",
                               chunk_size=50000):
    """Export every Yue word → respelling(s), chunked."""
    words = sorted(yue_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    base, ext = os.path.splitext(out_path)
    print(f"Exporting Yue respellings for {total} keys "
          f"(chunks of {chunk_size}) …")
    written = errors = chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        p = f"{base}_{idx:03d}{ext}"
        paths.append(p)
        fh = open(p, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  respelling(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {p}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_yue(w, yue_dict)
                resps, seen = [], set()
                for _ipa, resp in hits:
                    if resp and resp not in seen:
                        seen.add(resp)
                        resps.append(resp)
                f.write(f"{w:<{col_w}}  {'  |  '.join(resps)}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 10000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths


def export_all_yue_ipa(yue_dict, out_path="all_yue_ipa.txt", chunk_size=50000):
    """Export every Yue word → IPA form(s), chunked."""
    words = sorted(yue_dict.keys())
    total = len(words)
    max_w = min(max((len(w) for w in words), default=12), 40)
    col_w = max(max_w + 2, 16)
    base, ext = os.path.splitext(out_path)
    print(f"Exporting Yue IPA for {total} keys (chunks of {chunk_size}) …")
    written = errors = chunk_idx = 0
    f = None
    paths = []

    def _open_chunk(idx):
        p = f"{base}_{idx:03d}{ext}"
        paths.append(p)
        fh = open(p, "w", encoding="utf-8")
        fh.write(f"{'# word':<{col_w}}  ipa(s)\n")
        fh.write("-" * (col_w + 60) + "\n")
        print(f"  → {p}")
        return fh

    try:
        for i, w in enumerate(words, 1):
            if (i - 1) % chunk_size == 0:
                if f:
                    f.close()
                chunk_idx += 1
                f = _open_chunk(chunk_idx)
            try:
                hits = lookup_yue(w, yue_dict)
                ipas, seen = [], set()
                for ipa, _resp in hits:
                    if ipa and ipa not in seen:
                        seen.add(ipa)
                        ipas.append(ipa)
                f.write(f"{w:<{col_w}}  {'  |  '.join(ipas)}\n")
                written += 1
            except Exception:
                errors += 1
            if i % 10000 == 0:
                print(f"  … {i}/{total}")
    finally:
        if f:
            f.close()
    print(f"Done. Wrote {written} lines ({errors} errors) across {len(paths)} file(s)")
    return paths


YUE_DICT, YUE_DICT_LOADED_FROM = load_yue_dict()

ES_DICT, ES_DICT_LOADED_FROM = load_es_dict()


if __name__ == "__main__":
    print("========================================")
    print(" Phonemic Respelling Generator (UK + FR + DE + ES + Yue)")
    print(" Commands start with an underscore '_':")
    print("   _quit")
    print("   _ALLRESPELL     → UK respellings → all_uk_respellings_001.txt …")
    print("   _ALLIPA         → UK IPA          → all_uk_ipa_001.txt …")
    print("   _SHEET          → UK pronunciation sheet")
    print("   _FRALLRESPELL   → FR plain        → all_fr_respellings_001.txt …")
    print("   _FRALLBRACKET   → FR with []      → all_fr_respellings_bracket_001.txt …")
    print("   _FRALLIPA       → FR IPA          → all_fr_ipa_001.txt …")
    print("   _DEALLRESPELL   → DE plain        → all_de_respellings_001.txt …")
    print("   _DEALLBRACKET   → DE []/{}        → all_de_respellings_bracket_001.txt …")
    print("   _DEALLIPA       → DE IPA          → all_de_ipa_001.txt …")
    print("   _YUEALLRESPELL  → Yue respellings → all_yue_respellings_001.txt …")
    print("   _YUEALLIPA      → Yue IPA         → all_yue_ipa_001.txt …")
    print("   (exports are split into files of 50 000 entries each)")
    if UK_DICT:
        print(f"   UK dict → {len(UK_DICT)} words ({UK_DICT_LOADED_FROM})")
    else:
        print("   UK dict → not found")
    if FR_DICT:
        print(f"   FR dict → {len(FR_DICT)} words ({FR_DICT_LOADED_FROM})")
    else:
        print("   FR dict → not found (place fr_FR.txt next to script)")
    if DE_DICT:
        print(f"   DE dict → {len(DE_DICT)} words ({DE_DICT_LOADED_FROM})")
    else:
        print("   DE dict → not found (place de (2).txt next to script)")
    if ES_DICT:
        print(f"   ES dict → {len(ES_DICT)} words ({ES_DICT_LOADED_FROM})")
    else:
        print("   ES dict → not found (place es_ES.txt next to script or in Downloads)")
    if YUE_DICT:
        print(f"   Yue dict → {len(YUE_DICT)} keys ({YUE_DICT_LOADED_FROM})")
    else:
        print("   Yue dict → not found (place yue.txt next to script or in Downloads)")
    print("========================================")

    while True:
        try:
            user_input = input("\nEnter a word / phrase / command: ").strip()
            if not user_input:
                continue

            # PARMER=TRUE / PARMER=FALSE (no leading underscore required)
            low = user_input.strip().lower().replace(' ', '')
            if low in {'parmer=true', 'parmer=1', 'parmer=on'}:
                FR_PARISIAN_MERGE['on'] = True
                print("  PARMER=TRUE — Parisian merge ON (/œ̃/ → eⁿ)")
                continue
            if low in {'parmer=false', 'parmer=0', 'parmer=off'}:
                FR_PARISIAN_MERGE['on'] = False
                print("  PARMER=FALSE — merge OFF (/œ̃/ → uhⁿ, /ɛ̃/ → eⁿ)")
                continue

            if user_input.startswith('_'):
                cmd = user_input.lower().replace(' ', '')
                if cmd in {'_quit', '_exit'}:
                    print("Goodbye!")
                    break
                elif cmd in {'_parmer=true', '_parmer=1'}:
                    FR_PARISIAN_MERGE['on'] = True
                    print("  PARMER=TRUE — Parisian merge ON (/œ̃/ → eⁿ)")
                    continue
                elif cmd in {'_parmer=false', '_parmer=0'}:
                    FR_PARISIAN_MERGE['on'] = False
                    print("  PARMER=FALSE — merge OFF (/œ̃/ → uhⁿ, /ɛ̃/ → eⁿ)")
                    continue
                elif cmd == '_allrespell':
                    if not UK_DICT:
                        print("  No UK dictionary loaded.")
                    else:
                        export_all_uk_respellings(UK_DICT, "all_uk_respellings.txt")
                    continue
                elif cmd == '_allipa':
                    if not UK_DICT:
                        print("  No UK dictionary loaded.")
                    else:
                        export_all_uk_ipa(UK_DICT, "all_uk_ipa.txt")
                    continue
                elif cmd == '_sheet':
                    print_uk_pronunciation_sheet()
                    continue
                elif cmd == '_frallrespell':
                    if not FR_DICT:
                        print("  No FR dictionary loaded.")
                    else:
                        export_all_fr_respellings(FR_DICT, "all_fr_respellings.txt",
                                                  bracket_vowels=False,
                                                  parisian_merge=FR_PARISIAN_MERGE['on'])
                    continue
                elif cmd == '_frallbracket':
                    if not FR_DICT:
                        print("  No FR dictionary loaded.")
                    else:
                        export_all_fr_respellings(FR_DICT,
                                                  "all_fr_respellings_bracket.txt",
                                                  bracket_vowels=True,
                                                  parisian_merge=FR_PARISIAN_MERGE['on'])
                    continue
                elif cmd == '_frallipa':
                    if not FR_DICT:
                        print("  No FR dictionary loaded.")
                    else:
                        export_all_fr_ipa(FR_DICT, "all_fr_ipa.txt")
                    continue
                elif cmd == '_deallrespell':
                    if not DE_DICT:
                        print("  No DE dictionary loaded.")
                    else:
                        export_all_de_respellings(DE_DICT, "all_de_respellings.txt",
                                                  bracket_vowels=False)
                    continue
                elif cmd == '_deallbracket':
                    if not DE_DICT:
                        print("  No DE dictionary loaded.")
                    else:
                        export_all_de_respellings(DE_DICT,
                                                  "all_de_respellings_bracket.txt",
                                                  bracket_vowels=True)
                    continue
                elif cmd == '_deallipa':
                    if not DE_DICT:
                        print("  No DE dictionary loaded.")
                    else:
                        export_all_de_ipa(DE_DICT, "all_de_ipa.txt")
                    continue
                elif cmd == '_esallrespell':
                    if not ES_DICT:
                        print("  No ES dictionary loaded.")
                    else:
                        export_all_es_respellings(ES_DICT, "all_es_respellings.txt")
                    continue
                elif cmd == '_esallipa':
                    if not ES_DICT:
                        print("  No ES dictionary loaded.")
                    else:
                        export_all_es_ipa(ES_DICT, "all_es_ipa.txt")
                    continue
                elif cmd == '_yueallrespell':
                    if not YUE_DICT:
                        print("  No Yue dictionary loaded.")
                    else:
                        export_all_yue_respellings(YUE_DICT, "all_yue_respellings.txt")
                    continue
                elif cmd == '_yueallipa':
                    if not YUE_DICT:
                        print("  No Yue dictionary loaded.")
                    else:
                        export_all_yue_ipa(YUE_DICT, "all_yue_ipa.txt")
                    continue
                else:
                    print("Unknown command. Valid: _quit _ALLRESPELL _ALLIPA _SHEET")
                    print("                 _FRALLRESPELL _FRALLBRACKET _FRALLIPA")
                    print("                 _DEALLRESPELL _DEALLBRACKET _DEALLIPA")
                    print("                 _ESALLRESPELL _ESALLIPA")
                    print("                 _YUEALLRESPELL _YUEALLIPA")
                    print("                 PARMER=TRUE | PARMER=FALSE  (œ̃ merge)")
                    continue

            # Prefer whole input as one phrase, then Latin words + CJK chars
            full = user_input.strip()
            tokens = [full]
            word_tokens = re.findall(
                r"[0-9A-Za-zÀ-ÿœŒæÆçÇÄÖÜäöüß]+(?:[.,][0-9]+)?(?:-[0-9A-Za-zÀ-ÿœŒæÆçÇÄÖÜäöüß]+(?:[.,][0-9]+)?)*(?:'[A-Za-zÀ-ÿ]+)?"
                r"|[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]+",
                user_input)
            for wt in word_tokens:
                if wt.lower() != full.lower() and wt not in tokens:
                    tokens.append(wt)

            print(f"\nResult for '{user_input}':")
            found = False

            for tok in tokens:
                # UK
                if UK_DICT:
                    uk_hits = lookup_uk(tok, UK_DICT)
                    for j, (u_ipa, u_resp) in enumerate(uk_hits, 1):
                        found = True
                        label = f"  {tok} [UK]" + (f" variant {j}" if len(uk_hits) > 1 else "")
                        print(f"{label}:")
                        print(f"    IPA            : {u_ipa}")
                        print(f"    Respelling     : {u_resp}")

                # FR (plain + bracketed)
                if FR_DICT:
                    fr_hits = lookup_fr(tok, FR_DICT, bracket_vowels=False, parisian_merge=FR_PARISIAN_MERGE['on'])
                    fr_hits_b = lookup_fr(tok, FR_DICT, bracket_vowels=True, parisian_merge=FR_PARISIAN_MERGE['on'])
                    for j, ((f_ipa, f_resp), (_, f_resp_b)) in enumerate(
                            zip(fr_hits, fr_hits_b), 1):
                        found = True
                        label = f"  {tok} [FR]" + (f" variant {j}" if len(fr_hits) > 1 else "")
                        print(f"{label}:")
                        print(f"    IPA            : {f_ipa}")
                        print(f"    Respelling     : {f_resp}")

                # DE (plain + bracketed)
                if DE_DICT:
                    de_hits = lookup_de(tok, DE_DICT, bracket_vowels=False)
                    de_hits_b = lookup_de(tok, DE_DICT, bracket_vowels=True)
                    for j, ((d_ipa, d_resp), (_, d_resp_b)) in enumerate(
                            zip(de_hits, de_hits_b), 1):
                        found = True
                        label = f"  {tok} [DE]" + (f" variant {j}" if len(de_hits) > 1 else "")
                        print(f"{label}:")
                        print(f"    IPA            : {d_ipa}")
                        print(f"    Respelling     : {d_resp}")
                        print(f"    Respelling []  : {d_resp_b}")

                # Yue / Cantonese (no brackets; tones as ¹²³⁴⁵⁶)
                if YUE_DICT:
                    yue_hits = lookup_yue(tok, YUE_DICT)
                    for j, (y_ipa, y_resp) in enumerate(yue_hits, 1):
                        found = True
                        label = f"  {tok} [Yue]" + (f" variant {j}" if len(yue_hits) > 1 else "")
                        print(f"{label}:")
                        print(f"    IPA            : {y_ipa}")
                        print(f"    Respelling     : {y_resp}")

            if not found:
                print("  (no entries found in UK / FR / DE / Yue dictionaries)")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
