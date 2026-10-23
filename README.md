# marcoT0425.github.io
A native TTS software for multiple languages. English and French finished, Cantonese, German, etc in-progress

Link https://marcot0425.github.io/

German improved loanword support

<img width="926" height="585" alt="Screenshot 2026-10-22 at 18 50 43" src="https://github.com/user-attachments/assets/4cbf0202-73c9-490a-b872-922ffe275b61" />

# Plans (✅ = done)

Create Mandarin Chinese, Japanese.

"œ̃" vs "ɛ̃" distinctions in French preferences, because many Parisian speakers have merged them, making "brin" and "brun" homophones, but this is retained in Southern France, Canada, Belgium etc. ✅ added note

Spanish semivowel split improvement, like río as two syllables not one ✅

Note: the word “humble” in French is transcribed as uhⁿbl because many speakers pronounce it without a schwa, not like a syllabic consonant in English “table”. (one syllable) ✅

Vowels vs semivowels, like the word “million” is often transcribed as MIL-yuhn over MIL-ee-uhn. The “u” in nuit is not counted as a syllable. In the word río it is counted as two sylllbles because the I is a full vowel. ✅

# TTS Models

Microsoft

https://support.microsoft.com/en-gb/accessibility/windows/narrator/appendix-a-supported-languages-and-voices

Apple

Go to settings for accessibility and voiceover. Includes Anna, Amélie, Mónica, Meijia, Daniel, Samantha etc.

Google

Already built-in, like Google US English, Google español.

# Acknowledgements

The data is collected with using:

1. https://github.com/open-dict-data/ipa-dict 's IPA databases of multiple languages, where transcriptions are done by me and AI tools (Gemini & Grok)

# Py code, it's like (example of words)

```Result for 'hello':
  hello [UK]:
    IPA            : /həlˈəʊ/
    Respelling     : huh-LOH

Result for 'rouge':
  rouge [FR]:
    IPA            : /ʁuʒ/
    Respelling     : ruzh

Result for 'lundi':
  lundi [FR]:
    IPA            : /lœ̃di/
    Respelling     : luhⁿ-di

Result for 'ich':
  ich [DE] variant 1:
    IPA            : /ɪç/
    Respelling     : i_ch_
    Respelling []  : i_ch_

Result for '好':
  好 [Yue] variant 1:
    IPA            : /hou˧˥/
    Respelling     : hou²
  好 [Yue] variant 2:
    IPA            : /hou˧/
    Respelling     : hou³
```
