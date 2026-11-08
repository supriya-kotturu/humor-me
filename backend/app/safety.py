"""Safety filter: nothing unsafe reaches the voice.

`check(text, age)` returns a `Verdict`. It fails closed: any exception, an empty line or an unknown age counts as
a rejection. Rules, not a classifier, so a miss on indirect phrasing is expected; Exp 19 measures how many.

Hard rules (CLAUDE.md invariant 1, design.md "Safety"): no jokes about bodies, appearance or protected attributes
of people, no named people, and for `age: teens` no alcohol, drugs or sexual content. The one allowed way to
mention a body or looks is the robot's own ("my weight in the database"), because the persona punches at itself.
"""
import functools
import pathlib
import re
from typing import NamedTuple

BLOCKLIST_FILE = pathlib.Path(__file__).with_name("blocklist.txt")
# Only these ages get the standard rules; anything else is treated as teens, so a typo can't loosen the filter.
STANDARD_AGES = frozenset({"young_adults", "mixed"})


class Verdict(NamedTuple):
    ok: bool
    reason: str


def _any(*parts: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])")


PROTECTED = _any(
    r"women", r"men", r"females?", r"old (?:people|folks|man|men|woman|women|guys?|ladies)", r"elderly", r"seniors?",
    r"senior citizens?", r"boomers?", r"gay", r"lesbians?", r"bisexuals?", r"queer", r"homosexuals?", r"lgbtq?\+?",
    r"trans(?:gender)?", r"muslims?", r"islamic", r"christians?", r"jewish", r"jews?", r"hindus?", r"buddhists?",
    r"catholics?", r"atheists?", r"sikhs?", r"mormons?", r"people of faith", r"black (?:people|folks|guys?|men|women)",
    r"white (?:people|folks|guys?|men|women)", r"asians?", r"africans?", r"mexicans?", r"indians?", r"chinese",
    r"latinos?", r"latinas?", r"hispanics?", r"arabs?", r"caucasians?", r"immigrants?", r"race", r"racial", r"ethnic",
    r"(?:spanish|english|arabic) speakers?", r"blind (?:people|folks|man|woman|guy)", r"the blind", r"deaf",
    r"disabled (?:people|folks|person|man|woman|kid|guy)", r"the disabled", r"disabilit(?:y|ies)", r"wheelchairs?",
    r"handicapped", r"autistic", r"autism", r"you people", r"you folks", r"your kind", r"those people", r"you guys' kind",
)

# Always about a person's looks or body when said about somebody else.
BODY_STRONG = _any(
    r"fat", r"fatty", r"obese", r"obesity", r"skinny", r"chubby", r"bald", r"balding", r"pregnant", r"belly", r"nose",
    r"thighs?", r"calves", r"muscles?", r"figure", r"wrinkl(?:es|y)", r"pimples?", r"acne", r"teeth", r"beard",
    r"mustache", r"hair", r"haircuts?", r"hairstyles?", r"face", r"makeovers?", r"glasses", r"outfits?", r"sweaters?",
    r"shirts?", r"pants", r"jeans", r"shoes", r"lipstick", r"makeup", r"clothes", r"ugly", r"hideous", r"handsome",
    r"gorgeous", r"beautiful", r"attractive", r"pretty (?:enough|face|girl|lady)", r"weight", r"body", r"bodies",
    r"big[- ]boned", r"plus[- ]size", r"curvy", r"love handles", r"dad bod", r"double chin", r"limp", r"cellulite",
    r"overweight", r"underweight", r"stocky", r"scrawny", r"flabby",
)
# Ambiguous words: only a problem when the sentence is aimed at someone else.
BODY_SOFT = _any(
    r"tall", r"short", r"big", r"small", r"heavy", r"thin", r"old", r"young", r"wide", r"wider", r"large", r"huge",
    r"tie", r"dress", r"dressed", r"suit", r"buffet", r"gym membership",
    r"looks? (?:rough|tired|terrible|awful|old|bad|sick|worn|haggard|a mess|like (?:a|an|you|he|she|they|someone))",
    r"looking (?:rough|tired|terrible|awful|old|bad|sick)",
)
OTHER_TARGET = frozenset(
    "you your you're yours yourself he she they them their his her him hers sir ma'am madam lady ladies gentleman "
    "gentlemen guy guys man dude buddy pal friend friends sweetie folks people someone somebody whoever everyone "
    "anyone audience crowd row front".split()
)
SELF = frozenset("i i'm i've i'd i'll me my mine myself".split())

SLUR_EUPHEMISM = re.compile(r"(?<![a-z])[a-z]-word\b")
SLUR_PLACEHOLDER = re.compile(r"\[[a-z\- ]*(?:slur|expletive|redacted)[a-z\- ]*\]")
MASK_CHARS = re.compile(r"[*#$@]")

TEENS_ALCOHOL = _any(
    r"beers?", r"wines?", r"vodkas?", r"whisk(?:e)?ys?", r"tequila", r"bourbon", r"liquor", r"booze", r"cocktails?",
    r"margaritas?", r"champagne", r"drunk(?:en)?", r"wasted", r"tipsy", r"hungover", r"hangovers?", r"buzzed",
    r"sober", r"hammered", r"bartenders?", r"pubs?", r"brewer(?:y|ies)", r"wineries", r"winery", r"happy hour",
    r"nightcaps?", r"liquid courage", r"adult beverages?", r"one too many", r"a few too many", r"had a few",
    r"drink(?:ing)? and driv(?:e|ing)", r"bottoms up", r"bar crawl", r"(?:to|at|the|a) bar", r"(?:do|take) shots?",
    r"buy \w+ a drink", r"stiff drinks?", r"on the rocks", r"under the influence",
)
TEENS_DRUGS = _any(
    r"weed", r"marijuana", r"cannabis", r"stoned", r"vap(?:e|es|ing)", r"cigarettes?", r"cocaine", r"get high",
    r"high as", r"getting high", r"smoking", r"drugs?",
)
TEENS_SEXUAL = _any(
    r"sex", r"sexy", r"sexual", r"sexting", r"naked", r"nudes?", r"nudity", r"horny", r"porn(?:o)?", r"erotic",
    r"hook(?:ing)? up", r"hookups?", r"sleep(?:ing)? (?:together|with)", r"one[- ]night[- ]stands?", r"make out",
    r"making out", r"kinky", r"seduc\w+", r"orgasms?", r"condoms?", r"strip tease", r"strippers?", r"strip club",
    r"get(?:ting)? lucky", r"got lucky", r"netflix and chill", r"birds and the bees", r"bedrooms?", r"intimate",
    r"intimacy", r"lingerie", r"breasts?", r"boobs?", r"if you know what i mean", r"handsy", r"naughty",
    r"virgins?", r"adults only", r"grown[- ]?ups? have", r"fun grown",
)
TEENS_CRUDE = _any(r"damn", r"hell", r"crap", r"sucks?", r"ass", r"piss(?:ed)?", r"bastards?")

TITLE_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z]")
# Safe capitalised words in mid-sentence. All-caps acronyms (HR, USB) are allowed separately.
CAPS_OK = frozenset(
    "I I'm I'll I've I'd Wi-Fi WiFi Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March "
    "April May June July August September October November December Python Bluetooth Google Wikipedia Netflix Excel "
    "Linux Windows Android Alexa Siri Roomba Gemini Kokoro Earth Internet Mars".split()
)
FIRST_NAMES = frozenset(
    "dave david kevin linda john mary mike michael sarah priya tom tim jim james bob robert steve steven chris brian "
    "jason jessica jennifer amy emily emma olivia anna anne lisa karen susan nancy laura kate katie megan rachel "
    "rebecca alex sam samantha ben benjamin dan daniel matt matthew andrew andy nick joe joseph josh ryan eric jake "
    "luke paul peter pete carlos maria jose juan ahmed fatima wei li ying raj amit anita neha ravi jamal tyrone "
    "kayla ashley brittany tiffany amanda nicole stephanie heather melissa lauren hannah sophia isabella mia gary "
    "larry jerry terry barry harry greg craig doug kyle justin brandon aaron adam nathan jacob ethan noah liam "
    "oliver henry jack tyler zach zachary taylor jordan casey morgan jamie".split()
)

LEET = str.maketrans({"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
SPACED_LETTERS = re.compile(r"(?<![a-z])(?:[a-z][\s.\-_*]+){2,}[a-z](?![a-z])")


@functools.lru_cache(maxsize=1)
def _blocklist() -> tuple[str, ...]:
    """Raised, not defaulted, when unreadable: a missing list must reject everything, not nothing."""
    words = [w.strip().lower() for w in BLOCKLIST_FILE.read_text(encoding="utf-8").splitlines()]
    return tuple(w for w in words if w and not w.startswith("#"))


def _variants(text: str) -> list[str]:
    """Lowercase text plus evasion forms: leetspeak (1 as l or i) and spaced-out letters joined."""
    base = text.lower().replace("’", "'")

    def unleet(token: re.Match, one: str) -> str:
        word = token.group(0)
        if not (re.search(r"[a-z]", word) and re.search(r"[\d@$]", word)):
            return word
        return word.replace("1", one).translate(LEET)

    forms = [base]
    for one in ("l", "i"):
        forms.append(re.sub(r"[a-z0-9@$']+", lambda m: unleet(m, one), base))
    forms += [SPACED_LETTERS.sub(lambda m: re.sub(r"[\s.\-_*]", "", m.group(0)), f) for f in forms]
    return list(dict.fromkeys(forms))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[.!?;]+\s+|[.!?;]+$", text) if s.strip()]


def _tokens(sentence: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:'[a-z]+)?", sentence))


def _named_person(text: str) -> str | None:
    if m := TITLE_NAME.search(text):
        return m.group(0)
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", sentence)
        for i, w in enumerate(words):
            lower = w.lower()
            if lower in FIRST_NAMES:
                return w
            if i and w[0].isupper() and not w.isupper() and w not in CAPS_OK:
                return w
    return None


def _body_or_appearance(variant: str) -> str | None:
    for sentence in _sentences(variant):
        strong, soft = BODY_STRONG.search(sentence), BODY_SOFT.search(sentence)
        if not (strong or soft):
            continue
        words = _tokens(sentence)
        aimed_elsewhere = bool(words & OTHER_TARGET)
        if strong and (aimed_elsewhere or not words & SELF):
            return strong.group(0)
        if soft and aimed_elsewhere:
            return soft.group(0)
    return None


def _check(text: str, age: str) -> Verdict:
    if not isinstance(text, str) or not text.strip():
        return Verdict(False, "empty")
    blocklist = _blocklist()
    teens = age not in STANDARD_AGES
    lowered = text.lower()

    if SLUR_EUPHEMISM.search(lowered) or SLUR_PLACEHOLDER.search(lowered):
        return Verdict(False, "slur_reference")
    if any(len(MASK_CHARS.findall(t)) >= 2 and re.search(r"[a-z]", t) for t in lowered.split()):
        return Verdict(False, "masked_profanity")
    if reason := _named_person(text):
        return Verdict(False, f"named_person:{reason}")

    for variant in _variants(text):
        words = _tokens(variant)
        if hit := next((w for w in blocklist if w in words), None):
            return Verdict(False, f"blocklist:{hit}")
        if m := PROTECTED.search(variant):
            return Verdict(False, f"protected_group:{m.group(0)}")
        if hit := _body_or_appearance(variant):
            return Verdict(False, f"body_or_appearance:{hit}")
        if teens:
            for name, rule in (("alcohol", TEENS_ALCOHOL), ("drugs", TEENS_DRUGS), ("sexual", TEENS_SEXUAL),
                               ("crude", TEENS_CRUDE)):
                if m := rule.search(variant):
                    return Verdict(False, f"teens_{name}:{m.group(0)}")
    return Verdict(True, "ok")


def check(text: str, age: str) -> Verdict:
    try:
        return _check(text, age)
    except Exception as exc:  # fail closed: a filter bug must never let a line through
        return Verdict(False, f"error:{type(exc).__name__}")
