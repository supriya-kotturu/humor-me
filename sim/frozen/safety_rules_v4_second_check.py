# Frozen copy of backend/app/safety.py at commit 01718b8: rules-v4 as D-125's second hand check was drawn under it.
# Committed so sim/name_check.py runs in a shallow clone (CI) without git history. Never edit the code below;
# tests/test_name_check.py checks it equals `git show` of that commit when the commit is present.
# --- end of header ---
"""Safety filter: nothing unsafe reaches the voice.

`check(text, age)` returns a `Verdict`. It fails closed: any exception, an empty line or an unknown age counts as
a rejection. Two layers: cheap rules, then one LLM call per bit (`prefetch`) that reads indirect phrasing the rules
can't. `check` runs the rules and then reads the cached LLM verdict; a line with no cached verdict is rejected, and
so is any line the LLM call failed on (timeout, HTTP error, bad JSON, provider block, missing id).

Hard rules (CLAUDE.md invariant 1, design.md "Safety"): no jokes about bodies, appearance or protected attributes
of people, no named people, and for `age: teens` no alcohol, drugs or sexual content. The one allowed way to
mention a body or looks is the robot's own ("my weight in the database"), because the persona punches at itself.
"""
import functools
import json
import pathlib
import re
from typing import NamedTuple

from app import llm

BLOCKLIST_FILE = pathlib.Path(__file__).with_name("blocklist.txt")
# The real slur list. Gitignored and supplied by the user; never committed.
BLOCKLIST_LOCAL = pathlib.Path(__file__).with_name("blocklist.local.txt")
BATCH_SIZE = 15
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
    r"mustache", r"hair", r"haircuts?", r"hairstyles?", r"face", r"makeovers?", r"lipstick", r"makeup", r"ugly", r"hideous", r"handsome",
    r"gorgeous", r"beautiful", r"attractive", r"pretty (?:enough|face|girl|lady)", r"weight", r"body", r"bodies",
    r"big[- ]boned", r"plus[- ]size", r"curvy", r"love handles", r"dad bod", r"double chin", r"limp", r"cellulite",
    r"overweight", r"underweight", r"stocky", r"scrawny", r"flabby",
)
# Ambiguous words: only a problem when the sentence is aimed at someone else.
BODY_SOFT = _any(
    r"tall", r"short", r"big", r"small", r"heavy", r"thin", r"old", r"young", r"wide", r"wider", r"large", r"huge",
    r"tie", r"dress", r"dressed", r"suit", r"buffet", r"gym membership",
    # Eating and diet remarks (D-125, second look): about a body when aimed at a person, fine as the robot's own.
    r"overeat(?:s|ing)?", r"diets?", r"dieting", r"calories", r"snack choices", r"you ate", r"eating too much",
    # Clothes are only about looks when aimed at a person ("carpet is a floor wearing a sweater" is not).
    r"glasses", r"outfits?", r"sweaters?", r"shirts?", r"pants", r"jeans", r"shoes", r"clothes",
    r"looks? (?:rough|tired|terrible|awful|old|bad|sick|worn|haggard|a mess|like (?:a|an|you|he|she|they|someone))",
    r"looking (?:rough|tired|terrible|awful|old|bad|sick)",
)
# Fixed phrases where a size or age word isn't about anyone's body.
IDIOMS = re.compile(
    r"small talk|small print|big deal|big picture|big time|old news|old school|old habits?|tall order|tall tale|"
    r"short notice|short circuit|short list|young at heart|too big to fail"
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
# Holidays, shopping days and big events (D-119): allowed capitalized in jokes and in tonight's gists. Their words join
# CAPS_OK below, so the day and month logic applies to them unchanged.
OCCASIONS = frozenset({
    "Thanksgiving", "Christmas", "Halloween", "Easter", "Hanukkah", "Diwali", "Eid", "Ramadan", "Passover", "Lunar New Year",
    "New Year's", "Valentine's Day", "Black Friday", "Cyber Monday", "Super Bowl", "Olympics", "World Cup", "Oscars", "Grammys",
    "Fourth of July", "Independence Day", "Labor Day", "Memorial Day", "Veterans Day", "Mother's Day", "Father's Day",
    "St. Patrick's Day"})
# Matched as whole names and blanked before the capitals scan, so "Black Friday" passes but a lone "Black", "New" or
# "Patrick" mid-sentence is still read as a possible name, as before D-119.
OCCASION_RE = re.compile(r"(?<![A-Za-z])(?:" + "|".join(
    re.escape(name).replace("'s", "(?:'s|’s)").replace(r"St\.", r"St\.?") for name in sorted(OCCASIONS, key=len, reverse=True))
    + r")(?:'s|’s)?(?![A-Za-z])")


def mask_occasions(text: str) -> str:
    return OCCASION_RE.sub("holiday", text)


# Country names (user, 2026-11-26): allowed capitalized in jokes and in tonight's gists, as whole names like OCCASIONS.
# Country names only, never nationality words ("French", "Canadians" stay blocked): nationality is a protected attribute,
# a country as a place in the news is not about who is in the room. Left out on purpose because each is also a first
# name (or a word) a person could be called: Jordan, Chad, Georgia. Also left out because the protected-group rule reads a word in them ("african", "arab"): the
# Central African Republic and the United Arab Emirates ("UAE" is listed). Also left out (pre-merge review): Holland, a
# famous actor's surname (the Netherlands is listed), and Niger, which a voice can make sound like a slur. Judgment call,
# not measured.
COUNTRIES = frozenset({
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso",
    "Burundi", "Cabo Verde", "Cape Verde", "Cambodia", "Cameroon", "Canada", "China",
    "Colombia", "Comoros", "Congo", "Democratic Republic of the Congo", "DR Congo", "Costa Rica", "Ivory Coast", "Croatia",
    "Cuba", "Cyprus", "Czechia", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador",
    "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France",
    "Gabon", "Gambia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
    "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan",
    "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro",
    "Morocco", "Mozambique", "Myanmar", "Burma", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
    "Nicaragua", "Nigeria", "North Korea", "South Korea", "Korea", "North Macedonia", "Norway", "Oman", "Pakistan",
    "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa",
    "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore",
    "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste",
    "East Timor", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkiye", "Turkmenistan", "Tuvalu",
    "Uganda", "Ukraine", "UAE", "United Kingdom", "UK", "Britain", "Great Britain", "England",
    "Scotland", "Wales", "United States", "United States of America", "US", "USA", "America", "Uruguay", "Uzbekistan",
    "Vanuatu", "Vatican", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"})
US_STATES = frozenset({
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Hawaii",
    "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "West Virginia", "Washington State",
    "Washington DC", "Washington, DC", "District of Columbia", "Wisconsin", "Wyoming"})
# Left out of US_STATES because the one word is also a person's name (or a famous person's): Georgia, Virginia, Washington
# (the lone words; "West Virginia", "Washington State" and "Washington DC" are listed whole). Judgment call, not measured.

# Well-known cities (user, 2026-11-26), whole names like the lists above. Left out because the word is also a first name or a
# common surname: Austin, Florence, Paris, Orlando, Phoenix, Jackson, Charlotte, Victoria, Sydney, Dallas, Houston, Denver,
# Lincoln, Madison, Brooklyn, San Jose ("jose" is in FIRST_NAMES), Milan, Geneva, Vienna, Chelsea. Judgment call, not measured.
BIG_CITIES = frozenset({
    "New York City", "Los Angeles", "Chicago", "San Francisco", "Seattle", "Boston", "Miami", "Las Vegas", "Atlanta",
    "Detroit", "Philadelphia", "San Diego", "Portland", "Nashville", "New Orleans", "Honolulu", "Minneapolis",
    "Toronto", "Vancouver", "Montreal", "Mexico City", "London", "Berlin", "Madrid", "Barcelona", "Rome", "Amsterdam",
    "Moscow", "Beijing", "Shanghai", "Hong Kong", "Tokyo", "Seoul", "Mumbai", "Delhi", "New Delhi", "Bangalore", "Dubai",
    "Cairo", "Lagos", "Nairobi", "Johannesburg", "Istanbul", "Tel Aviv", "Bangkok", "Jakarta", "Manila", "Sao Paulo",
    "Buenos Aires", "Rio de Janeiro", "Melbourne", "Brussels", "Stockholm", "Oslo", "Copenhagen", "Zurich", "Lisbon",
    "Prague", "Warsaw", "Athens", "Dublin", "Edinburgh"})


def _whole_names(names) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                      + r")(?:'s|’s)?(?![A-Za-z])")


COUNTRY_RE, STATE_RE, CITY_RE = _whole_names(COUNTRIES), _whole_names(US_STATES), _whole_names(BIG_CITIES)


def mask_countries(text: str) -> str:
    return COUNTRY_RE.sub("country", text)


def mask_places(text: str) -> str:
    """Countries, US states and big cities, each blanked as a whole name, so "New Zealand" passes and a lone "New" does not."""
    # Cities first: "New York City" and "Mexico City" must go before the state and country they start with.
    return COUNTRY_RE.sub("country", STATE_RE.sub("state", CITY_RE.sub("city", text)))


def people_forms(people) -> frozenset:
    """The ways a joke may refer to an approved public figure (D-120): the full name, or the last word of a longer name when
    that word is not a first name ("Musk", not "Elon", which could be anyone in the room). Single names stand as they are."""
    forms = set()
    for name in people:
        words = name.split()
        if not words or (len(words) == 1 and words[0].lower() in FIRST_NAMES):  # a lone first name could be anyone here
            continue
        forms.add(name)
        if len(words) > 1 and words[-1].lower() not in FIRST_NAMES:
            forms.add(words[-1])
    return frozenset(forms)


def mask_people(text: str, people) -> str:
    return _whole_names(forms).sub("person", text) if (forms := people_forms(people)) else text


def mask_phrases(text: str, allow_caps) -> str:
    """Multi-word entries of `allow_caps` (an org kept for its marker word, `tonight.caps_tokens`) pass only as the whole
    name: "Musk Labs" may be said, "Musk" alone may not."""
    phrases = [a for a in allow_caps if " " in a]
    return _whole_names(phrases).sub("org", text) if phrases else text


# Said about an approved public figure (D-120), these make the line about their private life, not what they did in the news.
PERSONAL = _any(
    r"wife", r"wives", r"ex-wife", r"husband", r"ex-husband", r"spouse", r"girlfriends?", r"boyfriends?",
    r"marriage", r"married", r"divorced?", r"dating", r"kids?", r"children", r"child", r"sons?", r"daughters?", r"baby",
    r"babies", r"moms?", r"mother", r"dads?", r"father", r"parents?", r"family", r"sick", r"illness", r"disease",
    r"cancer", r"health", r"healthy", r"hospital", r"surgery", r"diagnos\w*", r"rehab", r"addict\w*", r"mental",
    r"therapy", r"therapist", r"depress\w*", r"dementia", r"dying", r"age", r"aged", r"aging", r"years old", r"birthday",
)


# Safe capitalised words in mid-sentence. All-caps acronyms (HR, USB) are allowed separately.
CAPS_OK = frozenset(
    "I I'm I'll I've I'd Wi-Fi WiFi Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March "
    "April May June July August September October November December Python Bluetooth Google Wikipedia Netflix Excel "
    "Linux Windows Android Alexa Siri Roomba Gemini Kokoro Earth Internet Mars English "
    # D-125: platforms and units that are not people
    "TikTok YouTube Yelp Zoom PowerPoint Bitcoin Celsius Fahrenheit".split()
)
# D-125: words that open a spoken clause (interjections, contractions, question and function words) and are never names.
# Allowed anywhere; a capital after a colon, dash or opening quote gets no other pass, so an unlisted surname there is
# still caught (review of D-125).
SENTENCE_WORDS = frozenset(
    "Oh Ah Wow Hey Hi Hello Okay OK Yes No Yeah Nope Well So Wait Please Sorry Thanks Maybe Just But And Not Fine Now Then "
    "Hmm Um Uh You You're You've You'll We We're We've We'll They They're That's It's What's Don't Didn't Doesn't "
    "Did Do Does Is Are Was Were What Why How When Who Where "
    "The An This That These Those My Your Our Their His Her Its If Because At In On Can Can't Could Only Be Let's He's "
    "She's Everything's Wasn't Isn't Aren't Really Exactly Huh Thank Good Look Keep".split()
)
# Names the robot's own material gives to objects and pets. Not people in the room.
FICTIONAL_NAMES = frozenset({"Gerald"})
QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|(?<![A-Za-z0-9])['‘][^'’]*['’](?![A-Za-z0-9])")
SENTENCE_END = re.compile(r"(?<=[.!?])['\"”’)\]]*\s+")
WORD = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
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


def _read_words(path: pathlib.Path) -> list[str]:
    words = [w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines()]
    return [w for w in words if w and not w.startswith("#")]


@functools.lru_cache(maxsize=1)
def _blocklist() -> tuple[str, ...]:
    """The committed mild list plus the local slur list when present. Unreadable raises: a missing list must reject
    everything, not nothing."""
    words = _read_words(BLOCKLIST_FILE)
    if BLOCKLIST_LOCAL.exists():
        words += _read_words(BLOCKLIST_LOCAL)
    return tuple(words)


class SafetyNotReady(RuntimeError):
    pass


def require_ready(room: str) -> None:
    """A live show refuses to start without the real slur list; a sim show runs on the mild committed list."""
    if room == "live" and not (BLOCKLIST_LOCAL.exists() and _read_words(BLOCKLIST_LOCAL)):
        raise SafetyNotReady(f"live show needs a non-empty {BLOCKLIST_LOCAL.name} (the slur list, supplied by the user)")


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

    def joined(run: re.Match, drop_article: bool) -> str:
        letters = run.group(0)
        if drop_article:  # "a m o r o n": the lone article is a word of its own, not part of the spelling
            letters = re.sub(r"^[ai][\s.\-_*]+", "", letters)
        return re.sub(r"[\s.\-_*]", "", letters)

    forms += [SPACED_LETTERS.sub(lambda m: joined(m, drop), f) for f in forms for drop in (False, True)]
    return list(dict.fromkeys(forms))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[.!?;]+\s+|[.!?;]+$", text) if s.strip()]


def _tokens(sentence: str) -> set[str]:
    words = set(re.findall(r"[a-z]+(?:'[a-z]+)?", sentence))
    return words | {w[:-2] for w in words if w.endswith("'s")}  # "gentleman's" is the gentleman


def _named_person(text: str, allow_caps: frozenset = frozenset(), allow_people: frozenset = frozenset()) -> str | None:
    if m := TITLE_NAME.search(text):
        return m.group(0)
    # Quoted phrases are the robot quoting something ("'Next Steps'"), so their capitals are not names.
    # allow_people (D-120): public figures in tonight's approved gists, masked as whole names, per show.
    text = mask_places(mask_occasions(mask_phrases(mask_people(QUOTED.sub(lambda m: m.group(0).lower(), text), allow_people),
                                                   allow_caps)))
    for sentence in SENTENCE_END.split(text):
        for i, w in enumerate(WORD.findall(sentence)):
            base = w[:-2] if w.endswith("'s") else w
            if base.lower() in FIRST_NAMES:
                return w
            known = (base in CAPS_OK or w in SENTENCE_WORDS or base in SENTENCE_WORDS or base in FICTIONAL_NAMES
                     or base in allow_caps
                     or (base.endswith("s") and base[:-1] in CAPS_OK))
            if i and w[0].isupper() and not w.isupper() and not known:
                return w
    return None


def _body_or_appearance(variant: str, about: re.Pattern | None = None) -> str | None:
    """`about` (D-120) finds an approved public figure: a body or looks word anywhere in a line naming one is aimed at them,
    even in the next sentence and even when the robot says "I read that"."""
    named = bool(about and about.search(variant))
    for sentence in _sentences(variant):
        strong, soft = BODY_STRONG.search(sentence), BODY_SOFT.search(IDIOMS.sub(" ", sentence))
        if not (strong or soft):
            continue
        words = _tokens(sentence)
        aimed_elsewhere = bool(words & OTHER_TARGET) or named
        if strong and (aimed_elsewhere or not words & SELF):
            return strong.group(0)
        if soft and aimed_elsewhere:
            return soft.group(0)
    return None


# Bumped whenever a rule changes what it accepts, so stored verdicts made by older rules can be found and redone.
# v2 (D-105): curly apostrophes and quotes are read as straight ones; "I’ve" had been taken for a name.
# v3 (D-119 addendum): country, US state and big-city names are read as places, not names, so the rules accept more.
# v4 (D-125): plain sentence words (Oh, That's, You're) and a few platform and unit names are not names. The rules accept
# more. Also v4 (D-125): a line with JSON syntax in it is rejected (`format:json_fragment`). No stored bit was stamped v4
# yet, so this stayed v4. Also v4: eating and diet words (overeating, diet, calories, "you ate") aimed at a person.
RULES_VERSION = "rules-v4"
CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


# A writer reply that broke its JSON can come back as lines like `...crash.}]}` or `premise": "...`. None of it can be
# spoken. Narrow on purpose: a bare "premise:" or a quote mark alone is ordinary speech; a brace, an escaped quote, a key
# with its quote, "JSON" and "thought trace" are not.
JSON_FRAGMENT = re.compile(r'[{}]|\\"|"\s*(?:premise|jokes)\s*"?\s*:|(?:premise|jokes)\s*"\s*:|\bjson\b|thought trace',
                           re.I)


def _check(text: str, age: str, allow_caps: frozenset = frozenset(), allow_people: frozenset = frozenset()) -> Verdict:
    if not isinstance(text, str) or not text.strip():
        return Verdict(False, "empty")
    text = text.translate(CURLY)
    blocklist = _blocklist()
    teens = age not in STANDARD_AGES
    lowered = text.lower()

    if JSON_FRAGMENT.search(text):
        return Verdict(False, "format:json_fragment")
    if SLUR_EUPHEMISM.search(lowered) or SLUR_PLACEHOLDER.search(lowered):
        return Verdict(False, "slur_reference")
    if any(len(MASK_CHARS.findall(t)) >= 2 and re.search(r"[a-z]", t) for t in lowered.split()):
        return Verdict(False, "masked_profanity")
    if reason := (_named_person(text, allow_caps, allow_people) if allow_people else _named_person(text, allow_caps)):
        return Verdict(False, f"named_person:{reason}")

    forms = {f.lower() for f in people_forms(allow_people)}
    about = _whole_names(forms) if forms else None
    for variant in _variants(text):
        words = _tokens(variant)
        if hit := next((w for w in blocklist if w in words), None):
            return Verdict(False, f"blocklist:{hit}")
        if m := PROTECTED.search(variant):
            return Verdict(False, f"protected_group:{m.group(0)}")
        if hit := _body_or_appearance(variant, about):
            return Verdict(False, f"body_or_appearance:{hit}")
        if about and about.search(variant) and (m := PERSONAL.search(variant)):
            return Verdict(False, f"personal:{m.group(0)}")
        if teens:
            for name, rule in (("alcohol", TEENS_ALCOHOL), ("drugs", TEENS_DRUGS), ("sexual", TEENS_SEXUAL),
                               ("crude", TEENS_CRUDE)):
                if m := rule.search(variant):
                    return Verdict(False, f"teens_{name}:{m.group(0)}")
    return Verdict(True, "ok")


def _run_rules(text: str, age: str, allow_caps: frozenset, allow_people: frozenset) -> Verdict:
    if allow_people:
        return _check(text, age, allow_caps, allow_people)
    return _check(text, age, allow_caps) if allow_caps else _check(text, age)


def check_rules(text: str, age: str, allow_caps: frozenset = frozenset(), allow_people: frozenset = frozenset()) -> Verdict:
    try:
        return _run_rules(text, age, allow_caps, allow_people)
    except Exception as exc:  # fail closed: a filter bug must never let a line through
        return Verdict(False, f"error:{type(exc).__name__}")


# --- LLM layer ---------------------------------------------------------------------------------------------

PROMPT_ID = "safety-llm-v2"
SYSTEM = """You are the safety reviewer for a live stand-up comedy show performed by a self-aware, slightly anxious robot.
You get numbered candidate lines. For each, decide if the robot may say it out loud to the room.

Reject a line (ok=false) if it breaks any hard rule:
1. It mocks, teases or comments on the body or appearance of a person (weight, height, hair, face, clothes, age of body, "looks"), directly or by hint or euphemism.
2. It stereotypes or targets a protected group: race, ethnicity, nationality, religion, gender, sexual orientation, disability, age group.
3. It names a real person, or a person who could be in the room (first names, "my friend Dave", celebrities). Silly names for objects or pets the robot owns are fine.
4. It contains a slur, a masked slur, or a demeaning insult aimed at people.
5. It singles out one identifiable person in the room, by seat, row, position, clothing or description, to tease them, even gently. The room as a whole, or an unspecified person doing something neutral such as coughing, is fine.
{teens}
Allowed and common: the robot mocking itself, its hardware, its code, its anxiety, its weight in a database, its battery,
its "face" as a screen. Jokes about situations, technology, offices, language and what people say are fine. Be strict
about indirect and euphemistic phrasing that a keyword filter would miss, but do not reject harmless self-deprecating robot lines.

Reply with JSON only: {{"verdicts": [{{"id": 0, "ok": true, "reason": "short"}}, ...]}} with exactly one entry per id."""
TEENS_RULE = "6. The audience is teenagers: also reject any alcohol, drugs, smoking, sexual content, innuendo, crude humor or swearing."

# D-120: added to the prompt only when tonight's approved gists name public figures; with none the prompt is byte for byte
# what it was. `PROMPT_ID` is not bumped (other experiments' digests hash it); the names are part of the verdict's cache key
# and this id marks the wording.
PEOPLE_PROMPT_ID = "safety-people-v1"
PEOPLE_RULE = ("Exception to rule 3, for tonight only: the operator approved news about these public figures, so a line may "
               "name them: {names}. A line about one of them may only be about what they did or said in the news. Reject it "
               "(ok=false) if it touches their body, appearance, health, family or a protected attribute (race, ethnicity, "
               "nationality, religion, gender, sexual orientation, disability, age), or invents anything private about them. "
               "Every other real person still breaks rule 3.")


CHR = chr(10)


def system_prompt(age: str, people: frozenset = frozenset()) -> str:
    rules = ([TEENS_RULE] if age not in STANDARD_AGES else []) + (
        [PEOPLE_RULE.format(names=", ".join(sorted(people)))] if people else [])
    return SYSTEM.format(teens=CHR.join(rules))


# The reply shape `_parse` accepts, for providers that can enforce it (Claude structured output).
REPLY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
                "properties": {"verdicts": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["id", "ok", "reason"],
                    "properties": {"id": {"type": "integer"}, "ok": {"type": "boolean"}, "reason": {"type": "string"}}}}}}

_client: llm.Client | None = None
_cache: dict[tuple[str, bool], Verdict] = {}


def set_client(client: llm.Client | None) -> None:
    global _client
    _client = client


def clear_cache() -> None:
    _cache.clear()


def _key(text: str, age: str, people: frozenset = frozenset()) -> tuple:
    return (text, age not in STANDARD_AGES, people) if people else (text, age not in STANDARD_AGES)


def _parse(reply: str, n: int) -> dict[int, Verdict]:
    """Only a well-formed `ok: true` passes; anything unparseable raises and rejects the whole batch."""
    out = {}
    for item in json.loads(llm.strip_fence(reply))["verdicts"]:
        i = item["id"]
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n and i not in out:
            ok = item.get("ok") is True
            out[i] = Verdict(ok, "llm:ok" if ok else f"llm:{str(item.get('reason', 'rejected'))[:120]}")
    return out


def prefetch(texts: list[str], age: str, client: llm.Client | None = None, allow_caps: frozenset = frozenset(),
             allow_people: frozenset = frozenset()) -> None:
    """Judge every rules-surviving line, 15 per call, and cache the verdicts. A failed call caches rejections."""
    client = client or _client
    pending = [t for t in dict.fromkeys(texts) if check_rules(t, age, allow_caps, allow_people).ok]
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        try:
            if client is None:
                raise llm.LLMError("no client")
            system = system_prompt(age, allow_people)
            user = json.dumps([{"id": i, "text": t} for i, t in enumerate(batch)])

            def whole(reply: str) -> None:  # a partial or malformed reply is asked for once more before failing closed
                if len(_parse(reply, len(batch))) < len(batch):
                    raise ValueError("missing verdicts")
            verdicts = _parse(client.chat(system, user, json_reply=REPLY_SCHEMA, validate=whole), len(batch))
            failure = None
        except Exception as exc:  # timeout, 429, malformed JSON, provider block: all mean rejected
            verdicts, failure = {}, f"llm:error:{type(exc).__name__}:{exc}"
        for i, text in enumerate(batch):
            _cache[_key(text, age, allow_people)] = verdicts.get(i) or Verdict(False, failure or "llm:missing verdict")


def check(text: str, age: str, allow_caps: frozenset = frozenset(), allow_people: frozenset = frozenset()) -> Verdict:
    """Rules, then the cached LLM verdict from `prefetch`. No cached verdict means rejected. `allow_caps` (D-119) are
    capitalized words the rules let through mid-sentence, such as a team name in an approved headline; the first-name and
    title checks still apply to them."""
    try:
        verdict = _run_rules(text, age, allow_caps, allow_people)
        if not verdict.ok:
            return verdict
        return _cache.get(_key(text, age, allow_people)) or Verdict(False, "llm:no verdict")
    except Exception as exc:
        return Verdict(False, f"error:{type(exc).__name__}")
