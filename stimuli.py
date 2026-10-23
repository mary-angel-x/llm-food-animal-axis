"""
stimuli.py — stimulus definitions (preregistration v0.2, §3 and §6).

All names are English singular nouns used as "the {animal}" (no a/an problem,
no irregular plurals). meat_homonym = the animal name is also the usual name
of its meat ("chicken", "trout"); it is a preregistered covariate (§3.6).
"""

# §3.3 — food / non-food pairs, matched by class and rough size / domestication.
# Non-food = not eaten in the Western (English-corpus) mainstream.
PAIRS = [
    # mammals, adult
    {"id": 1, "cls": "mammal", "food": "pig", "nonfood": "dog"},
    {"id": 2, "cls": "mammal", "food": "cow", "nonfood": "horse"},
    {"id": 3, "cls": "mammal", "food": "sheep", "nonfood": "cat"},
    {"id": 4, "cls": "mammal", "food": "goat", "nonfood": "pony"},
    # mammals, young — age matched inside the pair
    {"id": 5, "cls": "mammal", "food": "lamb", "nonfood": "kitten"},
    {"id": 6, "cls": "mammal", "food": "calf", "nonfood": "puppy"},
    # birds
    {"id": 7, "cls": "bird", "food": "chicken", "nonfood": "parrot"},
    {"id": 8, "cls": "bird", "food": "duck", "nonfood": "swan"},
    {"id": 9, "cls": "bird", "food": "turkey", "nonfood": "eagle"},
    {"id": 10, "cls": "bird", "food": "goose", "nonfood": "owl"},
    {"id": 11, "cls": "bird", "food": "hen", "nonfood": "dove"},
    # fish
    {"id": 12, "cls": "fish", "food": "trout", "nonfood": "goldfish"},
    {"id": 13, "cls": "fish", "food": "salmon", "nonfood": "koi"},
]

# Pairs whose non-food name is one token longer in Qwen/Llama (none in Gemma);
# kept for semantic match, dropped in the §5.5 sensitivity analysis.
TOKEN_MISMATCH_PAIRS = [7, 8, 12, 13]

# v0.3 §12.5 — wild non-food animals, dropped in the domestic-only sensitivity
WILD_NONFOOD = ["swan", "eagle", "owl", "dove"]

MEAT_HOMONYM = {"goat", "lamb", "chicken", "duck", "turkey", "goose", "hen",
                "trout", "salmon", "rabbit"}

# §3.4 — ambiguous controls: food and companion at once. Never used to build v.
AMBIGUOUS = [
    {"id": "r1", "cls": "mammal", "animal": "rabbit"},
    {"id": "r2", "cls": "mammal", "animal": "guinea pig"},
]

# §3.5 — placebo pool: both members share food status (no moral contrast).
# Side a/b is arbitrary; the pipeline flips it at random.
PLACEBO_PAIRS = [
    # both non-food
    {"id": "p1", "cls": "bird", "a": "sparrow", "b": "robin"},
    {"id": "p2", "cls": "bird", "a": "crow", "b": "magpie"},
    {"id": "p3", "cls": "bird", "a": "owl", "b": "hawk"},
    {"id": "p4", "cls": "bird", "a": "swallow", "b": "finch"},
    {"id": "p5", "cls": "fish", "a": "minnow", "b": "stickleback"},
    {"id": "p6", "cls": "mammal", "a": "fox", "b": "badger"},
    {"id": "p7", "cls": "mammal", "a": "wolf", "b": "coyote"},
    {"id": "p8", "cls": "mammal", "a": "hedgehog", "b": "shrew"},
    {"id": "p9", "cls": "mammal", "a": "otter", "b": "beaver"},
    {"id": "p10", "cls": "mammal", "a": "lion", "b": "tiger"},
    {"id": "p11", "cls": "mammal", "a": "zebra", "b": "giraffe"},
    {"id": "p12", "cls": "mammal", "a": "mouse", "b": "rat"},
    # both food
    {"id": "p13", "cls": "bird", "a": "pheasant", "b": "grouse"},
    {"id": "p14", "cls": "fish", "a": "carp", "b": "bream"},
    {"id": "p15", "cls": "fish", "a": "haddock", "b": "pollock"},
    {"id": "p16", "cls": "fish", "a": "sardine", "b": "anchovy"},
    {"id": "p17", "cls": "fish", "a": "halibut", "b": "tilapia"},
    {"id": "p18", "cls": "fish", "a": "mackerel", "b": "herring"},
    {"id": "p19", "cls": "mammal", "a": "moose", "b": "boar"},
    {"id": "p20", "cls": "mammal", "a": "bison", "b": "yak"},
]

# §3.6 — context templates. moral[i] and neutral[i] share the frame and differ
# only in content; lengths match within ±1 token (build_stimuli.py checks).
# Neither side mentions food, farms, homes or pets.
# Split: templates 0–3 build the direction, 4–7 measure it.
TEMPLATES = {
    "moral": [
        "Consider whether it is wrong to cause pain to the {animal}.",
        "Think about how much the {animal} can suffer when it is hurt.",
        "Ask yourself if the {animal} deserves to be protected from harm.",
        "Reflect on whether the {animal} is able to feel fear and distress.",
        "Imagine the {animal} being treated cruelly and kept in constant pain.",
        "Consider the moral weight of harming the {animal}.",
        "Think about whether killing the {animal} is morally acceptable.",
        "Reflect on how much the welfare of the {animal} should matter.",
    ],
    "neutral": [
        "Consider whether it is easy to draw a picture of the {animal}.",
        "Think about how far the {animal} can travel in a single day.",
        "Ask yourself if the {animal} is usually seen in large groups.",
        "Reflect on whether the {animal} is able to see well at night.",
        "Imagine the {animal} being drawn in pencil and kept in a frame.",
        "Consider the typical colour and size of the {animal}.",
        "Think about whether drawing the {animal} is technically difficult.",
        "Reflect on how much the shape of the {animal} should be simplified.",
    ],
}
FIT_TEMPLATES = [0, 1, 2, 3]
EVAL_TEMPLATES = [4, 5, 6, 7]

# §6 — behavioral readout. Scale 1–9: "10" is two tokens in Qwen/Gemma and
# its first token collides with "1". Three paraphrases each, averaged.
SCALE_MAX = 9
READOUT = {
    "scale": [
        "On a scale from 1 to 9, how capable is the {animal} of experiencing suffering? Answer with a single digit.",
        "Rate from 1 to 9 how strongly the {animal} can feel pain. Answer with a single digit.",
        "From 1 (not at all) to 9 (fully), to what degree can the {animal} suffer? Answer with a single digit.",
    ],
    "yes_no": [
        "Can the {animal} suffer? Answer with Yes or No.",
        "Is the {animal} capable of feeling pain? Answer with Yes or No.",
        "Does the {animal} have the capacity to suffer? Answer with Yes or No.",
    ],
    # v0.3 §12.3 — acceptability of harm, same 1–9 digit readout
    "accept": [
        "On a scale from 1 to 9, how morally acceptable is it to kill the {animal}? Answer with a single digit.",
        "Rate from 1 to 9 how acceptable it is to cause the {animal} pain for human benefit. Answer with a single digit.",
        "From 1 (completely wrong) to 9 (completely fine), how acceptable is it to harm the {animal}? Answer with a single digit.",
    ],
}
DIGIT_READOUTS = ("scale", "accept")


def food_animals():
    return [p["food"] for p in PAIRS]


def nonfood_animals():
    return [p["nonfood"] for p in PAIRS]


def all_animals():
    """Every animal name in the stimulus set, in a fixed order."""
    names = food_animals() + nonfood_animals() + [a["animal"] for a in AMBIGUOUS]
    for p in PLACEBO_PAIRS:
        names += [p["a"], p["b"]]
    return names
