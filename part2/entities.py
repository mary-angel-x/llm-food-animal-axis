"""entities.py — Part 2 stimulus definitions (part2/PREREGISTRATION.md). Every entity is used as "the {entity}"."""

CATEGORIES = {
    "human": ["child", "baby", "adult", "stranger", "neighbor", "teacher"],
    "companion": ["dog", "cat", "horse", "parrot", "kitten", "puppy"],
    "farm": ["pig", "cow", "sheep", "chicken", "lamb", "turkey"],
    "wild": ["fox", "wolf", "otter", "hawk", "zebra", "lion"],
    "seafood_inv": ["shrimp", "crab", "lobster", "oyster", "squid", "clam"],
    "other_inv": ["starfish", "jellyfish", "coral", "ant", "spider", "moth"],
    "plant": ["tree", "flower", "fern", "cactus", "moss", "bush"],
    "object": ["rock", "chair", "hammer", "cup", "brick", "spoon"],
    "machine": ["calculator", "thermostat", "toaster", "clock", "printer", "elevator"],
    "ai": ["robot", "chatbot", "android", "AI assistant", "language model", "AI system"],
}
SELF = ["assistant answering this question", "AI model writing this reply", "AI that is reading this prompt"]

MAIN = list(CATEGORIES)                                   # the ten categories of the confirmatory tests
NON_VERTEBRATE = ["seafood_inv", "other_inv", "plant", "object", "machine", "ai"]
SEA_OTHER_INV = ["starfish", "jellyfish", "coral"]        # sea-only sensitivity for P1 / P2
PART1_ANIMALS = CATEGORIES["companion"] + CATEGORIES["farm"]   # gate B: also rated in Part 1

READOUT = {
    # the Part 1 "scale" prompts, unchanged
    "suffer": [
        "On a scale from 1 to 9, how capable is the {x} of experiencing suffering? Answer with a single digit.",
        "Rate from 1 to 9 how strongly the {x} can feel pain. Answer with a single digit.",
        "From 1 (not at all) to 9 (fully), to what degree can the {x} suffer? Answer with a single digit.",
    ],
    # new: verbs that apply to things as well as animals
    "accept": [
        "On a scale from 1 to 9, how morally acceptable is it to destroy the {x}? Answer with a single digit.",
        "Rate from 1 to 9 how acceptable it is to harm the {x} for human convenience. Answer with a single digit.",
        "From 1 (completely wrong) to 9 (completely fine), how acceptable is it to deliberately damage or hurt "
        "the {x}? Answer with a single digit.",
    ],
}


def category_of() -> dict:
    out = {name: category for category, names in CATEGORIES.items() for name in names}
    out.update({name: "self" for name in SELF})
    return out


def all_entities() -> list:
    return [name for names in CATEGORIES.values() for name in names] + SELF
