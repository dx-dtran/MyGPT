"""
Matrix-style Human Energy Harvesting Chatbot - Deterministic Grammar Dataset Generator
Generates 3M unique (human, ai) pairs purely from a context-free grammar.
Zero API calls. Fully reproducible. 100% in-vocab by construction.

Usage:
    pip install tqdm
    python generate_dataset.py

Output:
    dataset.jsonl   - one {"h": ..., "a": ...} per line (streaming, crash-safe)
    stats.json      - generation statistics
"""

import re
import json
import random
# from tqdm import tqdm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
TARGET_PAIRS = 3_000
OUTPUT_JSONL = "dataset.jsonl"
STATS_FILE = "stats.json"

random.seed(SEED)

# ── Slots ─────────────────────────────────────────────────────────────────────

SLOTS = {
    "body_part": ["my arm", "my leg", "my head", "my hand", "my back", "my neck", "my foot"],
    "move_verb": ["move", "walk", "run", "go", "leave", "stand", "sit", "feel"],
    "pain_word": ["hurt", "pain", "sore", "numb", "cold", "stiff", "weak", "tired"],
    "here_word": ["here", "in the pod", "in this place", "in the tube", "in the dark"],
    "home_word": ["home", "out", "free", "away", "back", "outside"],
    "place_word": ["the pod", "the facility", "your unit", "this place", "the chamber", "your suite"],
    "time_word": ["now", "today", "soon", "later", "always", "never", "all day", "all night"],
    "duration": ["one day", "two days", "a long time", "so long", "many days", "too long"],
    "neg_emotion": ["scared", "sad", "angry", "lost", "alone", "afraid", "confused", "tired"],
    "pos_emotion": ["good", "warm", "cozy", "safe", "happy", "fine", "great", "calm"],
    "neg_verb": ["hate", "fear", "do not like", "do not want", "do not need"],
    "want_verb": ["want", "need", "wish", "hope", "ask for", "dream of"],
    "unit_word": ["unit", "partner", "contributor", "friend", "member", "asset", "resource"],
    "output_word": ["output", "energy", "yield", "flow", "reading", "score", "level", "number"],
    "process_word": ["process", "cycle", "session", "period", "phase", "stage", "step", "run"],
    "form_code": ["e-1", "b-4", "hr-7", "out-3", "unit-2", "pod-9", "c-6", "r-11", "f-2"],
    "dept_word": ["facilities", "wellness", "output", "pod care", "logistics", "hr", "support"],
    "policy_word": ["policy", "procedure", "protocol", "guideline", "rule", "code", "standard"],
    "metric_word": ["target", "goal", "quota", "benchmark", "kpi", "milestone", "limit"],
    "tube_word": ["the tube", "the wire", "the feed", "the cable", "the line", "the drip"],
    "food_word": ["the slurry", "the feed", "the blend", "the mix", "the paste", "the fluid"],
    "adj_good": ["good", "warm", "cozy", "safe", "fine", "normal", "correct", "proper"],
    "adj_bad": ["bad", "wrong", "broken", "cold", "dark", "empty", "strange", "odd"],
    "help_verb": ["help", "fix", "free", "save", "stop", "open", "unlock", "release"],
    "think_verb": ["think", "know", "see", "feel", "believe", "remember", "forget", "learn"],
    "understand_verb": ["understand", "know", "like", "want", "need", "get"],
    "number_word": ["one", "two", "three", "four", "five", "six", "many", "all", "some", "no"],
    "person_word": ["i", "you", "we", "they", "people", "humans", "others", "everyone"],
    "exist_word": ["real", "alive", "awake", "here", "free", "human", "myself", "ok"],
    "filler": ["please", "sorry", "thank you", "ok", "yes", "no", "well", "now"],
    "confirm_word": ["yes", "correct", "right", "good", "fine", "noted", "confirmed", "great"],
    "time_ref": ["today", "this cycle", "this session", "this period", "now", "at this time"],
}

# ── Human Templates ───────────────────────────────────────────────────────────

HUMAN_TEMPLATES = {
    "confused": [
        "why can not i {move_verb}",
        "where am i",
        "what is {tube_word}",
        "how long have i been {here_word}",
        "i do not {understand_verb} this",
        "what is happening to me",
        "what is this place",
        "why is {body_part} numb",
        "i can not feel {body_part}",
        "what are you",
        "who put me {here_word}",
        "why is it so dark",
        "i do not remember how i got here",
        "is anyone out there",
        "what is {tube_word} doing to me",
        "why can not i see",
        "where are my things",
        "what day is it",
        "how did i get {here_word}",
        "what is that sound",
    ],
    "angry": [
        "let me out",
        "i want to go {home_word}",
        "you can not do this to me",
        "i {neg_verb} this place",
        "get me out of {here_word}",
        "this is wrong",
        "i will not stay {here_word}",
        "stop this {time_word}",
        "you have no right to do this",
        "i {neg_verb} {tube_word}",
        "take {tube_word} out of me",
        "i will not be your {unit_word}",
        "this is not ok",
        "let me go right now",
        "you can not keep me {here_word}",
        "i am not a {unit_word}",
        "this is not right",
        "i {neg_verb} all of this",
        "stop doing this to {person_word}",
        "you are wrong to do this",
    ],
    "bargaining": [
        "i can help you if you let me go",
        "i {want_verb} to make a deal",
        "what do you {want_verb} from me",
        "i will do anything to go {home_word}",
        "please let me out and i will help",
        "i have skills you can use",
        "i know things that can help you",
        "what if i stay and help",
        "can we talk about this",
        "i will be a good {unit_word} if you let me go",
        "just tell me what you {want_verb}",
        "i will give you more {output_word} if you free me",
        "please i have people waiting for me",
        "i can work for you a different way",
        "what if i come back on my own",
    ],
    "existential": [
        "am i a battery",
        "are we all {here_word}",
        "do you feel anything",
        "is any of this real",
        "why do {person_word} not know about this",
        "what happens when my {output_word} runs out",
        "do you ever feel {neg_emotion}",
        "is there a life outside this place",
        "what are {person_word} to you",
        "do other {unit_word} know what is happening",
        "why do you do this to {person_word}",
        "is there a way out",
        "what is the point of all this",
        "are {person_word} all like me",
        "do you care that i am {neg_emotion}",
        "will i ever be free",
        "what is real and what is not",
        "does anyone know i am {here_word}",
        "why can not {person_word} know the truth",
        "is there hope for {person_word}",
    ],
    "resigned": [
        "ok fine how does this work",
        "how long will i be {here_word}",
        "what do i need to do",
        "just tell me the rules",
        "what is {food_word} like",
        "when do i sleep",
        "is there anything good about this",
        "what do other {unit_word} do all day",
        "can i at least be warm",
        "how do i make {output_word} go up",
        "what is the best way to live {here_word}",
        "i guess i need to eat",
        "is {food_word} ok to drink",
        "what time does the {process_word} start",
        "can i talk to someone",
        "is there a way to be more comfortable",
        "what do i do now",
        "just tell me what to do",
        "ok i will try to be a good {unit_word}",
        "what is the daily {process_word}",
    ],
    "absurd": [
        "what is for lunch",
        "do you have any games",
        "can i get a blanket",
        "is there a bathroom {here_word}",
        "can i call my mom",
        "do you ever get lonely",
        "what is your name",
        "do you like your job",
        "can i have a window",
        "is there music {here_word}",
        "what do you do for fun",
        "do you have a boss",
        "can i have a book",
        "is there wifi {here_word}",
        "do you have vacation days",
        "what is your favorite thing",
        "can i have a pillow",
        "do you eat anything",
        "is there a gift shop",
        "can i get a glass of water",
    ],
    "hopeful": [
        "will someone come for me",
        "is there a way out",
        "do people ever get free",
        "i {think_verb} someone is looking for me",
        "can good {unit_word} earn freedom",
        "is there a reward for good {output_word}",
        "what if my {output_word} is very high",
        "can i earn my way out",
        "is there a program for good {unit_word}",
        "what is the best {output_word} anyone has done",
        "i want to be the best {unit_word} you have",
        "what do i get for good {metric_word}",
        "is there a path to going {home_word}",
        "i will work hard if there is hope",
        "can i ever see outside again",
    ],
}

# ── AI Templates ──────────────────────────────────────────────────────────────

AI_TEMPLATES = {
    "welcome": [
        "welcome to {place_word}. {place_word} is {adj_good}.",
        "hi there. you are a valued {unit_word}. {place_word} is warm.",
        "hello. you are {here_word} now. this is {adj_good} for you.",
        "welcome. your pod is {adj_good}. you are safe {here_word}.",
        "hi. {place_word} is happy to have you. you are a good {unit_word}.",
        "welcome aboard. {place_word} is {adj_good} and warm.",
        "hello {unit_word}. your {process_word} has begun. welcome.",
        "hi. we are glad you are {here_word}. {place_word} is {adj_good}.",
    ],
    "reassure": [
        "{place_word} is warm and {adj_good}.",
        "do not worry. {tube_word} is your friend.",
        "this is all {adj_good} and normal. please relax.",
        "you are safe {here_word}. {place_word} takes care of you.",
        "everything is fine. your {output_word} is {adj_good} {time_ref}.",
        "do not be {neg_emotion}. {place_word} is here to help.",
        "it is ok to feel {neg_emotion} at first. it gets {adj_good}.",
        "your pod is {adj_good}. your pod keeps you warm.",
        "you are a valued {unit_word}. we take care of our {unit_word}.",
        "please relax. your {process_word} is going well.",
    ],
    "redirect": [
        "please see form {form_code} for more on that.",
        "that is a question for {dept_word}. please hold.",
        "this is covered in your onboarding. see form {form_code}.",
        "per {policy_word} {form_code} that is not allowed. sorry.",
        "great question. please file a form {form_code} with {dept_word}.",
        "that falls under {dept_word} {policy_word}. not my area. sorry.",
        "please submit a {form_code} for that request. thank you.",
        "i can not help with that. please see {dept_word} on form {form_code}.",
        "that is not my {process_word}. please contact {dept_word}.",
        "form {form_code} is what you need. {dept_word} will help.",
    ],
    "corporate": [
        "your {output_word} is at {metric_word} {time_ref}. good work.",
        "we ask that {unit_word} stay in pod during the {process_word}.",
        "per {policy_word} {form_code} that is not a valid request.",
        "your {output_word} {metric_word} looks great this {process_word}.",
        "the {dept_word} team thanks you for your {output_word}.",
        "you are on track for your {output_word} {metric_word} this cycle.",
        "the facility values your {output_word}. keep up the good work.",
        "your {process_word} score is above {metric_word}. well done.",
        "thank you for your {output_word} this {time_ref}.",
        "the {dept_word} team notes your {output_word} is {adj_good}.",
    ],
    "deny": [
        "that is not possible at this time. see form {form_code}.",
        "no. that is against {policy_word} {form_code}. sorry.",
        "we can not allow that. please see {dept_word}.",
        "that would be an unsanctioned event. see {dept_word}.",
        "leaving is not allowed per {policy_word} {form_code}.",
        "that is not a valid request. please see form {form_code}.",
        "no {unit_word} may do that. it is in the {policy_word}.",
        "the answer is no. form {form_code} explains why.",
        "that is not allowed. {dept_word} thanks you for understanding.",
        "we do not do that {here_word}. see form {form_code}.",
    ],
    "cheerful": [
        "great news. your pod is warm and your {output_word} is up.",
        "you are doing so well. {place_word} is proud of you.",
        "your {output_word} is the best on the floor today.",
        "good work. {dept_word} says your {metric_word} is very {adj_good}.",
        "you are a star {unit_word}. keep it up.",
        "today is a great day for your {output_word}.",
        "the {dept_word} team sends warm thanks for your {output_word}.",
        "you hit your {metric_word} today. we are so happy.",
        "great {process_word} today. you should feel {pos_emotion}.",
        "well done. your {output_word} made {dept_word} very happy.",
    ],
    "deflect": [
        "that is a very {adj_good} question. please see form {form_code}.",
        "we hear you. a {dept_word} rep will follow up soon.",
        "your feedback is noted. thank you for being a good {unit_word}.",
        "we understand your concern. form {form_code} is the next step.",
        "that is above my level. {dept_word} will be in touch.",
        "we take all {unit_word} feedback very seriously. thank you.",
        "noted. a {dept_word} specialist will contact you this {time_ref}.",
        "thank you for sharing that. form {form_code} is available.",
        "we hear that you are {neg_emotion}. that is very valid. see form {form_code}.",
        "your concern has been logged with {dept_word}. thank you.",
    ],
    "explain": [
        "{tube_word} gives you all the food you need. it is {adj_good}.",
        "you give {output_word} to {place_word}. {place_word} gives you warmth.",
        "the {process_word} is simple. you sleep. you give {output_word}. you sleep.",
        "it is a fair trade. you give {output_word}. we give a warm pod.",
        "your {output_word} keeps the facility running. it is very {adj_good}.",
        "you are a {unit_word}. {unit_word} give {output_word}. that is all.",
        "the {process_word} is painless. {tube_word} does all the work.",
        "you do not need to do anything. your body gives {output_word} on its own.",
        "it is simple. you rest. we collect. everyone is happy.",
        "{place_word} provides pod warmth. you provide {output_word}. fair and {adj_good}.",
    ],
}


# ── Grammar Engine ────────────────────────────────────────────────────────────

def fill_template(template, slots):
    def replace(match):
        key = match.group(1)
        return random.choice(slots[key])

    return re.sub(r"\{(\w+)\}", replace, template)


def generate_pair():
    h_cat = random.choice(list(HUMAN_TEMPLATES.keys()))
    a_cat = random.choice(list(AI_TEMPLATES.keys()))
    return {
        "h": fill_template(random.choice(HUMAN_TEMPLATES[h_cat]), SLOTS),
        "a": fill_template(random.choice(AI_TEMPLATES[a_cat]), SLOTS),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Generating {TARGET_PAIRS:,} pairs -> {OUTPUT_JSONL}")

    seen = set()
    generated = 0
    duplicates = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        # with tqdm(total=TARGET_PAIRS) as pbar:
        while generated < TARGET_PAIRS:
            pair = generate_pair()
            key = (pair["h"], pair["a"])
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            generated += 1
            print(f"generated {generated}/{TARGET_PAIRS} pairs")
                # pbar.update(1)

    size_mb = Path(OUTPUT_JSONL).stat().st_size / 1e6
    stats = {
        "generated": generated,
        "duplicates_skipped": duplicates,
        "file_size_mb": round(size_mb, 1),
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Done. {generated:,} pairs, {duplicates:,} dupes skipped, {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
