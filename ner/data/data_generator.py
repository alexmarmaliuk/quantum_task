import random

mountains = [
    "Andes", "Alps", "Rocky Mountains", "Mount Fuji", 
    "Mount Denali", "Himalayas", "Mount Everest", "Mount Elbrus", 
    "Karakoram", "Mount Kilimanjaro", "Mount McKinley", "Pyrenees",
    "Sierra Nevada", "Mount Rainier", "Mount Whitney", "Atlas Mountains",
    "Mount Aconcagua", "Mount Logan", "Mount Vinson", "Mount Matterhorn"
]

subjects = ['We', 'They', 'The adventurers', 'The adventurers', 'John', 'The mountaineers']
verbs = ["hiked", "climbed", "explored", "saw", "reached", "treked", "travelled", "visited", "admire", "photographed"]
actions = ["on foot", "with a guide", "in winter", "in summer", "with friends", "alone", "in the morning", "at sunset"]

places = ["summit", "trail", "basecamp", "ridge", "peak", "glacier", "plateau"]
adjectives = ["snow-capped", "towering", "majestic", "steep", "rocky", "treacherous", "serene", "breathtaking"]

def generate_sentence(mountain):
    name_parts = mountain.split()
    tagged_mountain = [f"{name_parts[0]} B-MOUNTAIN\n"] + \
                      [f"{part} I-MOUNTAIN\n" for part in name_parts[1:]]
    sentence = [
        "We O\n", 
        random.choice(verbs) + " O\n",
        "the O\n",
        random.choice(adjectives) + " O\n",
        random.choice(places) + " O\n",
        "of O\n"
    ] + tagged_mountain + [". O"]
    return sentence

def generate_dataset(size=1000):
    dataset = []
    for _ in range(size):
        mountain = random.choice(mountains)
        sentence = generate_sentence(mountain)
        dataset.append("".join(sentence))
    return "\n\n".join(dataset)

large_dataset = generate_dataset(1000)  # Adjust size as needed

with open("data/generated.csv", "w") as f:
    f.write(large_dataset)

print("Dataset generated and saved.")
