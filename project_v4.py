import json
import time
from collections import Counter

import spacy
from string import punctuation
from torch import cosine_similarity
from torchtext.vocab import GloVe


print("Starting")
time_start = time.time()

global_vectors = GloVe(name='840B', dim=300)
nlp = spacy.load("en_core_web_lg")

print(f"Models loaded in {time.time() - time_start} seconds")


def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

# Function to get the closest words by meaning
def get_similar(word, limit=10, similarity_threshold=0.75):
    if word not in global_vectors.stoi:
        return []
    vec = global_vectors.get_vecs_by_tokens([word], True)
    distances = cosine_similarity(vec, global_vectors.vectors)
    shortest_distances = distances.flatten().argsort()[-limit:]
    filtered_shortest_distances = shortest_distances[distances[shortest_distances] > similarity_threshold]
    return [(global_vectors.itos[i], distances[i]) for i in filtered_shortest_distances][::-1]


while True:
    tags = {}
    text = input("Enter your text: ")
    text = text.lower()
    output = get_hotwords(text)
    most_common_list = Counter(output).most_common(10)
    for tag in most_common_list:
        t = str(tag[0])
        tags[t.lower()] = max(float(tag[1]), tags.setdefault(t.lower(), 0))
        for phrase in t.split(" "):
            try:
                similar = get_similar(t)
                for similar_word, sim_conf in similar:
                    tags[similar_word.lower()] = max(
                        float(sim_conf),
                        tags.setdefault(similar_word.lower(), 0)
                    )
            except Exception as e:
                print(e)

    tags = {key: value for key, value in sorted(tags.items(), key=lambda x: x[1], reverse=True)}
    print(json.dumps(tags, indent=4, ensure_ascii=False))
