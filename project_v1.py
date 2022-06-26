import json
import time

from torch import cosine_similarity
from torchtext.vocab import GloVe
import spacy
from rake_spacy import Rake

print("Starting")
time_start = time.time()

global_vectors = GloVe(name='840B', dim=300)
# spacy.prefer_gpu()

# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

r = Rake(
    nlp=nlp,
    min_length=1,
    max_length=2,
)
print(f"Models loaded in {time.time() - time_start} seconds")


# Function to get the closest words by meaning
def get_similar(word, limit=10):
    if word not in global_vectors.stoi:
        return []
    vec = global_vectors.get_vecs_by_tokens([word], True)
    distances = cosine_similarity(vec, global_vectors.vectors)
    shortest_distances = distances.flatten().argsort()[-limit:]
    filtered_shortest_distances = shortest_distances[distances[shortest_distances] > 0.75]
    return [(global_vectors.itos[i], distances[i]) for i in filtered_shortest_distances][::-1]


while True:
    tags = {}
    text = input("Enter your text: ")
    text = text.lower()
    ranklist = r.apply(text)
    for tag in ranklist:
        t = str(tag[1])
        tags[t.lower()] = max(float(tag[0]), tags.setdefault(t.lower(), 0))
        try:
            similar = get_similar(t)
            for similar_word, sim_conf in similar:
                tags[similar_word.lower()] = max(
                    float(sim_conf),
                    tags.setdefault(similar_word.lower(), 0)
                )
        except Exception as e:
            print(e)

    tags = {key.replace(" ", "_"): value for key, value in sorted(tags.items(), key=lambda x: x[1], reverse=True)}
    print(json.dumps(tags, indent=4))
