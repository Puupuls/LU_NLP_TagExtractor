import json
import time

from torch import cosine_similarity
from torchtext.vocab import GloVe
from rake_nltk import Rake
import nltk


print("Starting")
nltk.download('stopwords')
nltk.download('punkt')

time_start = time.time()

global_vectors = GloVe(name='840B', dim=300)
r = Rake(
    max_length=2,
    include_repeated_phrases=False,
)

print(f"Models loaded in {time.time() - time_start} seconds")


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
    r.extract_keywords_from_text(text)
    ranklist = r.get_ranked_phrases_with_scores()
    for tag in ranklist:
        t = str(tag[1])
        tags[t.lower()] = max(float(tag[0]), tags.setdefault(t.lower(), 0))
        for phrase in t.split(" "):
            try:
                similar = get_similar(
                    t,
                    similarity_threshold=0.75 if " " not in t else 0.8,
                )
                for similar_word, sim_conf in similar:
                    tags[similar_word.lower()] = max(
                        float(sim_conf),
                        tags.setdefault(similar_word.lower(), 0)
                    )
            except Exception as e:
                print(e)

    tags = {key.replace(" ", "_"): value for key, value in sorted(tags.items(), key=lambda x: x[1], reverse=True)}
    print(json.dumps(tags, indent=4, ensure_ascii=False))
