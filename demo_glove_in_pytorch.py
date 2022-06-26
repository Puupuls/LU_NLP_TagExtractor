from torch import cosine_similarity, dist
from torchtext.vocab import GloVe

global_vectors = GloVe(name='840B', dim=300)
print(global_vectors.vectors.shape)
# print(global_vectors.get_vecs_by_tokens(["cat"], True))


def get_vector(word):
    return global_vectors.get_vecs_by_tokens([word], True)


# Function to get the closest words by meaning (allows passing a tensor for playing with word arithmetics)
def get_similar(word, limit=10):
    if isinstance(word, str):
        vec = get_vector(word)
    else:
        vec = word
    distances = cosine_similarity(vec, global_vectors.vectors)
    shortest_distances = distances.flatten().argsort()[-limit:]
    return [(global_vectors.itos[i], distances[i]) for i in shortest_distances][::-1]


print(get_similar("cat"))
print(get_similar("dog"))
print(get_similar("king"))
print(get_similar(get_vector("king") + get_vector("queen")))
print(get_similar(get_vector("king") - get_vector("queen")))
print(get_similar(get_vector("king") - get_vector("queen")))
print(get_similar("test"))
print(get_similar("text"))
