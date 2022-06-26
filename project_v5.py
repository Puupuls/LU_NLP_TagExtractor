from __future__ import annotations

import nltk
import torch
from loguru import logger
from collections import Counter
from string import punctuation

from rake_nltk import Rake as NLTKRake
from rake_spacy import Rake as SpacyRake
import flask
import spacy
from flask import render_template, request
from spacy import Language
from torch import cosine_similarity
from torchtext.vocab import GloVe

app = flask.Flask(__name__)
nltk.download('stopwords')
nltk.download('punkt')
glove: None | GloVe = None
nlp: None | Language = None

DEVICE = "cpu"
if torch.cuda.is_available():
    spacy.prefer_gpu()
    DEVICE = "cuda"


def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result


# Function to get the closest words by meaning
def get_similar(word, limit=10, similarity_threshold=0.75):
    if word not in glove.stoi:
        return []
    vec = glove.get_vecs_by_tokens([word], True)
    distances = cosine_similarity(vec, glove.vectors)
    shortest_distances = distances.flatten().argsort()[-limit:]
    filtered_shortest_distances = shortest_distances[distances[shortest_distances] > similarity_threshold]
    return [(glove.itos[i], distances[i]) for i in filtered_shortest_distances][::-1]


def text2camel(text):
    return text.title().replace(" ", "")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get-tags', methods=['POST'])
def get_tags():
    global glove, nlp
    if not glove:
        logger.info('Loading glove')
        glove = GloVe(name='840B', dim=300)
        glove.vectors.to(DEVICE)
    if not nlp:
        logger.info('Loading spacy')
        nlp = spacy.load("en_core_web_lg")

    text = request.form['text']
    result = {
        "text": text
    }
    text = text.lower()

    logger.info("Processing SpaCy + GloVe")
    spacy_hotwords = {}
    spacy_hotwords_glove = {}
    try:
        most_common_list = Counter(get_hotwords(text)).most_common(15)
        spacy_hotwords = {text2camel(tag[0]): float(tag[1]) for tag in most_common_list}
        for tag in spacy_hotwords:
            try:
                for similar_word, sim_conf in get_similar(tag):
                    if text2camel(similar_word) not in spacy_hotwords:
                        spacy_hotwords_glove[text2camel(similar_word)] = max(float(sim_conf), spacy_hotwords_glove.setdefault(text2camel(similar_word), 0))
            except Exception as e:
                logger.exception(e)
    except Exception as e:
        logger.exception(e)
    result['spacy_hotwords'] = spacy_hotwords
    result['spacy_hotwords_glove'] = spacy_hotwords_glove

    logger.info("Processing spacy_rake + GloVe")
    rake_spacy_hotwords = {}
    rake_spacy_hotwords_glove = {}
    try:
        r = SpacyRake(
            nlp=nlp,
            min_length=1,
            max_length=2,
        )
        extracted = [(float(v), str(t)) for v, t in r.apply(text)]
        rake_spacy_hotwords = {text2camel(str(t)): v for v, t in extracted}
        for _, tag in extracted:
            try:
                for similar_word, sim_conf in get_similar(tag):
                    if text2camel(similar_word) not in rake_spacy_hotwords:
                        rake_spacy_hotwords_glove[text2camel(similar_word)] = max(float(sim_conf),
                                                                                  rake_spacy_hotwords_glove.setdefault(
                                                                                      text2camel(similar_word), 0))
            except Exception as e:
                logger.exception(e)
    except Exception as e:
        logger.exception(e)
    result['rake_spacy_hotwords'] = rake_spacy_hotwords
    result['rake_spacy_hotwords_glove'] = rake_spacy_hotwords_glove

    logger.info("Processing rake_nltk + GloVe")
    rake_nltk_hotwords = {}
    rake_nltk_hotwords_glove = {}
    try:
        r = NLTKRake(
            max_length=2,
            include_repeated_phrases=False,
        )
        r.extract_keywords_from_text(text)
        ranklist = r.get_ranked_phrases_with_scores()
        rake_nltk_hotwords = {text2camel(tag[1]): float(tag[0]) for tag in ranklist}
        for _, tag in ranklist:
            try:
                for similar_word, sim_conf in get_similar(tag):
                    if text2camel(similar_word) not in rake_spacy_hotwords:
                        rake_nltk_hotwords_glove[text2camel(similar_word)] = max(float(sim_conf),
                                                                                 rake_nltk_hotwords_glove.setdefault(
                                                                                     text2camel(similar_word), 0))
            except Exception as e:
                logger.exception(e)
    except Exception as e:
        logger.exception(e)
    result['rake_nltk_hotwords'] = rake_nltk_hotwords
    result['rake_nltk_hotwords_glove'] = rake_nltk_hotwords_glove

    tags = list(spacy_hotwords.keys()) + list(rake_spacy_hotwords.keys()) + list(rake_nltk_hotwords.keys()) + \
           list(spacy_hotwords_glove.keys()) + list(rake_spacy_hotwords_glove.keys()) + \
           list(rake_nltk_hotwords_glove.keys())

    counts = Counter(tags)
    tags = [(tag, count) for tag, count in counts.most_common() if count > 1]

    result['tags'] = tags

    logger.info(result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
