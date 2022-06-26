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


def get_hotwords(text: str, limit: int = 10):
    """
    Get most common adjectives and Nouns in text and mark them as tags.
    """
    result = []
    pos_tag = ['ADJ', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)
    return Counter(result).most_common(limit)


def get_similar(word, limit=10, similarity_threshold=0.75):
    """
    Get the closest words by meaning using GloVe embeddings.
    """
    if word not in glove.stoi:
        return []
    vec = glove.get_vecs_by_tokens([word], True)
    distances = cosine_similarity(vec, glove.vectors)
    shortest_distances = distances.flatten().argsort()[-limit:]
    filtered_shortest_distances = shortest_distances[distances[shortest_distances] > similarity_threshold]
    return [(glove.itos[i], distances[i]) for i in filtered_shortest_distances][::-1]


def text2camel(text):
    """
    Convert text to camel case.
    """
    return text.title().replace(" ", "")


@app.route('/')
def index():
    """
    Show the web page.
    """
    return render_template('index.html')


@app.route('/get-tags', methods=['POST'])
def get_tags():
    """
    Get the tags from the text using 4 different methods/models.
    """
    # Do not load models on startup, instead load them on demand and cache them to speed up subsequent requests
    global glove, nlp
    if not glove:
        logger.info('Loading glove')
        glove = GloVe(name='840B', dim=300)
    if not nlp:
        logger.info('Loading spacy')
        nlp = spacy.load("en_core_web_lg", exclude=['parser', 'ner', "lemmatizer"])

    # Get the text from the web page and create response object
    text = request.form['text']
    result = {
        "text": text
    }

    # Merge words with "-" in them as they frequently cause bad tags
    # Use title case as it is most common way of creating multi-word hashtags
    text = text.title().replace("-", "")

    # Get spacy tags that are based on Part of Speech (POS) tagging
    logger.info("Processing SpaCy")
    spacy_hotwords = {}
    try:
        most_common_list = get_hotwords(text)
        spacy_hotwords = {text2camel(tag[0]): float(tag[1]) for tag in most_common_list}
    except Exception as e:
        logger.exception(e)
    result['spacy_hotwords'] = spacy_hotwords

    # Get tags using spacy Rake
    logger.info("Processing spacy_rake")
    rake_spacy_tags = {}
    try:
        r = SpacyRake(
            nlp=nlp,
            min_length=1,
            max_length=3,
        )
        extracted = [(float(v), str(t)) for v, t in r.apply(text)]
        rake_spacy_tags = {text2camel(str(t)): v for v, t in extracted}
    except Exception as e:
        logger.exception(e)
    result['rake_spacy_tags'] = rake_spacy_tags

    # Get tags using NLTK Rake these usually are almost the same as above ones as they use the same algorithm, but
    # sometimes they are different
    logger.info("Processing rake_nltk")
    rake_nltk_tags = {}
    try:
        r = NLTKRake(
            min_length=1,
            max_length=3,
            include_repeated_phrases=False,
        )
        r.extract_keywords_from_text(text)
        ranklist = r.get_ranked_phrases_with_scores()
        rake_nltk_tags = {text2camel(tag[1]): float(tag[0]) for tag in ranklist}
    except Exception as e:
        logger.exception(e)
    result['rake_nltk_tags'] = rake_nltk_tags

    # Merge tags from above 3 methods
    tags = list(spacy_hotwords.keys()) + list(rake_spacy_tags.keys()) + list(rake_nltk_tags.keys())

    # Get tags that are mentioned more than once
    counts = Counter(tags)
    tags = {tag: count for tag, count in counts.most_common() if count > 1}
    result['tags'] = tags

    # Get similar words based on meaning using GloVe
    logger.info("Processing GloVe")
    glove_tags = {}
    tags_lower = [tag.lower() for tag in tags]
    for tag in tags:
        try:
            for similar_word, sim_conf in get_similar(tag):
                # If the similar word is not in the list of tags, add it to the list of tags
                if similar_word.lower() not in tags_lower:
                    glove_tags[text2camel(similar_word)] = max(float(sim_conf),
                                                               glove_tags.setdefault(
                                                                   text2camel(similar_word), 0))
        except Exception as e:
            logger.exception(e)
    result['glove'] = glove_tags

    # Merge previously found tags with the tags found using GloVe
    result['tags'].update(glove_tags)

    logger.info(result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
