from __future__ import annotations
import nltk
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
from torchtext.vocab import GloVe, FastText
import gensim.downloader

app = flask.Flask(__name__)
nltk.download('stopwords')
nltk.download('punkt')
glove: None | GloVe = None
fast_text: None | Language = None
nlp: None | Language = None
word2vec: None = None


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


def get_similar(model, word, limit=10, similarity_threshold=0.75):
    """
    Get the closest words by meaning using GloVe embeddings.
    """
    if word not in glove.stoi:
        return []
    # Get word embedding
    vec = glove.get_vecs_by_tokens([word], True)
    # Get all cosine similarities
    distances = cosine_similarity(vec, glove.vectors)
    # Sort by similarity
    shortest_distances = distances.flatten().argsort()[-limit:]
    # Filter by similarity threshold
    filtered_shortest_distances = shortest_distances[distances[shortest_distances] > similarity_threshold]
    # Return the closest words and their similarity
    return [(glove.itos[i], distances[i]) for i in filtered_shortest_distances][::-1]


def text2camel(text):
    """
    Convert text to camel case.
    """
    return text.title().replace(" ", "").replace("_", "")


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
    global glove, nlp, fast_text, word2vec
    if not glove:
        logger.info('Loading GloVe')
        glove = GloVe(name='840B', dim=300)
    if not fast_text:
        logger.info('Loading FastText')
        fast_text = FastText('en')
    if not nlp:
        logger.info('Loading spacy')
        nlp = spacy.load("en_core_web_lg", exclude=['parser', 'ner', "lemmatizer"])
    if not word2vec:
        logger.info('Loading Word2Vec')
        word2vec = gensim.downloader.load('word2vec-google-news-300')

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
            for similar_word, sim_conf in get_similar(glove, tag):
                # If the similar word is not in the list of tags, add it to the list of tags
                if similar_word.lower() not in tags_lower:
                    glove_tags[text2camel(similar_word)] = max(float(sim_conf),
                                                               glove_tags.setdefault(
                                                                   text2camel(similar_word), 0))
        except Exception as e:
            logger.exception(e)
    result['glove'] = glove_tags

    # Get similar words based on meaning using FastText
    logger.info("Processing FastText")
    fast_text_tags = {}
    for tag in tags:
        try:
            for similar_word, sim_conf in get_similar(fast_text, tag):
                # If the similar word is not in the list of tags, add it to the list of tags
                if similar_word.lower() not in tags_lower:
                    fast_text_tags[text2camel(similar_word)] = max(float(sim_conf),
                                                                   fast_text_tags.setdefault(
                                                                       text2camel(similar_word), 0))
        except Exception as e:
            logger.exception(e)
    result['fasttext'] = fast_text_tags

    # Get similar words based on meaning using Word2Vec
    logger.info("Processing Word2Vec")
    word2vec_tags = {}
    for tag in tags:
        try:
            for similar_word, sim_conf in word2vec.most_similar(tag.lower(), topn=10):
                # If the similar word is not in the list of tags, add it to the list of tags
                if similar_word.lower() not in tags_lower and sim_conf > 0.75:
                    word2vec_tags[text2camel(similar_word)] = max(float(sim_conf),
                                                                  word2vec_tags.setdefault(
                                                                      text2camel(similar_word), 0))
        except KeyError:
            logger.warning(f"Word2Vec does not have the word: {tag}")
        except Exception as e:
            logger.exception(e)
    result['word2vec'] = word2vec_tags

    # Merge previously found tags with the tags found using GloVe
    result['tags'].update(fast_text_tags)

    logger.info(result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
