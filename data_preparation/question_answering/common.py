import os
import sentencepiece as spm
import numpy as np
from elasticsearch import helpers
import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

sentencepiece_model = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../plt5_tokenizer.model")
)
t5_tokenizer = spm.SentencePieceProcessor(model_file=sentencepiece_model)  # type: ignore


def prepare_es_index(es, index_name, index_config, generator_fn):
    logger.info("Creating target index")
    es.indices.create(index=index_name, body=index_config)

    logger.info("Indexing wikipedia")
    paragraphs = generator_fn(index_name)
    success, errors = helpers.bulk(es, paragraphs, stats_only=True)

    logger.info("Indexed %d wikipedia passages", success)
    if errors:
        logger.warning("Failed to index %d passages", errors)


def token_length(text):
    return len(t5_tokenizer.EncodeAsIds(text))


def get_token_offests(text):
    return np.cumsum(list(map(len, t5_tokenizer.EncodeAsPieces(" " + text))))


def make_inputs(question, search_results, limit=510):
    context = question + " context"
    title, paragraph = "", ""
    limit -= token_length(context)

    for h in search_results["hits"]["hits"]:
        source = h["_source"]

        if title != source["title"] and limit - token_length(source["title"]) > 0:
            title = source["title"]
            context += " " + title
            limit -= token_length(title)
        if paragraph == source["paragraph"]:
            continue

        paragraph = source["paragraph"]
        limit -= token_length(paragraph)
        if limit > 0:
            context += " " + paragraph
        else:
            context += " " + paragraph[: get_token_offests(paragraph)[limit]]
            return context
    return context
