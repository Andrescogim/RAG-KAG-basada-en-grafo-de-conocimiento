from src.utils.conexion_Neo4j import ConexionNeo4j
import spacy

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from config.parametros_construccion_grafo import *


def build_construccion_grafo_config():

    embed_model_st = SentenceTransformer(EMBED_MODEL)
    con_Neo4j = ConexionNeo4j(DATABASE)

    if ORIGEN_TRIPLETAS == 'rebel':
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(REBEL_MODEL).to("cuda")
        nlp = spacy.load(SPACY_MODEL)
    else:
        tokenizer = None
        model = None
        nlp = None

    parametros = {
        "con_Neo4j": con_Neo4j,
        "origen_tripletas": ORIGEN_TRIPLETAS,
        "n_registros": N_REGISTROS,
        "database": DATABASE,
        "reemplazar_database": REEMPLAZAR_DATABASE,
        "tokenizer": tokenizer,
        "model": model,
        "nlp": nlp,
        "n_window": N_WINDOW,
        "embed_model_st": embed_model_st,
        "vector_index_name": VECTOR_INDEX_NAME,
        "vector_index_dim": VECTOR_INDEX_DIM,
        "similarity_func_index": SIMILARITY_FUNC_INDEX,
        "text_index_name": TEXT_INDEX_NAME,
    }

    return parametros
