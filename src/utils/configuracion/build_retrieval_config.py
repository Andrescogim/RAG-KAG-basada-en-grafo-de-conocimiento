from src.utils.conexion_Neo4j import ConexionNeo4j
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
from config.parametros_graph_retrieval import (
    DATABASE_NEO,
    PARAMETROS_GRAFO,
    RERANKER_MODEL,
    EMBED_MODEL,
    NER_MODEL,
    PROMPT_BASE
)

def build_retrieval_config():
    con_neo4j = ConexionNeo4j(DATABASE_NEO)

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
    )

    embed_model_st = SentenceTransformer(EMBED_MODEL)

    ner = spacy.load(NER_MODEL)

    parametros = PARAMETROS_GRAFO.copy()

    parametros.update({
        "con_Neo4j": con_neo4j,
        "reranker": reranker,
        "embed_model_st": embed_model_st,
        "ner_model": ner,
        "prompt_base": PROMPT_BASE,
    })

    return parametros