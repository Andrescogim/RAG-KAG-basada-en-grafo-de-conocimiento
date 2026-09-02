# ORIGEN_TRIPLETAS = "evidences"
ORIGEN_TRIPLETAS = "rebel"

N_REGISTROS = 5

DATABASE = "PRUEBA.AUTO"
# DATABASE = "2wiki.rebel.500"

REEMPLAZAR_DATABASE = True

TOKENIZER_MODEL = "Babelscape/rebel-large"
REBEL_MODEL = "Babelscape/rebel-large"

N_WINDOW = 4

SPACY_MODEL = "en_core_web_sm"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

VECTOR_INDEX_NAME = "entity_embedding_index"
VECTOR_INDEX_DIM = 384
SIMILARITY_FUNC_INDEX = "cosine"

TEXT_INDEX_NAME = "entidadesIndex"