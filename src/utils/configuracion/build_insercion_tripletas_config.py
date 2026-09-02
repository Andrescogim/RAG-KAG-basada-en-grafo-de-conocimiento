from pathlib import Path
import sys

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent.parent
sys.path.append(str(root_dir))


from src.utils.conexion_Neo4j import ConexionNeo4j
from sentence_transformers import SentenceTransformer
from config.parametros_insercion_tripletas import (
    DATABASE_NEO,
    EMBED_MODEL_ST,
    ARCHIVOS,
    RUTA_TRIPLETAS,
)

def build_insercion_tripletas_config():
    con_Neo4j = ConexionNeo4j(DATABASE_NEO)

    embed_model_st = SentenceTransformer(EMBED_MODEL_ST)

    parametros = {
        "con_Neo4j": con_Neo4j,
        "archivos": ARCHIVOS,
        "ruta_tripletas": root_dir / RUTA_TRIPLETAS,
        "embed_model_st": embed_model_st,
    }

    return parametros