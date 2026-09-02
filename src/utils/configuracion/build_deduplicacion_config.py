from pathlib import Path
import sys

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent.parent
sys.path.append(str(root_dir))

from src.utils.conexion_Neo4j import ConexionNeo4j
import spacy
from config.parametros_deduplicacion import (
    DATABASE_NEO,
    N_CANDIDATOS,
    RUTA_GUARDADO_CANDIDATOS,
    RUTA_GUARDADO_FUSIONADOS,
    SPACY_MODEL
)

def build_config_deduplicacion():

    nlp = spacy.load(SPACY_MODEL)

    con_Neo4j = ConexionNeo4j(DATABASE_NEO)

    parametros = {
        "con_Neo4j": con_Neo4j,
        "database_Neo": DATABASE_NEO,
        "nlp": nlp,
        "n_candidatos": N_CANDIDATOS,
        "ruta_candidatos":root_dir / RUTA_GUARDADO_CANDIDATOS,
        "ruta_fusionados": root_dir / RUTA_GUARDADO_FUSIONADOS
    }

    return parametros