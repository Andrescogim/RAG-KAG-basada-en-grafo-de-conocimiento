from pathlib import Path
import sys

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent.parent
sys.path.append(str(root_dir))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
from config.parametros_extraccion_guardado_tripletas import (
    REGISTRO_INICIO_EXTRACCION,
    REGISTRO_FIN_EXTRACCION,
    SPLIT,
    TOKENIZER,
    MODEL,
    NLP,
    N_WINDOW,
    RUTA_GUARDAR_TRIPLETAS
)

def build_extraccion_tripletas_config():
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to("cuda")
    nlp = spacy.load(NLP)
    
    parametros = {
        "registro_in": REGISTRO_INICIO_EXTRACCION,
        "registro_fin": REGISTRO_FIN_EXTRACCION,
        "split": SPLIT,
        "tokenizer": tokenizer,
        "model": model,
        "nlp": nlp,
        "n_window": N_WINDOW,
        "ruta_guardar_tripletas": root_dir / RUTA_GUARDAR_TRIPLETAS
    }

    return parametros