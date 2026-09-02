from src.utils.funciones_carga_datos import load_filter_dataset_HuggingFace
from src.utils.funciones_construir_grafo import(
    extraccion_tripletas_2wiki_rebel,
    limpiar_tripletas,
)
from src.utils.funciones_guardado import guardar_como_json


def extraer_tripletas_2Wiki(dataset_2Wiki, tokenizer, model, nlp, n_window_nac):

    tripletas_rebel, triplestas_nacionalidad = extraccion_tripletas_2wiki_rebel(dataset_2Wiki, tokenizer, model, nlp, n_window_nac)
    tripletas_limpias = list(set(limpiar_tripletas(tripletas_rebel)))
    tripletas_limpias_nac = list(set(limpiar_tripletas(triplestas_nacionalidad)))
    return tripletas_limpias, tripletas_limpias_nac


def guardar_tripletas(tripletas_limpias, tripletas_limpias_nac, registro_in, registro_fin, ruta):
    
    nombre_result = f"tripletas_Rebel_2Wiki_registros_{registro_in}-{registro_fin}"
    nombre_result_nac = f"tripletas_Nacionalidad_2Wiki_registros_{registro_in}-{registro_fin}"
    _ = guardar_como_json(tripletas_limpias, nombre_result, ruta)
    _ = guardar_como_json(tripletas_limpias_nac, nombre_result_nac, ruta)
    
    
def extraccion_y_guardado(registro_in, registro_fin, split, tokenizer, model, nlp, n_window, ruta_guardar_tripletas):
    
    subset = f"{split}[{registro_in}:{registro_fin}]"
    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_subset = None, split = subset)

    tripletas_limpias, tripletas_limpias_nac = extraer_tripletas_2Wiki(dataset_2Wiki, tokenizer, model, nlp, n_window)

    guardar_tripletas(tripletas_limpias, tripletas_limpias_nac, registro_in, registro_fin, ruta_guardar_tripletas)

