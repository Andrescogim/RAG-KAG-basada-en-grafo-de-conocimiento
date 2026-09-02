from pathlib import Path
import sys

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent
sys.path.append(str(root_dir))

from datetime import datetime as dt
import json
import pandas as pd
from src.utils.metricas.metricas_2Wiki import metricas_totales
from config.parametros_graph_retrieval import (
    DATABASE_NEO,
    PARAMETROS_GRAFO,
    RERANKER_MODEL,
    EMBED_MODEL,
    NER_MODEL,
    RUTA_GUARDADO_RESULTADOS,
    RUTA_GUARDADO_REGISTRO,
    RUTA_GUARDADO_RECURSOS,
    PROMPT_BASE,
    COMENTARIOS
)



def guardar_como_json(resultados, nombre_archivo, ruta):
    
    ahora = dt.now()
    fecha_hora = ahora.strftime("%Y-%m-%d_%H-%M")
    out_name = f"{nombre_archivo}_{fecha_hora}.json"
    out_file = ruta / out_name
    resultados_json = json.dumps(resultados, indent=4, ensure_ascii=False)
    with open(out_file, "w", encoding="utf-8") as archivo:
        archivo.write(resultados_json)
    return resultados_json


def guardar_registro(ruta, comentarios, metricas_totales, modelo_LLM, prompt, parametros_retrieval):
    """
        Guarda registro de ejecucion en un txt.
    """
    
    ahora = dt.now()
    fecha_hora = ahora.strftime("%Y-%m-%d_%H-%M")
    out_name = f"Registro_{fecha_hora}.txt"
    out_file = ruta / out_name

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"fecha_ejecucion: {fecha_hora}\n\n")
        f.write(f"Comentarios: {comentarios}\n\n")
        f.write(f"Metricas: {json.dumps(metricas_totales, indent = 2, ensure_ascii = False)}\n\n")
        f.write(f"modelo_utilizado: {modelo_LLM}\n\n")
        f.write(f"parametros generales: {parametros_retrieval}\n\n")
        f.write(f'prompt_utilizado: \n"{prompt}"')
    return 1


def guardar_df_as_csv(df, nombre_archivo, ruta):
    
    ahora = dt.now()
    fecha_hora = ahora.strftime("%Y-%m-%d_%H-%M")
    out_name = f"{nombre_archivo}_{fecha_hora}.csv"
    out_file = ruta / out_name
    df.to_csv(out_file, index=False)



def guardar_resultados_grafo(resultados, recursos_por_iteracion, recursos_general):

    dic_param_reg = PARAMETROS_GRAFO.copy()
    
    dic_param_reg["reranker"] = RERANKER_MODEL
    dic_param_reg["embed_model"] = EMBED_MODEL
    dic_param_reg["ner_model"] = NER_MODEL

    rango_in = dic_param_reg["rango_in_data"]
    rango_fin = dic_param_reg["rango_fin_data"]
    llm = dic_param_reg["llm_name"].replace(":", "-")
    nombre_result = f"graph_answer_DB_{DATABASE_NEO}_{llm}_NREG_{rango_in}-{rango_fin}"
    
    metricas_agg = metricas_totales(resultados)    

    ruta_resultados = root_dir / RUTA_GUARDADO_RESULTADOS
    ruta_registro = root_dir / RUTA_GUARDADO_REGISTRO
    ruta_recursos = root_dir / RUTA_GUARDADO_RECURSOS
    
    _ = guardar_como_json(resultados, nombre_result, ruta_resultados)

    guardar_registro(
        ruta_registro,
        COMENTARIOS,
        # parametros["n_registros"],
        metricas_agg,
        dic_param_reg["llm_name"],
        dic_param_reg,
        PROMPT_BASE,
        )

    _ = guardar_como_json(recursos_general, nombre_result, ruta_recursos)
    
    df_recursos = pd.DataFrame(recursos_por_iteracion)
    archivo_recursos = RUTA_GUARDADO_RECURSOS / f"{nombre_result}.csv"
    df_recursos.to_csv(archivo_recursos,index=False)
    
    return metricas_agg

