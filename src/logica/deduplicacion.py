import sys
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.funciones_deduplicacion import (
    calcular_rank_B25s,
    encontrar_candidatos_iguales_principal,
    get_columnas,
    calcular_distancias,
    excluir_por_numeros,
    excluir_por_nombre,
    filtrar_fusionables,
    unir_candidatos,
    grupos_fusion,
    seleccionar_nodo_principal
)
from src.utils.funciones_guardado import guardar_como_json, guardar_df_as_csv


def obtener_candidatos(con_Neo4j, nlp, n_candidatos):

    print("Extrayendo Entidades...")
    entidades = con_Neo4j.extraer_all_entidades_neo4j()
    # n_candidatos = 3
    print("Calculando scores y distancias...")
    df_deduplicacion = calcular_rank_B25s(entidades, n_candidatos)
    df_deduplicacion = encontrar_candidatos_iguales_principal(df_deduplicacion, n_candidatos)
    cols_distancias, cols_exc_num, cols_exc_nom = get_columnas(n_candidatos)
    df_deduplicacion[cols_distancias] = df_deduplicacion.apply(calcular_distancias, args=(nlp, n_candidatos), axis = 1, result_type='expand')
    df_deduplicacion[cols_exc_num] = df_deduplicacion.apply(excluir_por_numeros, args=(n_candidatos,), axis = 1, result_type='expand')
    df_deduplicacion[cols_exc_nom] = df_deduplicacion.apply(excluir_por_nombre, args=(nlp, n_candidatos), axis = 1, result_type='expand')
    print("Filtro y preparacion de deduplicaciones...")
    df_deduplicacion = filtrar_fusionables(df_deduplicacion, n_candidatos)
    df_deduplicacion_final = unir_candidatos(df_deduplicacion, n_candidatos)
    return df_deduplicacion_final


def agrupar_y_fusionar(con_Neo4j, df_deduplicacion_final):
    grupos = grupos_fusion(df_deduplicacion_final)
    print("Obteniendo grados de los nodos...")
    grados = con_Neo4j.obtener_grados_nodos(df_deduplicacion_final)
    nodos_fusionados = seleccionar_nodo_principal(grupos, grados)
    print("Fusionando nodos en Neo4j...")
    N_nodos_fusionados = con_Neo4j.fusionar_nodos(nodos_fusionados)
    return nodos_fusionados, N_nodos_fusionados
    

def guardar_resultados_fusion(df_deduplicacion_final, ruta_candidatos, nodos_fusionados, ruta_fusionados, database_Neo):
    nombre_guardado_df = f"Deduplicaciones_2Wiki_DB_{database_Neo}"
    _ = guardar_df_as_csv(df_deduplicacion_final, nombre_guardado_df, ruta_candidatos)
    
    nombre_guardado = f"Deduplicaciones_2Wiki_DB_{database_Neo}"
    _ = guardar_como_json(nodos_fusionados, nombre_guardado, ruta_fusionados)


    
def deduplicar_nodos(    
    con_Neo4j,
    database_Neo,
    nlp,
    n_candidatos,
    ruta_candidatos,
    ruta_fusionados):
    
    df_deduplicacion_final = obtener_candidatos(con_Neo4j, nlp, n_candidatos)
    nodos_fusionados, N_nodos_fusionados = agrupar_y_fusionar(con_Neo4j, df_deduplicacion_final)
    guardar_resultados_fusion(df_deduplicacion_final, ruta_candidatos, nodos_fusionados, ruta_fusionados, database_Neo)

    print(f"Nº de nodos fusionados: {df_deduplicacion_final.shape[0]}")
    print(f"Nº de nodos tras fusion: {N_nodos_fusionados}")