from src.utils.funciones_construir_grafo import(
    obtencion_entidades_de_tripletas,
)
from src.utils.funciones_generales import(
    lectura_json,
)

def insercion_tripletas(con_Neo4j, archivos, ruta_tripletas, embed_model_st):
    tripletas_all = []
    for archivo in archivos:
        tripletas = lectura_json(ruta_tripletas, archivo)
        tripletas_all += tripletas

    print("INSERTANDO TRIPLETAS EN Neo4j...")
    summary = con_Neo4j.insertar_triplets_batch(tripletas_all)

    print("CREACION DE EMBEDDINGS Neo4j...")
    entidades = obtencion_entidades_de_tripletas(tripletas_all)
    sumary = con_Neo4j.añadir_embeddings_como_propiedad_neo4j(entidades, embed_model_st)
    
    print("INSERCION DE TRIPLETAS FINALIZADA")
