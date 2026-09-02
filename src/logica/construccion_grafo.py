import sys
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.funciones_carga_datos import load_filter_dataset_HuggingFace
from src.utils.funciones_construir_grafo import(
    limpiar_tripletas,
    extraccion_tripletas_2wiki_rebel,
    tripletas_from_evidences_2Wiki
)


def tripletas_from_evidencias(dataset_2Wiki):
    tripletas = tripletas_from_evidences_2Wiki(dataset_2Wiki)
    tripletas_limpias = list(set(limpiar_tripletas(tripletas)))
    return tripletas_limpias


def tripletas_from_rebel(dataset_2Wiki, tokenizer, model, nlp, n_window, database_Neo, registros_subset):
    
    tripletas, tripletas_nacionalidad = extraccion_tripletas_2wiki_rebel(dataset_2Wiki, tokenizer, model, nlp, n_window)
    tripletas_limpias = limpiar_tripletas(tripletas)
    tripletas_limpias_nac = limpiar_tripletas(tripletas_nacionalidad)
    tripletas_all = list(set(tripletas_limpias + tripletas_limpias_nac))
    return tripletas_all

    # ----------------- GUARDADO TRIPLETAS EN LOCAL -----------------
    ruta_tripletas = root_dir / "outputs" / "tripletas_generadas" / "dataset_2Wiki"

    nombre_result = f"tripletas_Rebel_2Wiki_DB_{database_Neo}_registros_{registros_subset}"
    nombre_result_nac = f"tripletas_Nacionalidad_2Wiki_DB_{database_Neo}_registros_{registros_subset}"
    _ = guardar_resultados(tripletas_limpias, nombre_result, ruta_tripletas)
    _ = guardar_resultados(tripletas_limpias_nac, nombre_result_nac, ruta_tripletas)


def construir_grafo(
    con_Neo4j,
    origen_tripletas,
    n_registros,
    database,
    reemplazar_database,
    tokenizer,
    model,
    nlp,
    embed_model_st,
    n_window,
    vector_index_name,
    vector_index_dim,
    similarity_func_index,
    text_index_name,
):

    registros_subset = f"0-{n_registros}"

    dataset = load_filter_dataset_HuggingFace(
        "xanhho/2wikimultihopqa",
        n_registros,
        "train",
    )
    print("OBTENCION DATASET OK")

    if reemplazar_database:
        con_Neo4j.crear_reemplazar_database()
    else:
        con_Neo4j.crear_database()

    print("CREACION DATABASE EN Neo4j OK")

    print("COMENZANDO EXTRACCION Y CARGA DE TRIPLETAS...")

    if origen_tripletas == "evidences":
        tripletas = tripletas_from_evidencias(dataset)

    elif origen_tripletas == "rebel":
        tripletas = tripletas_from_rebel(
            dataset,
            tokenizer,
            model,
            nlp,
            n_window,
            database,
            registros_subset,
        )

    _ = con_Neo4j.insertar_triplets_batch(tripletas)
    print("EXTRACCION Y CARGA DE TRIPLETAS OK")

    print("COMENZANDO CREACION DE EMBEDDINGS Y TEXT INDEX EN Neo4j")

    entidades = con_Neo4j.extraer_all_entidades_neo4j()

    con_Neo4j.añadir_embeddings_como_propiedad_neo4j(entidades, embed_model_st)

    con_Neo4j.crear_vector_index_neo4j(
        vector_index_name,
        vector_index_dim,
        similarity_func_index,
    )

    con_Neo4j.crear_fulltext_index(text_index_name)

    print("CREACION DE EMBEDDINGS Y TEXT INDEX EN Neo4j OK")