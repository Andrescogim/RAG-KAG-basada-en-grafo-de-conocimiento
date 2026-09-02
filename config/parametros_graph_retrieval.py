from pathlib import Path

DATABASE_NEO = "2wiki.gold.500"
OPCIONES_LLM = {
    "temperature": 0,
}
PARAMETROS_GRAFO = {
    "split": "train",
    "rango_in_data": 0,
    "rango_fin_data": 1,
    "vector_index": "entity_embedding_index",
    "fulltext_index": "entidadesIndex",
    "n_saltos": 2,
    "llm_name": "qwen2.5:7b-instruct",
    "opciones_llm": OPCIONES_LLM,
    "n_rel_max": 35,
    "min_score_parcial": 2,
    "min_score_fuzzy": 2,
    "n_final_fuzz_parc": 3,
    "n_resultados_embedding": 3,
    "peso_tripleta": 0.65,
    "peso_rel": 0.35,
    "n_maximos": 3,
    "min_score": 0.3,
}
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
NER_MODEL = "en_core_web_sm"

PROMPT_BASE = """
        You are a strict question-answering assistant. 

        1. You MUST answer the question using ONLY the facts provided in the [knowledge graph] section.
        2. Do NOT use any external knowledge or assume anything.
        3. Be direct and concise. Do NOT explain your answer. Give only the exact name, place, date, or "Yes"/"No" as requested.
        4. If the [Knowledge graph] does not contain enough information to answer the question, you MUST reply exactly with: "I don't know".
        
        [Knowledge graph]:
        {tripletas_formateadas}

        Question:
        {question}

        Answer:
        """

COMENTARIOS = "NUEVO RERANKER ; min_score RELAJADO "
RUTA_GUARDADO_RESULTADOS = Path("outputs/resultados")
RUTA_GUARDADO_REGISTRO = Path("outputs/registro")
RUTA_GUARDADO_RECURSOS = Path("outputs/medicion_recursos")
