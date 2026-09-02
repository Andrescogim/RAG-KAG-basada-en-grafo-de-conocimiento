import sys
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.configuracion.build_retrieval_config import build_retrieval_config
from src.utils.funciones_guardado import guardar_resultados_grafo
from src.logica.graph_retrieval import contestar_2Wiki_con_grafo
from src.utils.funciones_generales import medir_recursos


def main():
 
    parametros = build_retrieval_config()
    resultados, recursos_por_iteracion = contestar_2Wiki_con_grafo(**parametros)
    recursos_general = medir_recursos.acumulado
    metricas_agg = guardar_resultados_grafo(resultados, recursos_por_iteracion, recursos_general)
    print(metricas_agg)
    
if __name__ == "__main__":
    main()
