import sys
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.configuracion.build_construccion_grafo_config import build_construccion_grafo_config
from src.logica.construccion_grafo import construir_grafo


def main():
    
    parametros = build_construccion_grafo_config()
    construir_grafo(
        **parametros,
    )

if __name__ == "__main__":
    main()