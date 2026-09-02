import sys
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.configuracion.build_deduplicacion_config import build_config_deduplicacion
from src.logica.deduplicacion import deduplicar_nodos


def main():
    
    parametros = build_config_deduplicacion()
    deduplicar_nodos(**parametros)
    
if __name__ == "__main__":
    main()