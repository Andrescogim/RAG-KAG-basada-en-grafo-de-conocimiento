import sys
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.configuracion.build_insercion_tripletas_config import build_insercion_tripletas_config
from src.logica.insercion_tripletas import insercion_tripletas

def main():
    
    parametros = build_insercion_tripletas_config()
    insercion_tripletas(**parametros)

if __name__ == "__main__":
    main()