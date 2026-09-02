import sys
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.configuracion.build_extraccion_tripletas_config import build_extraccion_tripletas_config
from src.logica.extraccion_tripletas import extraccion_y_guardado


def main():

    parametros = build_extraccion_tripletas_config()
    extraccion_y_guardado(**parametros)

if __name__ == "__main__":
    main()