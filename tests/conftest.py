"""Configuración compartida de los tests.

Los tests se corren desde la raíz del repo con `pytest`. Acá se agrega
al path la raíz (para importar entrenamiento y api.main) y docker/ (para
importar inferencia.py, que es un script suelto, sin paquete).
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "docker"))
