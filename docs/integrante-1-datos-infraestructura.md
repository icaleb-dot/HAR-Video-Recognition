# Entrega del integrante 1: Datos e infraestructura

## Estado

La capa de datos e infraestructura está implementada en la rama `feat/data-infra` y validada localmente. El [PR #1 hacia `main`](https://github.com/icaleb-dot/HAR-Video-Recognition/pull/1) está abierto y pendiente de revisión; por tanto, los cambios todavía no forman parte de `main`.

Validaciones realizadas con el archivo local `data/raw/UCF101_subset.tar`:

- Extracción correcta a `data/processed/UCF101_subset/`.
- Análisis de 366 videos de las nueve clases configuradas, con CSVs en `results/metrics/` y gráfica en `results/figures/`.
- Smoke test correcto para `train`, `val` y `test`: clips `(4, 3, 16, 112, 112)`, `float32`; etiquetas `(4,)`, `int64`.
- Verificación de que datos, checkpoints y resultados generados están excluidos de Git.

## Qué incluye esta entrega

- `config/config.yaml`: rutas, clases, tamaño y cantidad de frames, normalización, parámetros del `DataLoader` y configuración base de entrenamiento/evaluación.
- `src/data/extract_dataset.py`: extrae el TAR y rechaza rutas de salida fuera del directorio destino y tipos de archivo no admitidos.
- `src/data/dataset.py` y `src/data/transforms.py`: leen videos bajo demanda, muestrean frames uniformemente, convierten BGR a RGB, redimensionan, normalizan y aplican inversión temporal aleatoria solo al split de entrenamiento.
- `src/data/dataset_analysis.py`: genera metadatos por video, conteos por split/clase y gráfica de distribución.
- `src/data/test_pipeline.py`: smoke test del formato y tipos del contrato de datos.
- `.gitignore`: excluye videos, datos procesados, checkpoints, resultados generados, entornos virtuales y archivos del IDE; mantiene `.gitkeep` para preservar directorios.
- `environment.yml`: nombre corregido del manifiesto Conda. PyTorch/torchvision se instalan según el hardware siguiendo la guía enlazada en el README.

## Preparar el entorno y los datos

Desde la raíz del repositorio, activa el entorno del proyecto y coloca `UCF101_subset.tar` en `data/raw/`. Después ejecuta:

```bash
python -m src.data.extract_dataset
python -m src.data.dataset_analysis
python -m src.data.test_pipeline
```

La extracción genera `data/processed/UCF101_subset/{train,val,test}/`. Los CSVs y la figura son artefactos locales; no deben añadirse al commit.

## Contrato para los integrantes de modelos, entrenamiento y evaluación

```python
from src.data import create_dataloaders

loaders, class_to_idx = create_dataloaders("config/config.yaml")
clips, labels = next(iter(loaders["train"]))
```

- `loaders` contiene las claves `train`, `val` y `test`.
- `clips`: `torch.float32`, forma `(B, 3, T, H, W)`, RGB y normalizado con media/desviación de Kinetics-400. La configuración actual usa `T=16`, `H=W=112`.
- `labels`: `torch.int64`, forma `(B,)`, índices de clase entre `0` y `8`.
- `class_to_idx`: mapa estable entre nombre de clase e índice. La capa de salida del clasificador debe tener nueve clases y usar estos mismos índices; no asumir un orden alfabético distinto.
- Los datos se leen desde disco al solicitar cada ejemplo; no se cachea el dataset completo en memoria.

La lista contiene nueve clases configuradas. El TAR también incluye `ApplyEyeMakeup`, que se excluye intencionalmente del contrato actual.

## Handoff al integrante 2: Arquitecturas de modelos

1. Confirma que el PR #1 esté aprobado e integrado en `main`. Luego actualiza tu copia de `main` y crea una rama propia, por ejemplo `feat/models`. Si necesitas empezar antes de la integración, coordina con el equipo para basar tu rama temporalmente en `feat/data-infra`.
2. Implementa las arquitecturas en `src/models/` y haz que su `forward` consuma el batch `clips` con forma `(B, C, T, H, W)`; el modelo produce logits con nueve salidas.
3. Usa `class_to_idx` para interpretar predicciones y etiquetas; no recrees el mapeo de clases en el código del modelo.
4. Mantén el modelo independiente de la lectura de archivos: recibe tensores desde los `DataLoader`s y devuelve logits. Así el integrante 3 puede conectarlo al loop de entrenamiento.
5. No incluyas datos, checkpoints ni métricas en Git. Añade pruebas de forma con tensores sintéticos pequeños y documenta dependencias/preentrenamiento si la arquitectura los necesita.

## Colaboración y publicación

El integrante 1 publicó el commit `ddf5652` en `feat/data-infra`; el PR #1 apunta a `main`. No se debe hacer merge sin la revisión acordada por el equipo. Para cambios nuevos, trabajar en una rama propia y abrir un PR. No subir `data/`, archivos de checkpoints ni resultados generados.
