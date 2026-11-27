# Integrante 1: Datos e infraestructura

Esta nota registra las decisiones y el estado de la capa de datos para que modelos, entrenamiento y evaluación compartan la misma interfaz.

## Responsabilidades

- Mantener `config/config.yaml` como fuente de verdad para rutas, clases, dimensiones y parámetros del `DataLoader`.
- Extraer y analizar el subconjunto UCF101.
- Entregar `UCF101VideoDataset` y `create_dataloaders()` con un contrato estable.
- Documentar la preparación local, el formato de tensores y las reglas de colaboración.

## Datos y rutas

El archivo `data/raw/UCF101_subset.tar` se conserva localmente y no se versiona. `python -m src.data.extract_dataset` lo extrae a `data/processed/UCF101_subset/`, que contiene `train/`, `val/` y `test/`, cada uno organizado en carpetas por clase. Las carpetas de datos, checkpoints y resultados se mantienen fuera de Git; los `.gitkeep` solo preservan los directorios vacíos.

Las nueve clases usadas se declaran una vez en `config/config.yaml`. El archivo TAR incluye además `ApplyEyeMakeup`; se omite intencionalmente porque no forma parte del alcance acordado. No cambiar el orden de las clases sin coordinarlo con entrenamiento y evaluación, ya que determina `class_to_idx`.

## Contrato de datos

```python
from src.data import create_dataloaders
loaders, class_to_idx = create_dataloaders("config/config.yaml")
clips, labels = next(iter(loaders["train"]))
```

`clips` es `float32` con forma `(B, 3, T, H, W)`, RGB y normalización Kinetics-400. `labels` es `int64` con forma `(B,)`. Las claves de `loaders` son `train`, `val` y `test`; el mismo mapeo de clases se utiliza en los tres splits. El acceso es por video bajo demanda: no se almacenan todos los frames en memoria.

## Flujo de trabajo

1. Activar el entorno local y seguir las instrucciones de `README.md`.
2. Extraer con `python -m src.data.extract_dataset`.
3. Generar CSVs locales con `python -m src.data.dataset_analysis`.
4. Comprobar formas y tipos con `python -m src.data.test_pipeline`.
5. Trabajar en una rama propia y abrir un Pull Request hacia `main`; nunca incluir datos, pesos ni resultados derivados.

## Archivos de esta capa

- `extract_dataset.py`: extracción con validación de rutas del TAR.
- `dataset_analysis.py`: metadatos de video y conteos por split/clase en `results/metrics/`.
- `transforms.py`: conversión RGB, resize, normalización y augmentación temporal de entrenamiento.
- `dataset.py`: lectura uniforme de frames y creación de `DataLoader`s.
- `test_pipeline.py`: comprobación rápida del contrato compartido.

El smoke test está implementado; ejecútalo localmente después de extraer el dataset y configurar el entorno. Estos cambios aún no se han integrado en `main` ni enviado al remoto.
