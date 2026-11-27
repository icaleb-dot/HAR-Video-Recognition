# HAR-Video-Recognition

Proyecto de **Reconocimiento de Actividades Humanas (HAR)** basado en video sobre un subconjunto de **UCF101**, desarrollado en equipo con PyTorch.

## Actividades objetivo

1. `ApplyLipstick`
2. `Archery`
3. `BabyCrawling`
4. `BalanceBeam`
5. `BandMarching`
6. `BaseballPitch`
7. `Basketball`
8. `BasketballDunk`
9. `BenchPress`

*(Nota: la configuración de clases se centraliza en `config/config.yaml`.)*

---

## Estructura del repositorio

```text
HAR-Video-Recognition/
├── config/
│   └── config.yaml              # Hiperparámetros, rutas, clases y preprocesamiento
├── data/                        # Ignorado en Git (raw, processed, splits)
├── checkpoints/                 # Pesos guardados de modelos (.pth/.pt, ignorados)
├── results/
│   ├── figures/                 # Gráficas (distribución, matriz de confusión, curvas)
│   └── metrics/                 # CSVs y reportes de métricas
├── notebooks/                   # Experimentos y exploración
├── experiments/                 # Scripts o bitácoras de ablaciones
├── src/
│   ├── data/
│   │   ├── extract_dataset.py   # Extracción de UCF101_subset.tar
│   │   ├── dataset_analysis.py  # Análisis exploratorio del dataset
│   │   ├── transforms.py        # Preprocesamiento espacial y data augmentation temporal
│   │   ├── dataset.py           # UCF101VideoDataset y create_dataloaders()
│   │   └── test_pipeline.py     # Prueba rápida (smoke test) del contrato de datos
│   ├── models/                  # Arquitecturas (Baseline CNN+RNN, R3D-18, etc.)
│   ├── training/                # Loop de entrenamiento, early stopping y checkpoints
│   └── evaluation/              # Métricas en test, matriz de confusión y análisis
├── requirements.txt
└── enviroment.yml
```

---

## Configuración del entorno

### Opción A: `pip` + `venv`
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Opción B: `conda`
```bash
conda env create -f enviroment.yml
conda activate har_project
```

> **Nota sobre PyTorch (`torch` / `torchvision`):** Si usas `conda` (o quieres una versión específica de CUDA para tu GPU), instala `torch` y `torchvision` por separado siguiendo las instrucciones oficiales de [pytorch.org](https://pytorch.org/get-started/locally/) según tu hardware (GPU CUDA o solo CPU).

---

## Preparación de datos y pruebas

1. Coloca el archivo `UCF101_subset.tar` en `data/raw/`.
2. Extrae los splits (`train`, `val`, `test`):
   ```bash
   python -m src.data.extract_dataset
   ```
3. (Opcional) Genera el análisis exploratorio y `results/metrics/dataset_metadata.csv`:
   ```bash
   python -m src.data.dataset_analysis
   ```
4. Verifica que el pipeline de preprocesamiento y los `DataLoader`s funcionan correctamente:
   ```bash
   python -m src.data.test_pipeline
   ```

---

## Contrato de Datos (para Modelado y Evaluación)

Todo el equipo debe consumir los datos mediante `create_dataloaders("config/config.yaml")` en `src/data/dataset.py`:

```python
from src.data import create_dataloaders

dataloaders, class_to_idx = create_dataloaders("config/config.yaml")
train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]
```

- **Tensor de entrada (`clips`):** `torch.float32` con forma **`(B, C, T, H, W)`**
  - `B`: `batch_size` (definido en `config/config.yaml`)
  - `C`: `3` canales (RGB normalizados con media/desviación estándar de Kinetics-400)
  - `T`: `num_frames` frames muestreados uniformemente por video (por defecto `16`)
  - `H, W`: `img_size` (por defecto `112 x 112`)
- **Etiquetas (`labels`):** `torch.int64` con forma **`(B,)`** y valores en `[0, num_classes - 1]`.
- **Mapeo de clases (`class_to_idx`):** Diccionario `{nombre_clase: indice}` consistente en `train`, `val` y `test`.

---

## Flujo de trabajo en equipo (GitHub)

- **Integrante 1 (Datos e Integración):** `src/data/`, `config/config.yaml`, integración de PRs e interfaz de inferencia (`gradio`).
- **Integrante 2 (Modelos y Entrenamiento):** `src/models/` y `src/training/`.
- **Integrante 3 (Evaluación y Experimentos):** `src/evaluation/`, `results/` y `experiments/`.

Trabajar en ramas separadas (`feat/models-training`, `feat/evaluation-analysis`) y abrir Pull Requests hacia `main`. Nunca subir archivos de `data/`, `checkpoints/*.pth` ni `results/`.
