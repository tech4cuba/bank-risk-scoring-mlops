# Reproducir y probar el modelo real en la API

Esta guía utiliza únicamente **datos bancarios sintéticos**. No hay registros reales de clientes. El objetivo es demostrar que el modelo entrenado puede guardarse, abrirse y realizar una predicción con la misma API que construimos para el proyecto.

## 1. Preparación

Descarga el repositorio y sitúate en su carpeta. Instala las dependencias:

```bash
python -m pip install -r requirements-test.txt
```

Conserva tu archivo `dataset_bbva_sintetico_v2.zip` generado desde Colab **fuera del repositorio**. El ZIP debe contener `features_solicitud.csv` y `solicitudes_prestamos.csv`.

## 2. Reconstruir el modelo

```bash
python -m scripts.reproducir_modelo \
  --dataset-zip "/ruta/dataset_bbva_sintetico_v2.zip" \
  --output-dir models
```

El script usa las **18 variables**, las mismas transformaciones y los parámetros de regresión logística del cuaderno. Ajusta el modelo **solo con TRAIN**. No usa VALIDACION ni TEST para ajustar parámetros.

Se crean, dentro de la carpeta ignorada `models/`:

- `modelo_scoring_crediticio_v1.joblib`: modelo entrenado y listo para cargar.
- `metadatos_modelo_v1.json`: versión de bibliotecas, cantidad de observaciones por partición y huellas SHA-256.

Se trata de una **reconstrucción** con el dataset, no de la recuperación del archivo originalmente exportado por Colab. Distintas versiones de bibliotecas pueden cambiar ligeramente los valores o impedir abrir el archivo en otro equipo.

## 3. Probar una predicción completa

```bash
python -m scripts.probar_modelo_real \
  --model models/modelo_scoring_crediticio_v1.joblib \
  --metadata models/metadatos_modelo_v1.json \
  --sample examples/solicitud_sintetica.json
```

Este paso:

1. Comprueba que el archivo del modelo coincide con la huella guardada en los metadatos.
2. Carga el modelo y comprueba el orden de sus 18 variables.
3. Abre la API FastAPI de prueba y verifica los endpoints de salud.
4. Envía la solicitud JSON a `POST /predict`.
5. Verifica que la probabilidad y el score coinciden con los del modelo llamado directamente.

**Resultado observado con el ZIP sintético usado en este proyecto y las dependencias locales:** 0,070229 de probabilidad de incumplimiento (7,02 %) y 628 puntos. Es solo un ejemplo ficticio, no una recomendación financiera.

## 4. Abrir la API para explorarla

Con el archivo del modelo en `models/`, ejecuta desde la raíz:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

En el navegador, abre `http://127.0.0.1:8000/docs`. Allí podrás enviar el JSON de ejemplo y examinar la respuesta.

Para usar Docker, consulta [la guía de API](api.md). El contenedor necesita que se monte externamente el archivo `.joblib`, por lo que el modelo **no se incorpora a la imagen de CI**.

## 5. Qué verifica realmente GitHub Actions

El workflow ejecuta una prueba de entrenamiento, guardado y predicción mediante FastAPI, pero **con un pequeño conjunto de datos ficticios de prueba** que se crea durante CI. Así se comprueba que el código funciona sin subir el dataset original al repositorio.

La reproducción completa del modelo a partir del ZIP bancario se realizó localmente. El archivo `.joblib` no se publica en Git. Los nuevos scripts pertenecen al trabajo posterior a `v1.0.0` y no cambian la publicación ya etiquetada.

**Seguridad y limitaciones:** los archivos Joblib pueden ejecutar código al cargarse. Utiliza exclusivamente modelos propios o de origen plenamente confiable. Una huella SHA-256 detecta alteraciones, pero no acredita por sí sola que el proveedor sea confiable. El modelo es experimental, entrenado sobre datos simulados, y no sirve para aprobar ni rechazar préstamos reales.
