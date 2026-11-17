# Monitorización exploratoria del dataset sintético

Este módulo compara la distribución de las **18 variables** que recibe el modelo
de scoring crediticio. Los informes son descriptivos: **data drift no demuestra
degradación del rendimiento del modelo**, ni justifica automáticamente un
reentrenamiento. El proyecto no utiliza datos reales ni aplica decisiones de
crédito.

## Qué calcula

- **PSI (Population Stability Index):** compara histogramas numéricos
  derivados de cuantiles de la muestra de referencia; para categorías compara
  sus frecuencias. Utiliza suavizado para impedir logaritmos de cero.
- **Calidad:** tasa de valores ausentes en ambas cohortes y diferencia absoluta.
- **Categorías nuevas:** número de categorías de situacion_laboral presentes
  solo en la cohorte reciente.
- **Estado agregado:** ok, warning, critical o insufficient_data.

Los umbrales didácticos son PSI >= 0,10 (aviso) y >= 0,25 (crítico), o
incrementos absolutos de faltantes de 5 y 15 puntos porcentuales. **No son
umbrales regulatorios ni validados para este banco.** La muestra de entrenamiento
del modelo tiene solo 119 préstamos etiquetados. Con tamaños pequeños, el PSI
puede ser muy inestable; por debajo de 30 observaciones en cualquier cohorte
se informa insufficient_data en vez de declarar una alarma conclusiva.

## Ejecución con el ZIP producido en Colab

Instalar las dependencias:

    python -m pip install -r requirements.txt

Desde la raíz del repositorio:

    python -m src.monitoring \
      --dataset-zip /ruta/dataset_bbva_sintetico_v2.zip \
      --baseline-partition TRAIN \
      --current-partition MONITORIZACION_SIN_ETIQUETA_MADURA \
      --output-dir reports/monitoring

La aplicación abre features_solicitud.csv directamente desde el ZIP.
**No requiere ni usa las etiquetas** incumplimiento_90d_12m, por lo que
permite examinar cohortes futuras con resultados todavía no observados.

**Importante:** el ejemplo anterior compara cohortes completas de solicitudes,
no exactamente la población de 119 préstamos aprobados y etiquetados utilizada
para entrenar el clasificador. Las cohortes de producción deben elegirse
consistentemente según la población a la que se aplique el servicio.
Una comparación con TEST no debe utilizarse para reajustar el modelo cuya
evaluación ya se cerró.

## Ejecución con dos CSV independientes

    python -m src.monitoring \
      --baseline-csv /ruta/referencia.csv \
      --current-csv /ruta/solicitudes_recientes.csv \
      --output-dir reports/monitoring

Ambos archivos necesitan las 18 columnas del contrato de inferencia
(src/scoring.py) con los mismos significados y unidades. No es necesario
añadir identificadores o variables de resultado. Con
--fail-on-critical, el proceso termina con código 2 ante un estado crítico;
sin esa opción, escribe el informe y devuelve 0 incluso cuando detecta drift.

Se generan:

- drift_report.json: métricas agregadas para procesos automatizados.
- drift_report.md: tabla legible para revisión y GitHub Actions.

Los informes no contienen nombres, IDs de clientes ni filas de entrada.
Se excluyen de Git mediante .gitignore. Revise igualmente el acceso a
los archivos de entrada, incluso cuando sean sintéticos.

## Demostración autocontenida

    python -m examples.drift_demo --output-dir reports/monitoring

El ejemplo construye muestras ficticias deterministas y desplaza
deliberadamente el gasto mensual y los accesos digitales. La alarma
critical que aparece es **esperada y simulada**, no un incidente del
sistema bancario ni una medición de los datos reales del trabajo.

GitHub Actions ejecuta los tests del monitor y una demostración, adjuntando
los informes agregados como artefacto al workflow. La demostración **no
monitoriza de forma continua la API ni la llegada de datos reales**. Para
operación continua haría falta una fuente segura de solicitudes recientes,
un calendario apropiado, supervisión de privacidad y una referencia
congelada y versionada.

## Pruebas

    python -m unittest discover -s tests -p 'test_monitoring.py' -v

Incluyen muestras sin cambio, desplazamiento numérico, categorías
desconocidas, faltantes, referencias constantes, muestras pequeñas y uso
de la CLI con CSV y ZIP.

## Diferencia entre drift y degradación predictiva

El PSI mide cambios en la distribución de X. Para medir **performance
degradation** se necesitan las probabilidades históricas de cada predicción,
el resultado real posterior ya maduro, muestras suficientes y métricas
comparables como ROC-AUC, Average Precision y Brier Score. En este proyecto
las etiquetas de impago requieren 12 meses y podrían no estar disponibles
al evaluar cohortes recientes. Por ello este módulo **no calcula métricas de
desempeño futuro ni dispara reentrenamientos**.

Los cambios introducidos por el escenario artificial ESCENARIO_DRIFT_2026
podrían reflejarse en los reportes, pero no se supone su detección sin ejecutar
el análisis sobre el ZIP concreto. No se transfieren datos a servicios
externos; solo las pruebas autocontenidas utilizan GitHub Actions.
