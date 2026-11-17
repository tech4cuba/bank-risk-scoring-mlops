# Historial de cambios

## [1.0.0] - 2026-11-17

Primera pre-release académica del prototipo MLOps.

### Incorporado
- Generación de datos bancarios sintéticos y cuadernos de investigación.
- Scoring mediante regresión logística, 18 características y transformación del score.
- Servicio experimental FastAPI y construcción Docker.
- Pruebas CI para cuadernos, API y contenedor.
- Monitor exploratorio de PSI, faltantes y categorías desconocidas, con reportes agregados.
- Flujo de ramas `feature/*`, `develop`, `release/*` y `main`.

### Limitaciones
- Modelo entrenado no distribuido; no se habilita inferencia sin artefacto confiable.
- Evaluación retrospectiva en datos sintéticos, sin validación de producción.
- Data drift no equivale a deterioro de métricas predictivas.

Véanse las notas detalladas en [docs/releases/v1.0.0.md](docs/releases/v1.0.0.md).
