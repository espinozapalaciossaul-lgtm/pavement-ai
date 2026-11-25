# PAVEMENT AI

Aplicación web para estimar el Índice de Condición del Pavimento (PCI) con Regresión Lineal.
**SENATI · Huánuco **

## Instalación (en la terminal de Visual Studio Code)

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
python -m pip install -r requirements.txt
```

## Ejecución

```bash
python -m streamlit run app.py
```

Se abre en http://localhost:8501

## Contenido

- `app.py` → aplicación (Streamlit + scikit-learn + Plotly)
- `dataset_pavimento.csv` → 50 inspecciones simuladas
- `registros_inspeccion.csv` → se crea solo al registrar inspecciones desde la app
- `requirements.txt` → dependencias

Los datos son **simulados** con fines educativos.
