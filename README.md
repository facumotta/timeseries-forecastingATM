# Trabajo Practico Final - Procesamiento Avanzado de Señales 1C 2025
### Facundo Alvarez Motta, Manuel Horn, Ignacio Rodriguez Sañudo


# Predicción de Extracción en Cajeros Automáticos

Este trabajo práctico se centra en la predicción de montos dispensados por cajeros automáticos utilizando modelos avanzados de series temporales. La solución implementada busca anticipar la demanda futura de efectivo con el fin de optimizar la logística de abastecimiento y minimizar faltantes o excesos.

## 📌 Objetivo

El objetivo principal es predecir la cantidad de dinero que será dispensado por cada cajero en los días siguientes, basándonos en el comportamiento histórico, variables temporales (como día de la semana y feriados), y otras señales disponibles.

## 🧠 Modelo Utilizado: TimesNet

Para abordar el problema, utilizamos **[TimesNet](https://arxiv.org/pdf/2210.02186)**, un modelo de última generación para series temporales multivariadas. Este modelo está diseñado específicamente para capturar patrones temporales complejos de manera eficiente y ha demostrado un excelente desempeño en benchmarks de forecasting.

> TimesNet combina mecanismos de convolución con atención jerárquica para capturar tanto patrones locales como globales en los datos temporales.

## 📁 Estructura del Proyecto

- `data/`: contiene los datos crudos y procesados.
- `models/`: contiene el modelo entrenado y scripts relacionados.
- `notebooks/`: notebooks exploratorios y de entrenamiento.
- `scripts/`: scripts de entrenamiento y evaluación.
- `results/`: gráficos y métricas obtenidas durante la experimentación.

## ⚙️ Tecnologías y Librerías

- Python 3.10
- PyTorch
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- [TimesNet](https://github.com/thuml/Time-Series-Library) (implementación adaptada)

## 🔍 Métricas de Evaluación

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Se evalúa el desempeño tanto en el horizonte total de predicción como día por día.

## 📈 Ejemplo de Resultados

![plot](results/prediction_example.png)

## 🚀 Próximos Pasos

- Afinar hiperparámetros del modelo.
- Incorporar nuevas variables externas (como clima, inflación, eventos).
- Implementar una interfaz web simple para visualizar predicciones por cajero.

---

> Trabajo realizado como parte de la cursada de [nombre de la materia] - [Universidad].
