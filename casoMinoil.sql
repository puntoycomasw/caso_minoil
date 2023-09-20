-- Databricks notebook source
-- MAGIC %md
-- MAGIC Obteniendo los datos ingestados desde el DBMS y creandolos en una tabla

-- COMMAND ----------

DROP TABLE IF EXISTS CASO_MINOIL;

CREATE TABLE CASO_MINOIL
USING csv
OPTIONS (path "/FileStore/disable_client_candidate.csv", header "true")


-- COMMAND ----------

describe CASO_MINOIL

-- COMMAND ----------

DROP TABLE IF EXISTS CASO_MINOIL;

CREATE TABLE CASO_MINOIL
(ruta STRING, vendedor STRING, lista_precio STRING, nombre_cliente STRING, canal	 STRING, ciudad	 STRING, direccion STRING, dias_credito	 STRING, limite_credito	 INT, zona STRING, desabilitado_churn INT, excepcion INT, visitado_churn STRING, visitado_DMS STRING
)
USING csv
OPTIONS (path "/FileStore/disable_client_candidate.csv", header "true")


-- COMMAND ----------

select *
from CASO_MINOIL

-- COMMAND ----------

describe CASO_MINOIL

-- COMMAND ----------

-- MAGIC %python
-- MAGIC path = "/FileStore/disable_client_candidate.csv"
-- MAGIC minoil_pd = spark.read.csv(path, header="true", inferSchema="true").toPandas()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC type(minoil_pd)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC minoil_pd.head(5)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import re
-- MAGIC def group_vars_by(var):
-- MAGIC   return minoil_pd.groupby(var).size()
-- MAGIC def extract_days(value):
-- MAGIC     pattern = r'\d+'
-- MAGIC     result = re.search(pattern, str(value))
-- MAGIC     if result:
-- MAGIC         return int(result.group())
-- MAGIC     else:
-- MAGIC         return 0
-- MAGIC minoil_pd['dias_credito'] = minoil_pd['dias_credito'].apply(extract_days)
-- MAGIC group_vars_by('dias_credito')

-- COMMAND ----------

-- MAGIC %python
-- MAGIC minoil_pd.describe()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC minoil_pd.head(5)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ONE-HOT ENCODING

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #features = ['ruta', 'vendedor', 'lista_precio', 'nombre_cliente', 'canal', 'ciudad',
-- MAGIC #       'direccion', 'dias_credito', 'zona',
-- MAGIC #       'desabilitado_churn', 'excepcion', 'visitado_churn', 'visitado_DMS']

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #import pandas as pd
-- MAGIC #dummies = pd.get_dummies(minoil_pd[features])
-- MAGIC #minoil_transformado = pd.concat([minoil_pd.drop(features, axis=1), dummies], axis=1)
-- MAGIC #minoil_transformado.head(10)
-- MAGIC !pip install pyspark
-- MAGIC !pip install matplotlib

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC from pyspark.sql import SparkSession
-- MAGIC from pyspark.ml.feature import VectorAssembler, StringIndexer
-- MAGIC from pyspark.ml.classification import LogisticRegression
-- MAGIC from pyspark.ml import Pipeline
-- MAGIC from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
-- MAGIC from pyspark.sql.functions import col

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC # Cargar el conjunto de datos desde un archivo CSV
-- MAGIC data = spark.read.csv(path, header=True, inferSchema=True)
-- MAGIC data.select("desabilitado_churn").distinct().show()
-- MAGIC
-- MAGIC # Filtrar las filas que tienen valores numéricos y que son igual a 0 o 1 en la columna "desabilitado_churn"
-- MAGIC data = data.filter((col("desabilitado_churn").cast("double").isin([0, 1])))
-- MAGIC
-- MAGIC # Mostrar los valores únicos después del filtro
-- MAGIC data.select("desabilitado_churn").distinct().show()
-- MAGIC # Eliminar filas con valores nulos en las columnas 'dias_credito' y 'limite_credito'
-- MAGIC data = data.dropna(subset=['dias_credito'])
-- MAGIC
-- MAGIC # Convertir columnas numéricas a tipo de datos entero
-- MAGIC data = data.withColumn("dias_credito", col("dias_credito").cast("integer"))
-- MAGIC data = data.withColumn("limite_credito", col("limite_credito").cast("integer"))
-- MAGIC
-- MAGIC # Convertir la columna 'desabilitado_churn' a tipo numérico usando StringIndexer
-- MAGIC indexer = StringIndexer(inputCol="desabilitado_churn", outputCol="label")
-- MAGIC data = indexer.fit(data).transform(data)
-- MAGIC
-- MAGIC # Definir las características
-- MAGIC feature_cols = ['dias_credito', 'limite_credito']
-- MAGIC vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Crear una sesión de Spark
-- MAGIC spark = SparkSession.builder.appName("LogisticRegressionExample").getOrCreate()
-- MAGIC path = "/FileStore/disable_client_candidate.csv"
-- MAGIC
-- MAGIC # Cargar el conjunto de datos desde un archivo CSV
-- MAGIC data = spark.read.csv(path, header=True, inferSchema=True)
-- MAGIC
-- MAGIC # Filtrar las filas que tienen valores numéricos y que son igual a 0 o 1 en la columna "desabilitado_churn"
-- MAGIC data = data.filter((col("desabilitado_churn").cast("double").isin([0, 1])))
-- MAGIC
-- MAGIC # Eliminar filas con valores nulos en las columnas 'dias_credito' y 'limite_credito'
-- MAGIC data = data.dropna(subset=['dias_credito'])
-- MAGIC
-- MAGIC # Convertir columnas numéricas a tipo de datos entero
-- MAGIC data = data.withColumn("dias_credito", col("dias_credito").cast("integer"))
-- MAGIC data = data.withColumn("limite_credito", col("limite_credito").cast("integer"))
-- MAGIC
-- MAGIC # Convertir la columna 'desabilitado_churn' a tipo numérico usando StringIndexer
-- MAGIC indexer = StringIndexer(inputCol="desabilitado_churn", outputCol="label")
-- MAGIC data = indexer.fit(data).transform(data)
-- MAGIC
-- MAGIC # Definir las características
-- MAGIC feature_cols = ['dias_credito', 'limite_credito']
-- MAGIC vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
-- MAGIC
-- MAGIC # Crear un modelo de Regresión Logística
-- MAGIC logistic_regression = LogisticRegression(featuresCol="features")
-- MAGIC
-- MAGIC # Crear un pipeline para ensamblar y entrenar el modelo
-- MAGIC pipeline = Pipeline(stages=[vector_assembler, logistic_regression])
-- MAGIC
-- MAGIC # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
-- MAGIC train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
-- MAGIC
-- MAGIC # Entrenar el modelo
-- MAGIC model = pipeline.fit(train_data)
-- MAGIC
-- MAGIC # Realizar predicciones en el conjunto de prueba
-- MAGIC predictions = model.transform(test_data)
-- MAGIC
-- MAGIC # Calcular el área bajo la curva PR (Precision-Recall)
-- MAGIC evaluator_precision_recall = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
-- MAGIC precision_recall = evaluator_precision_recall.evaluate(predictions)
-- MAGIC
-- MAGIC # Obtener el área bajo la curva ROC (AUC)
-- MAGIC evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
-- MAGIC auc = evaluator_auc.evaluate(predictions)
-- MAGIC
-- MAGIC # Obtener las tasas de falsos positivos (FPR) y verdaderos positivos (TPR)
-- MAGIC roc = model.stages[-1].summary.roc.toPandas()
-- MAGIC fpr = roc['FPR']
-- MAGIC tpr = roc['TPR']
-- MAGIC
-- MAGIC # Calcular el valor del AUC (Área Bajo la Curva ROC)
-- MAGIC roc_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(predictions)
-- MAGIC
-- MAGIC # Graficar la curva ROC
-- MAGIC plt.figure(figsize=(8, 6))
-- MAGIC plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
-- MAGIC plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
-- MAGIC plt.xlim([0.0, 1.0])
-- MAGIC plt.ylim([0.0, 1.05])
-- MAGIC plt.xlabel('Tasa de Falsos Positivos (FPR)')
-- MAGIC plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
-- MAGIC plt.title('Curva ROC')
-- MAGIC plt.legend(loc="lower right")
-- MAGIC plt.show()
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC # Imprimir el área bajo la curva ROC (AUC)
-- MAGIC print(f"AUC del modelo: {auc:.2f}")
-- MAGIC
-- MAGIC print(f"Área bajo la curva Precision-Recall: {precision_recall:.2f}")
-- MAGIC
-- MAGIC f1_score=0
-- MAGIC # Imprimir las métricas de precisión y recall
-- MAGIC print(f"Precisión del modelo (F1 Score): {f1_score:.2f}")
-- MAGIC
-- MAGIC # Detener la sesión de Spark
-- MAGIC spark.stop()
-- MAGIC
