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
-- MAGIC minoil_pd = spark.read.csv("/FileStore/disable_client_candidate.csv", header="true", inferSchema="true").toPandas()

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
-- MAGIC features = ['ruta', 'vendedor', 'lista_precio', 'nombre_cliente', 'canal', 'ciudad',
-- MAGIC        'direccion', 'dias_credito', 'zona',
-- MAGIC        'desabilitado_churn', 'excepcion', 'visitado_churn', 'visitado_DMS']

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pandas as pd
-- MAGIC dummies = pd.get_dummies(minoil_pd[features])
-- MAGIC berka_transformado = pd.concat([minoil_pd.drop(features, axis=1), dummies], axis=1)
-- MAGIC berka_transformado.head(10)
