-- Databricks notebook source
-- MAGIC %md
-- MAGIC Obteniendo los datos ingestados desde el DBMS y creandolos en una tabla

-- COMMAND ----------

DROP TABLE IF EXISTS CASO_MINOIL;

CREATE TABLE CASO_MINOIL
USING csv
OPTIONS (path "/Workspace/Repos/de.vil123@hotmail.com/caso_minoil/tabla_ubicacion_cliente.csv", header "true")


-- COMMAND ----------

describe CASO_BERKA

-- COMMAND ----------

DROP TABLE IF EXISTS CASO_BERKA;

CREATE TABLE CASO_BERKA
(CLIENT_ID INT,
BIRTH_NUMBER STRING,
TYPES STRING,
FREQUENCY STRING,
DATES STRING,
LOAN_DATE STRING,
AMOUNT DOUBLE,
DURATION STRING,
PAYMENTS DOUBLE,
STATUS STRING,
CHURN_Y INT
)
USING csv
OPTIONS (path "/FileStore/credit_card_candidate_churn_y.csv", header "true")


-- COMMAND ----------

select *
from CASO_BERKA

-- COMMAND ----------

-- MAGIC %python
-- MAGIC berka_pd = spark.read.csv("/FileStore/credit_card_candidate_churn_y.csv", header="true", inferSchema="true").toPandas()
-- MAGIC berka_pd.drop('client_id', axis=1, inplace=True)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC type(berka_pd)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC berka_pd.head()
