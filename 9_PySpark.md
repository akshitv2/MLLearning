# 9 PySpark

- Python Api For Apache Spark


## Core Architecture
- ## SparkSession: Entry point for DataFrame and SQL APIs.
- **Cluster Manager**: Manages resources (e.g., YARN, Mesos, Standalone).
- **Driver**: Runs the main function, creates SparkContext, and coordinates tasks.
- **Executor**: Runs tasks on worker nodes, stores data in memory/disk.
- **DAG (Directed Acyclic Graph)**: Represents computation stages; optimized by Catalyst Optimizer.
- **Catalyst Optimizer**: Query optimizer for DataFrame/SQL operations, improves execution plans.
- ## RDD: Resilient Distributed Dataset
    - Distributed Data structure
    - Represents Data in Spark's memory across clusters for parallel computation
    - Immutable : Doesn't change once created
    - Transactions are added to lineage graph instead of track changes
    - Lazy Evaluation -> Doesn't execute Immediately (evaluation is done when neccessary)
    - Fault Tolerant
- ## DataFrame:
    - Higher Level than RDD (RDD are still the basis of these)
    - Still Immutable (all things about RDD still apply)
    - Has tabular structure
    - Lazy Eval: Transformations (e.g., filter, groupBy) are not executed until an action (e.g., show, collect) is called.
  
## RDD vs. DataFrame
- **RDD**: Low-level, functional API; requires manual optimization.
- **DataFrame**: Higher-level, structured API; leverages Catalyst Optimizer for performance.
- **When to Use RDD**: Complex custom operations not supported by DataFrame/SQL APIs.

## Fault Tolerance
- **Lineage**: Tracks transformations to rebuild data if partitions are lost.
- **Checkpointing**: Saves intermediate data to disk (`df.checkpoint()`).

## Data Sources
- **Formats**: CSV, JSON, Parquet, ORC, Avro, JDBC, Delta.
- **Read**: `spark.read.format("csv").load("file.csv")`.
- **Write**: `df.write.format("parquet").save("output")`.
- **Modes**: `overwrite`, `append`, `ignore`, `error`.

- ## Parquet:
    - Parquet is a columnar storage file format that is widely used in big data processing
    - optimized for efficient data storage and retrieval
    - Supports compression (e.g., Snappy, Gzip), reducing storage footprint.
    - Parquet stores data column-wise instead of row-wise.
    - 🟢 When you query only age and salary, Parquet doesn’t read the name column at all, saving disk I/O and can filter
      rows before loading them into memory.
- ## .Collect():
    - Trigger computation of lineage graph and brings all data to driver node
    - Can crash driver if data is too large
    - Only suitable for DEV/small Dataset
    - Use instead:
        - .take(n)
        - .show(n)
- ## .Cache():
    - Marks Dataset to be cached for when you next compute it
    - Shorthand for .persist(StorageLevel.MEMORY_AND_DISK)
    - Caches on cluster memory, Stores on disk if too large for memory
    - Heavy objects can cause OOM Crash
- ## .unpersist():
    - Remove from cache

- ## Broadcast Join:
    - Join optimization technique used when joining a large Dataframe with a small one.
    - Instead of shuffling large Dataframe across cluster spark broadcasts small Dataframe to all worker codes to join
      with each partition of large dataframe
    - Reduces I/O
    - Make sure dataset being broadcasted is small enough to fit in memory of each node.
    - Usage:
      ```python
      from pyspark.sql import SparkSession
      from pyspark.sql.functions import broadcast
      spark = SparkSession.builder.appName("BroadcastJoinExample").getOrCreate()
      # Example DataFrames
      large_df = spark.read.csv("large_data.csv", header=True, inferSchema=True)
      small_df = spark.read.csv("small_data.csv", header=True, inferSchema=True)
      # Using broadcast join
      joined_df = large_df.join(broadcast(small_df), on="id", how="inner")
      ```
- ## User Defined Functions (UDF)
    - Custom User Created Functions that can be registered to spark
    - UDFs let you apply any custom Python logic to DataFrame columns.
    - 🔴 Performance note: UDFs are slower than built-in Spark SQL functions because they require serialization and
      Python-JVM communication.
    - Use built-in functions if possible.
    - 🟢 10-100x Faster than base python function usage
    - Example:
        - ```python
          from pyspark.sql import SparkSession
          from pyspark.sql.functions import udf
          from pyspark.sql.types import IntegerType, StringType
          spark = SparkSession.builder.appName("UDFExample").getOrCreate()
          def square(x):
            if x is not None:
                return x * x
          square_udf = udf(square, IntegerType())```
## Performance Optimization
- **Caching/Persisting**: Store DataFrame in memory (`df.cache()` or `df.persist(StorageLevel.MEMORY_AND_DISK)`).
- **Partitioning**: Control data distribution (`df.repartition(10)` or `df.coalesce(2)`).
- **Broadcast Join**: Optimize small-table joins (`spark.sql("SELECT /*+ BROADCAST(t2) */ * FROM t1 JOIN t2")`).
- **Shuffle Tuning**: Adjust `spark.sql.shuffle.partitions` (default: 200).
- **Skew Handling**: Address data skew with salting or repartitioning.
- **AQE (Adaptive Query Execution)**: Automatically optimizes query plans (enabled by default in Spark 3.0+).

- ## Repartition
    - Used to increase or decrease the number of partitions.
    - 🔴 repartition always performs a full shuffle, so it can be expensive.
    - Shuffles data
        - ```python 
            # Assume df has 4 partitions
            df_repart = df.repartition(8)  # Now it has 8 partitions```

- ## Coalesce
    - Reduce the number of partitions
    - Doesn't shuffle data
    - Used to merge small partitions to reduce overhead (common before writing to disk)
    - If you specify a larger number of partitions than the current ones, Spark will NOT perform a shuffle by default,
      and the extra partitions may end up empty
    - ```python 
            # Assume df has 8 partitions
            df_repart = df.coalesce(8)  # Now it has 4 partitions```
- ## Skew
  - When data is unevenly distributed across partitions.

## Predicate Pushdown
## ETL (Extract Transform Load)
## Graph:
  1. Graph Frames: 
  
## STreaming
## Model Persistence

## Best Practices
- Use DataFrame/SQL over RDDs for better performance.
- Minimize shuffles (e.g., reduce joins, groupBy).
- Cache strategically to avoid recomputation.
- Use appropriate data formats (e.g., Parquet for columnar storage).
- Monitor and tune partition sizes based on data volume.

## Storage Levels

| Storage Level           | Use Disk | Use Memory | Use Off-Heap | Deserialized | Replication |
|-------------------------|----------|------------|--------------|--------------|-------------|
| `DISK_ONLY`             | Yes      | No         | No           | -            | 1           |
| `DISK_ONLY_2`           | Yes      | No         | No           | -            | 2           |
| `MEMORY_ONLY`           | No       | Yes        | No           | Yes          | 1           |
| `MEMORY_ONLY_2`         | No       | Yes        | No           | Yes          | 2           |
| `MEMORY_ONLY_SER`       | No       | Yes        | No           | No           | 1           |
| `MEMORY_ONLY_SER_2`     | No       | Yes        | No           | No           | 2           |
| `MEMORY_AND_DISK`       | Yes      | Yes        | No           | Yes          | 1           |
| `MEMORY_AND_DISK_2`     | Yes      | Yes        | No           | Yes          | 2           |
| `MEMORY_AND_DISK_SER`   | Yes      | Yes        | No           | No           | 1           |
| `MEMORY_AND_DISK_SER_2` | Yes      | Yes        | No           | No           | 2           |
| `OFF_HEAP`              | No       | No         | Yes          | No           | 1           |

# PySpark CheatSheet

## Initializing PySpark

```python
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder
.appName("MyApp")
.getOrCreate()
```

## Reading Config

```python
# where spark is created in last command
spark.sparkContext.getConf()
```

## Reading Data

```python
# Read CSV
df = spark.read.csv("file.csv", header=True, inferSchema=True)

# Read JSON
df = spark.read.json("file.json")

# Read Parquet
df = spark.read.parquet("file.parquet")

df = spark.read.textFile()

```

## Load Into RDDs

```python
# Load into DataFrame
spark.createDataFrame(data, ["colAName", ...])

# Create RDD from existing python object like list
SparkContext.parallelize(someList)
```

## Writing Data

```python
# Write to CSV
df.write.csv("output.csv", header=True, mode="overwrite")

# Write to Parquet
df.write.parquet("output.parquet", mode="overwrite")

# Write to JSON
df.write.json("output.json", mode="overwrite")
```

## DataFrame Operations

```python
# Show DataFrame
df.show(5)  # Display first 5 rows (Pretty Prints)
df.head(5)  # Get first 5 rows
df.tail(5)  # Get last 5 rows
df.printSchema()  # Display schema

# Select columns
df.select("col1", "col2").show()

# Filter rows
df.filter(df.col1 > 100).show()

# Rename Columns
df.withColumnRenamed("oldName", "newName")

# Group by and aggregate
df.groupBy("col1").agg({"col2": "sum"}).show()

# Sort
df.orderBy("col1", ascending=False).show()

# Drop duplicates
df.dropDuplicates(["col1"]).show()

# Drop columns
df.drop("col1").show()
```

## Collect, Count, and Take

```python
# Collect: Retrieve all rows as a list of Row objects (use cautiously, pulls data to driver)
data = df.collect()  # Returns entire DataFrame as a list
for row in data:
    print(row["col1"])

# Count: Get total number of rows
row_count = df.count()  # Returns a single integer
print(f"Total rows: {row_count}")

# Take: Retrieve first n rows as a list of Row objects
first_five = df.take(5)  # Returns first 5 rows
for row in first_five:
    print(row)
```

## Joins

```python
# Inner join
df_joined = df1.join(df2, ["key"], "inner")

# Left join
df_joined = df1.join(df2, ["key"], "left")

# Right join
df_joined = df1.join(df2, ["key"], "right")

# Full outer join
df_joined = df1.join(df2, ["key"], "outer")

# Cross Join
# 🔴 Rows Explode very quickly with this so very dangerous 
df_joined = df1.crossJoin(df2)
```

## Handling Missing Values

```python
# Drop rows with nulls
df.na.drop().show()

# Fill nulls with a value
df.na.fill(0).show()

# Fill nulls in specific columns
df.na.fill({"col1": 0, "col2": "unknown"}).show()
```

## SQL Queries

```python
# Register DataFrame as a temporary view
df.createOrReplaceTempView("table_name")

# Run SQL query
result = spark.sql("SELECT col1, COUNT(*) FROM table_name GROUP BY col1")
result.show()
```

## Window Functions

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank

# Define window
windowSpec = Window.partitionBy("col1").orderBy("col2")

# Add row number
df.withColumn("row_num", row_number().over(windowSpec)).show()

# Add rank
df.withColumn("rank", rank().over(windowSpec)).show()
```

## UDF (User-Defined Functions)

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


# Define UDF
def my_function(x):
    return x.upper() if x else x


udf_my_function = udf(my_function, StringType())

# Apply UDF
df.withColumn("upper_col", udf_my_function(df.col1)).show()
```

## Performance Optimization

```python
# Cache DataFrame
df.cache()

# Persist with specific storage level
from pyspark.storagelevel import StorageLevel

df.persist(StorageLevel.MEMORY_AND_DISK)

# Repartition DataFrame
df_repartitioned = df.repartition(10)

# Coalesce to reduce partitions
df_coalesced = df.coalesce(2)
```

## Common Functions

```python
from pyspark.sql.functions import col, lit, when, concat, to_date

# Add new column with literal value
df.withColumn("new_col", lit("value")).show()

# Conditional column
df.withColumn("new_col", when(col("col1") > 100, "High").otherwise("Low")).show()

# Concatenate columns
df.withColumn("full_name", concat(col("first_name"), lit(" "), col("last_name"))).show()

# Convert to date
df.withColumn("date_col", to_date(col("string_date"), "yyyy-MM-dd")).show()
```

## Stopping SparkSession

```python
spark.stop()
```

## ML Pipeline Ops

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# Prepare data: Combine features into a single vector
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features"
)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Define a model (e.g., Logistic Regression for classification)
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")

# Create pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Fit pipeline
model = pipeline.fit(train_df)

# Make predictions
predictions = model.transform(test_df)
predictions.select("prediction", "label").show()

# Evaluate model (e.g., for classification)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Example for regression
lr_reg = LinearRegression(featuresCol="scaled_features", labelCol="target")
pipeline_reg = Pipeline(stages=[assembler, scaler, lr_reg])
model_reg = pipeline_reg.fit(train_df)
predictions_reg = model_reg.transform(test_df)
```

## Tips

- Use `explain()` to view the execution plan: `df.explain()`
- Check number of partitions: `df.rdd.getNumPartitions()`
- Use `mode="append"` for incremental writes.
- Leverage `spark.sql.shuffle.partitions` to control shuffle partitions.