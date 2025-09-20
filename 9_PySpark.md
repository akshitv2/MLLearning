# 9 PySpark

- Python API for Apache Spark

## Core Concepts

### Core Architecture
- **SparkSession**: Entry point for DataFrame and SQL APIs. It provides a unified interface to access Spark's functionality.
- **Cluster Manager**: Manages resources across the cluster (📌e.g., YARN, Mesos, Kubernetes, Standalone).
- **Driver**: Runs the main function, creates SparkContext, and coordinates tasks by submitting them to executors.
- **Executor**: Runs tasks on worker nodes, stores data in memory or disk, and reports results back to the driver.
- **DAG (Directed Acyclic Graph)**: Represents computation stages as a graph of transformations and actions; optimized by the Catalyst Optimizer.
- **Catalyst Optimizer**: Query optimizer for DataFrame and SQL operations, which analyzes and improves execution plans through rule-based and cost-based optimizations.
- **Tungsten Execution Engine**: Low-level engine that optimizes memory management and code generation for faster performance.

### RDD (Resilient Distributed Dataset)
- Distributed data structure that represents data in Spark's memory across clusters for parallel computation.
- Immutable: Once created, it cannot be changed.
- Transformations are added to a lineage graph instead of tracking changes directly.
- Lazy Evaluation: Computations are not executed immediately; evaluation occurs only when an action is called.
- Fault Tolerant: Can recompute lost partitions using lineage.
- Supports two types of operations: Transformations (📌e.g., map, filter) and Actions (📌e.g., collect, count).

### DataFrame
- Higher-level abstraction built on top of RDDs.
- Still immutable and follows all RDD properties (📌e.g., lazy evaluation, fault tolerance).
- Has a tabular structure with named columns and schema.
- Lazy Evaluation: Transformations (📌e.g., filter, groupBy) are not executed until an action (📌e.g., show, collect) is called.
- Optimized using Catalyst Optimizer and supports SQL-like queries.

### RDD vs. DataFrame
- **RDD**: Low-level, functional API; requires manual optimization and is more flexible for unstructured data or custom operations.
- **DataFrame**: Higher-level, structured API; leverages Catalyst Optimizer for automatic performance improvements and is easier for SQL-like operations.
- **When to Use RDD**: For complex custom operations not supported by DataFrame or SQL APIs, or when working with unstructured data requiring fine-grained control.

### Fault Tolerance
- **Lineage**: Tracks transformations applied to the data, allowing Spark to rebuild lost partitions by recomputing from the original data source.
- **Checkpointing**: Saves intermediate data to disk for long-running jobs to break lineage and avoid recomputation (`df.checkpoint(eager=True)` for immediate checkpointing).
- **Persistence**: Caches data in memory or disk to prevent recomputation on failures (see Storage Levels below).

### Data Sources and Formats
- **Supported Formats**: CSV, JSON, Parquet, ORC, Avro, JDBC/ODBC (for databases), Delta Lake, Text, Hive tables.
- **Reading Data**: Use `spark.read.format("format").option("key", "value").load("path")` for flexible loading.
- **Writing Data**: Use `df.write.format("format").mode("mode").save("path")`.
- **Write Modes**: `overwrite` (replace existing data), `append` (add to existing), `ignore` (skip if exists), `error` (throw error if exists).
- **Parquet Format**:
  - Columnar storage file format optimized for big data processing.
  - Supports efficient data storage and retrieval with schema evolution.
  - Compression options: Snappy (fast, moderate compression), Gzip (higher compression, slower).
  - Stores data column-wise (vs. row-wise in formats like CSV), enabling predicate pushdown and skipping irrelevant columns.
  - Example Benefit: Querying only "age" and "salary" skips reading "name" column, reducing I/O and allowing row filtering before loading into memory.

### Operations and Functions
- **Collect()**: Triggers computation of the lineage graph and brings all data to the driver node as a list. Can cause driver OOM if data is too large; suitable only for development or small datasets. Alternatives: `.take(n)` (first n rows), `.show(n)` (pretty-print n rows).
- **Cache()**: Marks a dataset for caching on first computation. Shorthand for `.persist(StorageLevel.MEMORY_AND_DISK)`. Stores in cluster memory, spills to disk if too large. Can cause OOM for heavy objects.
- **Unpersist()**: Removes a dataset from cache to free up resources.
- **User-Defined Functions (UDF)**:
  - Custom Python functions registered in Spark for applying logic to DataFrame columns.
  - Performance Note: UDFs are slower than built-in Spark SQL functions due to serialization and Python-JVM overhead (10-100x slower than native functions in some cases).
  - Recommendation: Prefer built-in functions; use UDFs only when necessary. Pandas UDFs (vectorized) can be 10-100x faster than standard UDFs for batch operations.

### Joins
- Types: Inner, Left, Right, Full Outer, Cross (Cartesian product; dangerous as it can explode row count).
- **Broadcast Join**: Optimization for joining a large DataFrame with a small one. Broadcasts the small DataFrame to all worker nodes, avoiding shuffle of the large one. Reduces I/O but ensure the small dataset fits in each node's memory.
- Usage Hint: `from pyspark.sql.functions import broadcast; large_df.join(broadcast(small_df), on="key")`.

### Performance Optimization
- **Caching/Persisting**: Store DataFrames in memory or disk to avoid recomputation (`df.cache()` or `df.persist(StorageLevel.MEMORY_AND_DISK)`).
- **Partitioning**:
  - **Repartition**: Increases or decreases partitions with a full shuffle (expensive). Example: `df.repartition(8)` (from 4 to 8 partitions).
  - **Coalesce**: Reduces partitions without shuffling (merges existing ones). Cannot increase partitions effectively. Example: `df.coalesce(4)` (from 8 to 4 partitions). Useful before writing to disk to reduce small files.
- **Data Skew**: Uneven data distribution across partitions leading to slow tasks. Handle with salting (add random suffix to keys), custom partitioning, or AQE.
- **Shuffle Tuning**: Adjust `spark.sql.shuffle.partitions` (default 200) to control tasks during shuffles (📌e.g., joins, groupBy).
- **Broadcast Join**: As above, for small-large joins.
- **Adaptive Query Execution (AQE)**: Automatically optimizes query plans at runtime (enabled by default in Spark 3.0+; handles skew, coalesces partitions dynamically).
- **Skew Handling**: Beyond salting, use `spark.sql.adaptive.skewJoin.enabled=true` for AQE to split skewed partitions.

### Predicate Pushdown
- Optimization technique where filters (predicates) are pushed down to the data source (📌e.g., Parquet, JDBC).
- Reduces data loaded into Spark by filtering at the source level, minimizing I/O and network transfer.
- Enabled by default for supported formats; use `spark.read.option("pushDownPredicate", true)` if needed.
- Example: In Parquet, filters like `age > 30` are applied before reading, skipping irrelevant row groups.

### ETL (Extract, Transform, Load)
- **Extract**: Load data from sources using `spark.read` (📌e.g., databases, files).
- **Transform**: Apply operations like filtering, aggregating, joining, or UDFs on DataFrames.
- **Load**: Write transformed data to sinks using `df.write` (📌e.g., to HDFS, S3, databases).
- Best for batch processing; use Structured Streaming for real-time ETL.

### Graph Processing
- **GraphFrames**: A DataFrame-based library for graph processing (install via `--packages graphframes:graphframes:0.8.2-spark3.0-s_2.12`).
- Builds graphs from vertex and edge DataFrames.
- Supports algorithms: PageRank, Connected Components, Label Propagation, Shortest Paths, Triangle Count.
- Example: Create a graph with `g = GraphFrame(vertices, edges)`, then `g.pageRank(resetProbability=0.15, tol=0.01)`.

### Streaming
- **Structured Streaming**: Spark's API for processing streaming data as unbounded tables.
- Handles late data, watermarks for event-time processing, and fault-tolerant exactly-once semantics.
- Sources: Kafka, Files, Sockets; Sinks: Console, Files, Kafka, Foreach.
- Example: Read stream `df = spark.readStream.format("kafka").load()`, apply transformations, write `query = df.writeStream.outputMode("append").format("console").start()`.
- Supports windowed aggregations, joins with static or streaming data.

### Model Persistence
- In Spark MLlib, save trained models to disk for reuse.
- Use `model.save("path/to/model")` to persist (supports formats like Parquet for pipelines).
- Load with `loaded_model = PipelineModel.load("path/to/model")` or specific model class (📌e.g., `LogisticRegressionModel.load()`).
- Enables deployment, versioning, and sharing models across jobs.

### Best Practices
- Prefer DataFrame/SQL over RDDs for better optimization and readability.
- Minimize shuffles (📌e.g., avoid unnecessary joins, groupBy; use broadcast for small tables).
- Cache strategically for reused datasets, but unpersist when done.
- Use columnar formats like Parquet for storage efficiency and predicate pushdown.
- Monitor partitions: Aim for 100-200MB per partition; tune with repartition/coalesce.
- Enable AQE and monitor via Spark UI for bottlenecks.
- Avoid collect() on large data; use sampling or aggregations.
- Test on small data before scaling.

### MLlib (Machine Learning Library)
- **Overview**: Spark MLlib is Apache Spark's scalable machine learning library, designed for distributed data processing. It provides tools for classification, regression, clustering, recommendation, feature engineering, and pipeline construction, optimized for large-scale datasets.
- **Key Components**:
  - **Data Preparation**:
    - **VectorAssembler**: Combines multiple columns into a single feature vector.
    - **StandardScaler**, **MinMaxScaler**: Normalizes or scales features for consistent ranges.
    - **StringIndexer**, **OneHotEncoder**: Converts categorical variables into numerical representations.
    - **Tokenizer**, **CountVectorizer**, **TF-IDF**: Processes text data for natural language processing tasks.
  - **Algorithms**:
    - Classification: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, NaiveBayes, GBTClassifier (Gradient-Boosted Trees).
    - Regression: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor.
    - Clustering: KMeans, BisectingKMeans, GaussianMixture, LDA (Latent Dirichlet Allocation).
    - Recommendation: ALS (Alternating Least Squares) for collaborative filtering.
    - Frequent Pattern Mining: FP-Growth, PrefixSpan.
  - **Pipelines**:
    - Combines data preprocessing, feature engineering, and model training into a single workflow.
    - Ensures consistency between training and testing phases.
    - Example: Chain `VectorAssembler`, `StandardScaler`, and `LogisticRegression` in a `Pipeline`.
  - **Model Evaluation**:
    - Evaluators like `MulticlassClassificationEvaluator`, `RegressionEvaluator`, `ClusteringEvaluator`.
    - Metrics: Accuracy, precision, recall, F1-score, RMSE, MAE.
  - **Hyperparameter Tuning**:
    - Use `CrossValidator` or `TrainValidationSplit` with `ParamGridBuilder` to search for optimal model parameters.
    - Example: Tune `maxDepth` and `numTrees` for a `RandomForestClassifier`.
  - **Distributed Processing**: MLlib leverages Spark's distributed architecture, enabling parallel training and prediction on large datasets across clusters.
  - **Interoperability**: Supports integration with Python libraries like NumPy and Pandas (via Pandas UDFs) for advanced analytics.
- **Best Practices**:
  - Use `Pipeline` to streamline workflows and avoid manual staging errors.
  - Cache intermediate datasets (📌e.g., feature vectors) to speed up iterative algorithms.
  - Handle missing values and outliers before training (use DataFrame operations like `na.drop()` or `na.fill()`).
  - Optimize hyperparameters using cross-validation for robust models.
  - Test models on a small dataset before scaling to ensure correctness.
  - Monitor model performance using Spark UI to detect bottlenecks in training.

### Storage Levels
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

## PySpark Cheat Sheet

### Initializing PySpark
```python
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder.appName("MyApp").getOrCreate()
```

### Reading Config
```python
# Get configuration
spark.sparkContext.getConf()
```

### Reading Data
```python
# Read CSV
df = spark.read.csv("file.csv", header=True, inferSchema=True)

# Read JSON
df = spark.read.json("file.json")

# Read Parquet
df = spark.read.parquet("file.parquet")

# Read Text File
df = spark.read.text("file.txt")
```

### Loading Into RDDs/DataFrames
```python
# Create DataFrame from list
spark.createDataFrame(data, ["colAName", ...])

# Create RDD from Python list
spark.sparkContext.parallelize(someList)
```

### Writing Data
```python
# Write to CSV
df.write.csv("output.csv", header=True, mode="overwrite")

# Write to Parquet
df.write.parquet("output.parquet", mode="overwrite")

# Write to JSON
df.write.json("output.json", mode="overwrite")
```

### DataFrame Operations
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

### Collect, Count, and Take
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

### Joins
```python
# Inner join
df_joined = df1.join(df2, ["key"], "inner")

# Left join
df_joined = df1.join(df2, ["key"], "left")

# Right join
df_joined = df1.join(df2, ["key"], "right")

# Full outer join
df_joined = df1.join(df2, ["key"], "outer")

# Cross Join (dangerous, can explode rows)
df_joined = df1.crossJoin(df2)

# Broadcast Join
from pyspark.sql.functions import broadcast
joined_df = large_df.join(broadcast(small_df), on="id", how="inner")
```

### Handling Missing Values
```python
# Drop rows with nulls
df.na.drop().show()

# Fill nulls with a value
df.na.fill(0).show()

# Fill nulls in specific columns
df.na.fill({"col1": 0, "col2": "unknown"}).show()
```

### SQL Queries
```python
# Register DataFrame as a temporary view
df.createOrReplaceTempView("table_name")

# Run SQL query
result = spark.sql("SELECT col1, COUNT(*) FROM table_name GROUP BY col1")
result.show()
```

### Window Functions
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

### UDF (User-Defined Functions)
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType

# Define UDF
def square(x):
    if x is not None:
        return x * x
square_udf = udf(square, IntegerType())

# Apply UDF
df.withColumn("squared_col", square_udf(df.col1)).show()

# Another example
def my_function(x):
    return x.upper() if x else x
udf_my_function = udf(my_function, StringType())
df.withColumn("upper_col", udf_my_function(df.col1)).show()
```

### Performance Optimization
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

### Common Functions
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

### ML Pipeline Ops
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

# Define a model (📌e.g., Logistic Regression for classification)
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

# Evaluate model (📌e.g., for classification)
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

### Stopping SparkSession
```python
spark.stop()
```

### Tips
- Use `explain()` to view the execution plan: `df.explain()`
- Check number of partitions: `df.rdd.getNumPartitions()`
- Use `mode="append"` for incremental writes.
- Leverage `spark.sql.shuffle.partitions` to control shuffle partitions.