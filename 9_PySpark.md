# 9 PySpark

- Python Api For Apache Spark
- ## RDD: Resilient Distributed Dataset
    - Distributed Data structure
    - Represents Data in Spark's memory across clusters for parallel computation
    - Immutable : Doesn't change once created
    - Transactions are added to lineage graph instead of track changes
    - Lazy Evaluation -> Doesn't execute Immediately (evaluation is done when neccessary)
    - Fault Tolerant
- ## DataFrame:
    - Higher Level than RDD
    - Still Immutable (all things about RDD still apply)
    - Has tabular structure
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
spark.createDataFrame(data,["colAName",...])

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
df.show(5)  # Display first 5 rows
df.printSchema()  # Display schema

# Select columns
df.select("col1", "col2").show()

# Filter rows
df.filter(df.col1 > 100).show()

# Group by and aggregate
df.groupBy("col1").agg({"col2": "sum"}).show()

# Sort
df.orderBy("col1", ascending=False).show()

# Drop duplicates
df.dropDuplicates(["col1"]).show()

# Drop columns
df.drop("col1").show()
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

## Tips
- Use `explain()` to view the execution plan: `df.explain()`
- Check number of partitions: `df.rdd.getNumPartitions()`
- Use `mode="append"` for incremental writes.
- Leverage `spark.sql.shuffle.partitions` to control shuffle partitions.