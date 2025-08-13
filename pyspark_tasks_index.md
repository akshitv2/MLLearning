| Tier                                                  |   Task No. | Description                                                                  |
|:------------------------------------------------------|-----------:|:-----------------------------------------------------------------------------|
| Tier 1 – Beginner (Fundamentals)                      |       1.1  | Install PySpark locally or in a notebook environment                         |
| Tier 1 – Beginner (Fundamentals)                      |       1.2  | Initialize a SparkSession                                                    |
| Tier 1 – Beginner (Fundamentals)                      |       1.3  | Check Spark version & configuration                                          |
| Tier 1 – Beginner (Fundamentals)                      |       1.4  | Create RDDs from lists or text files                                         |
| Tier 1 – Beginner (Fundamentals)                      |       1.5  | `map()`, `filter()`, `flatMap()` transformations                             |
| Tier 1 – Beginner (Fundamentals)                      |       1.6  | `collect()`, `count()`, `take()` actions                                     |
| Tier 1 – Beginner (Fundamentals)                      |       1.7  | Create DataFrame from Python dictionary/list                                 |
| Tier 1 – Beginner (Fundamentals)                      |       1.8  | Create DataFrame from CSV/JSON/Parquet                                       |
| Tier 1 – Beginner (Fundamentals)                      |       1.9  | Show schema and data (`printSchema()`, `show()`)                             |
| Tier 1 – Beginner (Fundamentals)                      |       1.1  | Select specific columns                                                      |
| Tier 1 – Beginner (Fundamentals)                      |       1.11 | Filter rows using conditions                                                 |
| Tier 1 – Beginner (Fundamentals)                      |       1.12 | Rename columns                                                               |
| Tier 1 – Beginner (Fundamentals)                      |       1.13 | Add new columns (`withColumn`)                                               |
| Tier 1 – Beginner (Fundamentals)                      |       1.14 | Drop columns                                                                 |
| Tier 1 – Beginner (Fundamentals)                      |       1.15 | Register DataFrame as a SQL temporary view                                   |
| Tier 1 – Beginner (Fundamentals)                      |       1.16 | Run simple `SELECT` queries with `spark.sql()`                               |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.1  | String functions (`concat`, `upper`, `lower`, `trim`)                        |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.2  | Date/time functions (`current_date`, `datediff`, `date_format`)              |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.3  | Conditional logic with `when()` and `otherwise()`                            |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.4  | `groupBy()` with aggregation functions (`count`, `avg`, `sum`, `max`, `min`) |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.5  | Multiple aggregations in one statement                                       |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.6  | `orderBy()` ascending/descending                                             |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.7  | Multi-column ordering                                                        |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.8  | Inner, left, right, full joins                                               |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.9  | Semi and anti joins                                                          |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.1  | Self joins                                                                   |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.11 | Handling nulls (`fillna`, `dropna`, `na.replace`)                            |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.12 | Replace specific values in a column                                          |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.13 | Cache and uncache data                                                       |
| Tier 2 – Intermediate (Data Wrangling & Aggregations) |       2.14 | Explain storage levels                                                       |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.1  | Ranking functions (`rank`, `dense_rank`, `row_number`)                       |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.2  | Aggregations over a window (`lead`, `lag`, `sum over partition`)             |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.3  | Broadcast joins for small datasets                                           |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.4  | Cross joins                                                                  |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.5  | Create and register Python UDFs                                              |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.6  | Use Pandas UDFs for performance                                              |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.7  | Explode arrays and maps                                                      |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.8  | Access nested fields in structs                                              |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.9  | Read from multiple file formats (CSV, JSON, Parquet, ORC, Avro)              |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.1  | Write data with partitioning & bucketing                                     |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.11 | `repartition()` vs `coalesce()`                                              |
| Tier 3 – Advanced (Transformations & Optimizations)   |       3.12 | Skew handling                                                                |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.1  | Use of `explain()` and Catalyst optimizer                                    |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.2  | Predicate pushdown                                                           |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.3  | File format selection for efficiency                                         |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.4  | Broadcast Variables & Accumulators                                           |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.5  | Structured Streaming basics                                                  |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.6  | Reading from Kafka                                                           |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.7  | Writing streaming output to console/file                                     |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.8  | Time-based windowing for streaming                                           |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.9  | Complex ordering/partitioning                                                |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.1  | Data preprocessing (`VectorAssembler`, `StringIndexer`, `OneHotEncoder`)     |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.11 | Train/test split                                                             |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.12 | Train regression/classification models                                       |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.13 | Model evaluation & saving                                                    |
| Tier 4 – Expert (Performance, Streaming, ML)          |       4.14 | Using GraphFrames for network analysis                                       |
| Tier 5 – Production Mastery                           |       5.1  | Submit jobs with `spark-submit`                                              |
| Tier 5 – Production Mastery                           |       5.2  | Configuring executors, memory, and cores                                     |
| Tier 5 – Production Mastery                           |       5.3  | Partition Pruning                                                            |
| Tier 5 – Production Mastery                           |       5.4  | Use `checkpoint()` for streaming & iterative jobs                            |
| Tier 5 – Production Mastery                           |       5.5  | Connecting to JDBC sources (PostgreSQL, MySQL)                               |
| Tier 5 – Production Mastery                           |       5.6  | Reading from and writing to AWS S3, Azure Blob, GCS                          |
| Tier 5 – Production Mastery                           |       5.7  | Upserts & merge operations with Delta Lake                                   |
| Tier 5 – Production Mastery                           |       5.8  | Time travel queries with Delta Lake                                          |
| Tier 5 – Production Mastery                           |       5.9  | Airflow + Spark integration                                                  |
| Tier 5 – Production Mastery                           |       5.1  | Using Spark on Kubernetes or YARN                                            |
| Tier 5 – Production Mastery                           |       5.11 | Spark UI for job diagnostics                                                 |
| Tier 5 – Production Mastery                           |       5.12 | Event logs & metrics collection                                              |