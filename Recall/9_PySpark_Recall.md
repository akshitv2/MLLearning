---
title: PySpark
nav_order: 9
parent: Recall
layout: default
---

# PySpark Recall Questions

This Markdown file contains questions designed to strengthen recall of key concepts from the PySpark documentation. Each question includes a clickable "Goto" link to the relevant section in the original document for reference.

## Core Concepts Questions

1. **Question:** What is the role of the SparkSession in PySpark?[Goto](../Notes/9_PySpark.md#core-architecture)
2. **Question:** Describe the function of the Driver in Spark's core architecture.[Goto](../Notes/9_PySpark.md#core-architecture)
3. **Question:** What does DAG stand for in Spark, and what is its purpose?[Goto](../Notes/9_PySpark.md#core-architecture)
4. **Question:** Explain the Catalyst Optimizer and its role in Spark.[Goto](../Notes/9_PySpark.md#core-architecture)
5. **Question:** What are the key characteristics of an RDD (Resilient Distributed Dataset)?[Goto](../Notes/9_PySpark.md#rdd-resilient-distributed-dataset)
6. **Question:** How does lazy evaluation work in RDDs?[Goto](../Notes/9_PySpark.md#rdd-resilient-distributed-dataset)
7. **Question:** What makes DataFrames different from RDDs in terms of structure and optimization?[Goto](../Notes/9_PySpark.md#dataframe)
8. **Question:** When should you use RDDs instead of DataFrames?[Goto](../Notes/9_PySpark.md#rdd-vs-dataframe)
9. **Question:** How does Spark achieve fault tolerance through lineage?[Goto](../Notes/9_PySpark.md#fault-tolerance)
10. **Question:** What is checkpointing in Spark, and when is it useful?[Goto](../Notes/9_PySpark.md#fault-tolerance)
11. **Question:** Name some supported data formats in Spark for reading and writing.[Goto](../Notes/9_PySpark.md#data-sources-and-formats)
12. **Question:** What are the available write modes in Spark, and what does each do?[Goto](../Notes/9_PySpark.md#data-sources-and-formats)
13. **Question:** Why is Parquet a preferred format for big data processing in Spark?[Goto](../Notes/9_PySpark.md#data-sources-and-formats)
14. **Question:** What does the collect() action do, and what are its risks?[Goto](../Notes/9_PySpark.md#operations-and-functions)
15. **Question:** Explain the difference between cache() and persist() in Spark.[Goto](../Notes/9_PySpark.md#operations-and-functions)
16. **Question:** What are User-Defined Functions (UDFs), and why might they be slower than built-in functions?[Goto](../Notes/9_PySpark.md#operations-and-functions)
17. **Question:** Name the types of joins supported in Spark and describe a broadcast join.[Goto](../Notes/9_PySpark.md#joins)
18. **Question:** How can you optimize performance using partitioning in Spark?[Goto](../Notes/9_PySpark.md#performance-optimization)
19. **Question:** What is data skew, and how can it be handled?[Goto](../Notes/9_PySpark.md#performance-optimization)
20. **Question:** Explain predicate pushdown and its benefits.[Goto](../Notes/9_PySpark.md#predicate-pushdown)
21. **Question:** What are the steps in a typical ETL process using PySpark?[Goto](../Notes/9_PySpark.md#etl-extract-transform-load)
22. **Question:** What is GraphFrames, and what algorithms does it support?[Goto](../Notes/9_PySpark.md#graph-processing)
23. **Question:** Describe Structured Streaming in Spark.[Goto](../Notes/9_PySpark.md#streaming)
24. **Question:** How do you save and load a trained ML model in Spark MLlib?[Goto](../Notes/9_PySpark.md#model-persistence)
25. **Question:** List some best practices for working with PySpark.[Goto](../Notes/9_PySpark.md#best-practices)
26. **Question:** What are the different storage levels available for persistence in Spark?[Goto](../Notes/9_PySpark.md#storage-levels)
## PySpark Cheat Sheet Questions
27. **Question:** How do you initialize a SparkSession in PySpark?[Goto](../Notes/9_PySpark.md#initializing-pyspark)
28. **Question:** What code would you use to read a CSV file into a DataFrame?[Goto](../Notes/9_PySpark.md#reading-data)
29. **Question:** How can you create a DataFrame from a Python list?[Goto](../Notes/9_PySpark.md#loading-into-rddsdataframes)
30. **Question:** What is the syntax for writing a DataFrame to Parquet format?[Goto](../Notes/9_PySpark.md#writing-data)
31. **Question:** How do you select specific columns from a DataFrame and display them?[Goto](../Notes/9_PySpark.md#dataframe-operations)
32. **Question:** What method would you use to group by a column and aggregate another?[Goto](../Notes/9_PySpark.md#dataframe-operations)
33. **Question:** Explain the differences between collect(), count(), and take() in DataFrames.[Goto](../Notes/9_PySpark.md#collect-count-and-take)
34. **Question:** How do you perform an inner join between two DataFrames?[Goto](../Notes/9_PySpark.md#joins-1)
35. **Question:** What code drops rows with null values in a DataFrame?[Goto](../Notes/9_PySpark.md#handling-missing-values)
36. **Question:** How do you register a DataFrame as a temporary view for SQL queries?[Goto](../Notes/9_PySpark.md#sql-queries)
37. **Question:** What is a window specification in Spark, and how is it used with row_number()?[Goto](../Notes/9_PySpark.md#window-functions)
38. **Question:** Provide an example of defining and applying a UDF to a DataFrame column.[Goto](../Notes/9_PySpark.md#udf-user-defined-functions)
39. **Question:** How do you cache a DataFrame for performance?[Goto](../Notes/9_PySpark.md#performance-optimization-1)
40. **Question:** What function adds a new column with a literal value to a DataFrame?[Goto](../Notes/9_PySpark.md#common-functions)
41. **Question:** Describe the steps to build an ML pipeline in PySpark for classification.[Goto](../Notes/9_PySpark.md#ml-pipeline-ops)
42. **Question:** How do you stop a SparkSession?[Goto](../Notes/9_PySpark.md#stopping-sparksession)
43. **Question:** What does df.explain() do?[Goto](../Notes/9_PySpark.md#tips)