---
title: Data Engineering
nav_order: 11
parent: Notes
layout: default
---


# Data Engineering

## 1. **Core Concepts**

### Data Pipeline
- **Definition**: A series of processes to extract, transform, and load (ETL/ELT) data from sources to destinations.
- **Components**:
  - **Extraction**: Pulling data from sources (databases, APIs, files).
  - **Transformation**: Cleaning, aggregating, or enriching data.
  - **Loading**: Storing data in a destination (data warehouse, database).
- **Tools**: Apache Airflow, Apache NiFi, Prefect, Dagster.

### ETL vs. ELT
- **ETL (Extract, Transform, Load)**:
  - Transform data before loading into the destination.
  - Suitable for structured data and traditional data warehouses.
  - Example: Informatica, Talend, AWS Glue.
- **ELT (Extract, Load, Transform)**:
  - Load raw data into the destination, then transform.
  - Common in cloud-based data lakes/warehouses.
  - Example: Snowflake, Databricks, Google BigQuery.
  - **Cause for ELT**:
    - **Cloud Computing**: Cost-effective storage and scalable transformation in cloud data warehouses.
    - **Massive Data Volumes**: Loading raw data first allows flexible transformation later.
    - **Data Lake Popularity**: Fits raw, unstructured data storage.

### Data Lake vs. Data Warehouse
- **Data Lake**:
  - Stores raw, unstructured, semi-structured, or structured data.
  - Schema-on-read.
  - Example: AWS S3, Azure Data Lake, Databricks Delta Lake.
- **Data Warehouse**:
  - Stores structured, processed data optimized for analytics.
  - Schema-on-write.
  - Example: Snowflake, Google BigQuery, Amazon Redshift.
- **Data Mart**:
  - Smaller, subject-oriented subset of a data warehouse (e.g., for marketing or finance).

### Data Modeling
- **Definition**: Structuring data to represent real-world entities and relationships.
- **Types**:
  - **Conceptual**: High-level, business-focused (e.g., ER diagrams).
  - **Logical**: Defines data structure without physical storage details.
  - **Physical**: Specifies how data is stored (tables, indexes).
- **Schemas**:
  - **Star Schema**: Central fact table connected to dimension tables. Optimized for analytics.
  - **Snowflake Schema**: Normalized star schema with hierarchical dimension tables.
  - **Galaxy Schema**: Multiple fact tables sharing dimension tables.
- **Tools**: dbt (Data Build Tool), ERwin, Lucidchart.

### Dimensional Modeling
- **Fact Table**:
  - Contains numerical, measurable data (e.g., sales transactions).
  - Large, immutable, links to dimensions via foreign keys.
- **Dimension Table**:
  - Provides descriptive context (e.g., customer details, product info).
  - Smaller, less frequently updated.
- **Example**:
  ### Fact Table: FactSales
  | DateKey    | ProductKey | CustomerKey | Quantity | SalesAmount |
  |------------|------------|-------------|----------|-------------|
  | 2025-09-21 | 101        | 201         | 2        | 1500        |
  ### Dimension Table: DimProduct
  | ProductKey | Name     | Category | ListPrice |
  |------------|----------|----------|-----------|
  | 101        | Widget A | Gadget   | 800       |
  ### Dimension Table: DimCustomer
  | CustomerKey | Name  | AgeGroup | City  |
  |-------------|-------|----------|-------|
  | 201         | Alice | 25–34    | Delhi |

## 2. **Data Storage**

### Relational Databases
- **Definition**: Structured data stored in tables with rows and columns.
- **Key Features**: ACID compliance, SQL support, schema enforcement.
- **Examples**: PostgreSQL, MySQL, Oracle, SQL Server.
- **Normalization**: Reduces redundancy, suited for OLTP systems.
- **Denormalization**: Improves query performance for analytics (OLAP systems).

### NoSQL Databases
- **Types**:
  - **Key-Value**: Simple key-value pairs (e.g., Redis, DynamoDB).
  - **Document**: JSON/BSON documents (e.g., MongoDB, CouchDB).
  - **Column-Family**: Column-based storage (e.g., Cassandra, HBase).
  - **Graph**: Nodes and edges for relationships (e.g., Neo4j, ArangoDB).
- **Use Case**: High scalability, flexible schemas, big data.

### Data Lakes
- **Definition**: Centralized repository for raw data in native format.
- **Storage Formats**: Parquet, Avro, ORC (optimized for columnar storage).
- **Use Case**: Big data analytics, machine learning.
- **Examples**: AWS S3 + Lake Formation, Azure Data Lake, Google Cloud Storage, Hadoop HDFS.

### Data Warehouses
- **Definition**: Optimized for querying and reporting on structured data.
- **Key Features**: High-performance SQL, indexing, partitioning.
- **Examples**: Snowflake, BigQuery, Redshift.

## 3. **Data Ingestion**

### Batch Processing
- **Definition**: Processing data in large, scheduled chunks.
- **Use Case**: Daily/weekly reports, historical data analysis.
- **Tools**: Apache Spark, Hadoop MapReduce, AWS Glue.

### Stream Processing
- **Definition**: Processing data in real-time as it arrives.
- **Use Case**: Fraud detection, IoT, real-time analytics.
- **Tools**: Apache Kafka, Apache Flink, Apache Storm, AWS Kinesis.

### Data Integration
- **Definition**: Combining data from multiple sources into a unified view.
- **Methods**:
  - **APIs**: REST, GraphQL for real-time data.
  - **File Transfers**: CSV, JSON, Parquet files via SFTP/FTPS.
  - **Database Connectors**: JDBC/ODBC for relational databases.
- **Tools**: Apache NiFi, Talend, Fivetran, Stitch.

### Extraction Methods
- **Full Extraction**: Copies entire dataset each time (simple but resource-intensive).
- **Incremental Extraction**: Extracts only changed/added data using timestamps or changelogs.
- **Change Data Capture (CDC)**: Uses transaction logs for efficient incremental extraction.
- **Batch vs. Real-Time**:
  - **Batch**: Scheduled extractions (e.g., daily), suitable for non-urgent tasks.
  - **Real-Time/Streaming**: Continuous extraction for immediate insights (e.g., fraud detection).

## 4. **Data Transformation**

### Data Staging Area
- **Definition**: Temporary storage for data post-extraction, pre-transformation.
- **Purpose**: Prevents impact on source systems, allows safe manipulation.
- **Best Practice**: Transient storage, data deleted after successful load.

### Core Transformations
- **Data Cleaning**: Remove duplicates, handle missing values, standardize formats.
- **Data Validation**: Apply business rules to ensure data integrity.
- **Data Enrichment**: Add context (e.g., geolocation, external APIs).
- **Data Aggregation**: Summarize data (e.g., sum, average, count).
- **Data Type Conversions/Field Mapping**: Ensure compatibility with target schema.
- **Normalization/Denormalization**:
  - **Normalization**: Reduce redundancy in relational databases.
  - **Denormalization**: Combine tables for faster analytics queries.

### Advanced Transformations
- **Slowly Changing Dimensions (SCD)**:
  - **Type 1**: Overwrite old values, no history preserved.
  - **Type 2**: Create new records for changes, preserve history.

### Tools
- **Data Cleaning**: Pandas (Python), Spark SQL, OpenRefine.
- **Data Enrichment/Aggregation**: Python (e.g., requests), dbt, SQL.

## 5. **Data Loading**

### Loading Strategies
- **Full Load**: Replace all data in target (simple but slow for large datasets).
- **Incremental Load**: Load only new/changed data (efficient, requires change tracking).
- **Bulk Loading**: Fast, optimized for high-volume data.
- **Row-by-Row Loading**: Slow, used for small datasets.
- **Data Integrity**: Primary key checks, constraint validation to prevent duplicates and ensure schema compliance.

### Target Systems
- Dictate loading method and final data structure (e.g., data warehouse, data lake).

## 6. **Data Orchestration**

### Workflow Orchestration
- **Definition**: Automating and scheduling data pipeline tasks.
- **Key Features**: Dependency management, retries, monitoring.
- **Tools**:
  - **Apache Airflow**: DAG-based, Python-based workflows.
  - **Prefect**: Modern, Python-first orchestration.
  - **Dagster**: Data-aware with asset management.
  - **Luigi**: Lightweight, Python-based.
  - **Cloud-Native**: AWS Step Functions.

### Scheduling
- **Cron Jobs**: Time-based (e.g., `0 0 * * *` for daily at midnight).
- **Event-Driven**: Triggered by events (e.g., file arrival, API call).
- **Tools**: Airflow Scheduler, AWS EventBridge, Kubernetes CronJobs.

### Handling Dependencies
- Orchestrators manage job dependencies to ensure correct execution order.

## 7. **Data Governance**

### Data Quality
- **Aspects**:
  - **Accuracy**: Correctness of data.
  - **Completeness**: No missing values.
  - **Consistency**: Uniformity across systems.
  - **Timeliness**: Data availability when needed.
- **Tools**: Great Expectations, Soda, Apache Griffin.
- **Checks**: Applied during transformation phase for integrity.

### Data Lineage
- **Definition**: Tracking data flow from source to destination.
- **Use Case**: Debugging, compliance, impact analysis.
- **Tools**: Apache Atlas, DataHub, OpenLineage.

### Data Catalog
- **Definition**: Metadata repository for discovering and managing data assets.
- **Examples**: AWS Glue Data Catalog, Alation, Collibra.

### Data Security
- **Techniques**:
  - **Encryption**: AES-256 (at rest), TLS (in transit).
  - **Access Control**: RBAC, ABAC.
  - **Data Masking**: Obfuscate sensitive data.
- **Tools**: AWS IAM, Apache Ranger, HashiCorp Vault.

### Auditing and Compliance
- Track data access/changes for regulations (e.g., GDPR, HIPAA).
- Tools provide logging and access control features.

## 8. **Big Data Technologies**

### Distributed Computing
- **Hadoop**: Distributed storage (HDFS) and processing (MapReduce).
- **Spark**: In-memory processing for batch and streaming.
- **Flink**: Low-latency stream processing.

### Data Formats
- **Parquet**: Columnar storage, optimized for analytics.
- **Avro**: Row-based, schema evolution support.
- **ORC**: Optimized Row Columnar, high compression.

### Message Queues
- **Definition**: Systems for asynchronous data transfer.
- **Examples**: Apache Kafka, RabbitMQ, AWS SQS.

## 9. **Cloud Data Engineering**

### Cloud Providers
- **AWS**: S3, Redshift, Glue, Athena, Kinesis.
- **Azure**: Data Lake, Synapse Analytics, Data Factory.
- **GCP**: BigQuery, Dataflow, Dataproc, Pub/Sub.

### Serverless Data Processing
- **Definition**: Running pipelines without managing servers.
- **Examples**: AWS Lambda, Google Cloud Functions, Azure Functions.

### Managed Services
- **Data Warehouses**: Snowflake, BigQuery, Redshift.
- **ETL Tools**: AWS Glue, Azure Data Factory, Google Dataform.
- **Streaming**: AWS Kinesis, Azure Event Hubs, Google Pub/Sub.

## 10. **Monitoring and Logging**

### Monitoring
- **Metrics**: Latency, throughput, error rates.
- **Tools**: Prometheus, Grafana, Datadog.

### Logging
- **Purpose**: Debugging, auditing, performance tracking.
- **Tools**: ELK Stack (Elasticsearch, Logstash, Kibana), AWS CloudWatch.

### Alerting
- **Definition**: Notifying teams of pipeline failures or anomalies.
- **Tools**: PagerDuty, Slack integrations, AWS SNS.

## 11. **Performance Optimization**

### Tuning ETL Jobs
- Optimize code/processes (e.g., bulk loading vs. row-by-row).
- Use partitioning/indexing to improve query performance.

### Partitioning and Indexing
- **Partitioning**: Divide large tables for faster queries.
- **Indexing**: Speed up data retrieval.

## 12. **Best Practices**

- **Modularity**: Break pipelines into reusable components.
- **Idempotency**: Ensure repeated operations produce the same result.
- **Version Control**: Use Git for code, dbt for data models.
- **Testing**: Validate data quality, pipeline logic, and performance.
- **Documentation**: Maintain clear pipeline and schema documentation.
- **Scalability**: Design for increasing data volumes and complexity.
- **Data Lineage/Traceability**: Record data lifecycle for auditing and debugging.
- **Failure Recovery**: Plan for job restarts or retries after failures.