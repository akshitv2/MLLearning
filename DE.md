# Data Engineering Concepts Cheatsheet

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
  - Example: Informatica, Talend.
- **ELT (Extract, Load, Transform)**:
  - Load raw data into the destination, then transform.
  - Common in cloud-based data lakes/warehouses.
  - Example: Snowflake, Databricks.

### Data Lake vs. Data Warehouse
- **Data Lake**:
  - Stores raw, unstructured, semi-structured, or structured data.
  - Schema-on-read.
  - Example: AWS S3, Azure Data Lake, Databricks Delta Lake.
- **Data Warehouse**:
  - Stores structured, processed data optimized for analytics.
  - Schema-on-write.
  - Example: Snowflake, Google BigQuery, Amazon Redshift.

### Data Modeling
- **Definition**: Structuring data to represent real-world entities and relationships.
- **Types**:
  - **Conceptual**: High-level, business-focused (e.g., ER diagrams).
  - **Logical**: Defines data structure without physical storage details.
  - **Physical**: Specifies how data is stored (tables, indexes).
- **Schemas**:
  - **Star Schema**: Central fact table connected to dimension tables. Optimized for analytics.
  - **Snowflake Schema**: Normalized version of star schema with hierarchical dimension tables.
- **Tools**: dbt (Data Build Tool), ERwin, Lucidchart.

## 2. **Data Storage**

### Relational Databases
- **Definition**: Structured data stored in tables with rows and columns.
- **Key Features**: ACID compliance, SQL support, schema enforcement.
- **Examples**: PostgreSQL, MySQL, Oracle, SQL Server.

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

## 4. **Data Transformation**

### Data Cleaning
- **Tasks**: Removing duplicates, handling missing values, standardizing formats.
- **Tools**: Pandas (Python), Spark SQL, OpenRefine.

### Data Enrichment
- **Definition**: Enhancing data with additional context (e.g., geolocation, external APIs).
- **Tools**: Python libraries (e.g., requests), dbt, SQL.

### Data Aggregation
- **Definition**: Summarizing data (e.g., sum, average, count).
- **Tools**: SQL, Spark, Pandas.

### Data Normalization/Denormalization
- **Normalization**: Reducing redundancy in relational databases.
- **Denormalization**: Combining tables for faster querying in analytics.

## 5. **Data Orchestration**

### Workflow Orchestration
- **Definition**: Automating and scheduling data pipeline tasks.
- **Key Features**: Dependency management, retries, monitoring.
- **Tools**:
  - **Apache Airflow**: DAG-based workflows, Python-based.
  - **Prefect**: Modern, Python-first orchestration.
  - **Dagster**: Data-aware orchestration with asset management.
  - **Luigi**: Lightweight, Python-based.

### Scheduling
- **Cron Jobs**: Time-based scheduling (e.g., `0 0 * * *` for daily at midnight).
- **Event-Driven**: Triggered by events (e.g., file arrival, API call).
- **Tools**: Airflow Scheduler, AWS EventBridge, Kubernetes CronJobs.

## 6. **Data Governance**

### Data Quality
- **Aspects**:
  - **Accuracy**: Correctness of data.
  - **Completeness**: No missing values.
  - **Consistency**: Uniformity across systems.
  - **Timeliness**: Data availability when needed.
- **Tools**: Great Expectations, Soda, Apache Griffin.

### Data Lineage
- **Definition**: Tracking data flow from source to destination.
- **Use Case**: Debugging, compliance, impact analysis.
- **Tools**: Apache Atlas, DataHub, OpenLineage.

### Data Catalog
- **Definition**: Metadata repository for discovering and managing data assets.
- **Examples**: AWS Glue Data Catalog, Alation, Collibra.

### Data Security
- **Techniques**:
  - **Encryption**: Data at rest (AES-256) and in transit (TLS).
  - **Access Control**: Role-based access control (RBAC), attribute-based access control (ABAC).
  - **Data Masking**: Obfuscating sensitive data.
- **Tools**: AWS IAM, Apache Ranger, HashiCorp Vault.

## 7. **Big Data Technologies**

### Distributed Computing
- **Hadoop**: Distributed storage (HDFS) and processing (MapReduce).
- **Spark**: In-memory processing for batch and streaming.
- **Flink**: Stream processing with low latency.

### Data Formats
- **Parquet**: Columnar storage, optimized for analytics.
- **Avro**: Row-based, schema evolution support.
- **ORC**: Optimized Row Columnar, high compression.

### Message Queues
- **Definition**: Systems for asynchronous data transfer.
- **Examples**: Apache Kafka, RabbitMQ, AWS SQS.

## 8. **Cloud Data Engineering**

### Cloud Providers
- **AWS**: S3, Redshift, Glue, Athena, Kinesis.
- **Azure**: Data Lake, Synapse Analytics, Data Factory.
- **GCP**: BigQuery, Dataflow, Dataproc, Pub/Sub.

### Serverless Data Processing
- **Definition**: Running data pipelines without managing servers.
- **Examples**: AWS Lambda, Google Cloud Functions, Azure Functions.

### Managed Services
- **Data Warehouses**: Snowflake, BigQuery, Redshift.
- **ETL Tools**: AWS Glue, Azure Data Factory, Google Dataform.
- **Streaming**: AWS Kinesis, Azure Event Hubs, Google Pub/Sub.

## 9. **Monitoring and Logging**

### Monitoring
- **Metrics**: Latency, throughput, error rates.
- **Tools**: Prometheus, Grafana, Datadog.

### Logging
- **Purpose**: Debugging, auditing, performance tracking.
- **Tools**: ELK Stack (Elasticsearch, Logstash, Kibana), AWS CloudWatch.

### Alerting
- **Definition**: Notifying teams of pipeline failures or anomalies.
- **Tools**: PagerDuty, Slack integrations, AWS SNS.

## 10. **Best Practices**

- **Modularity**: Break pipelines into reusable components.
- **Idempotency**: Ensure repeated operations produce the same result.
- **Version Control**: Use Git for code and dbt for data models.
- **Testing**: Validate data quality, pipeline logic, and performance.
- **Documentation**: Maintain clear pipeline and schema documentation.
- **Scalability**: Design for increasing data volumes and complexity.