# Data Engineering Interview Questions (3-5 yrs experience)

---

## Question 1

**Q:** How would you design a data pipeline to process daily logs from millions of users, clean them, and make them
queryable for analytics?
**A:**

* Ingest via Kafka/Kinesis.
* Parallel processing with Spark.
* Clean, encode categorical, remove irrelevant data.
* Store raw + processed in data lake.
* Load curated data to warehouse/OLAP store.
* Use Airflow/DBT for orchestration.
* Schema management (Avro/Parquet/Delta Lake).
* Monitoring, retries, data quality checks.

---

## Question 2

**Q:** Dataset of 100M CSVs in S3; Athena queries slow. How to optimize storage & query performance?

**A:**

* Convert CSV → Parquet/ORC (columnar + compressed).
* Partition by date, region; bucket for high-cardinality fields.
* Column pruning to avoid reading irrelevant columns.
* Schema management via Glue/Hive.
* Optional: materialized views or hot subsets in warehouse.

---

## Question 3

**Q:** In streaming pipelines, how to handle late-arriving events?

**A:**

* Use event-time processing with watermarks.
* Route extreme late events to side output / dead-letter queue.
* Ensure idempotency, exactly-once semantics.
* Backfill or batch layer for reconciliation.

---

## Question 4

**Q:** Suppose you’re building a feature store for machine learning. How would you design the pipeline to ensure
features are: (1) consistent between training and serving, (2) available with low latency for real-time inference, and (
3) reproducible for audits?

**A:**

* Central feature registry to define transformations.
* Batch features: compute with Spark/DBT, store in Delta Lake/Parquet.
* Real-time features: stream from Kafka/Flink → online store (Redis/DynamoDB).
* Versioned snapshots for reproducibility.
* Raw data archival for audit.

---

## Question 5

**Q:** Ensure data quality in a pipeline ingesting multiple sources into a warehouse.

**A:**

* Multi-layer checks: ingest-time (schema, types, nulls), processing-time (dedupe, range, referential integrity),
  post-load (reconciliation).
* Schema enforcement + contracts (Avro/Protobuf, registry).
* Automated tests (Great Expectations / Deequ / dbt tests) in CI/CD.
* Monitoring/alerts (null rate, drift, throughput).
* Lineage & governance (catalog, owners, runbooks).
* Reconciliation/backfill; idempotent pipelines.
* Human review of quarantined samples; feedback loop to producers.
