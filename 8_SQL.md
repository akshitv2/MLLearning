# 8 SQL

This guide provides an overview of essential SQL concepts, commands, and data types for managing and querying relational databases. Each section includes explanations and examples to help you understand and apply SQL effectively.

## Table Management

### Creating and Modifying Tables
Tables are the core structure for storing data in a relational database. Below are commands to create, modify, and delete tables.

- **Create a Table**:
  Use the `CREATE TABLE` statement to define a new table with specified columns and data types.
  ```sql
  CREATE TABLE Table1 (
      ColA INT,
      ColB VARCHAR(50),
      ColC DATE
  );
  ```

- **Alter a Table**:
  Modify an existing table to add, change, or remove columns.
  ```sql
  ALTER TABLE Table1 ADD ColD FLOAT;           -- Adds a new column
  ALTER TABLE Table1 MODIFY ColA BIGINT;       -- Modifies an existing column's data type
  ALTER TABLE Table1 DROP ColC;               -- Removes a column
  ```

- **Drop a Table**:
  Delete an entire table and its data permanently.
  ```sql
  DROP TABLE Table1;
  ```

### Indexes
Indexes improve the speed of data retrieval but may slow down data insertion and require additional storage.

- **Create an Index**:
  ```sql
  CREATE INDEX Index1 ON Table1(ColA);
  ```
- **Drop an Index**:
  ```sql
  DROP INDEX Index1;
  ```

### Views
A view is a virtual table based on the result of an SQL query. It dynamically reflects changes in the underlying tables.

- **Create a View**:
  ```sql
  CREATE VIEW ViewName AS
  SELECT ColA, ColB
  FROM Table1
  WHERE ColA > 10;
  ```
  Views are useful for simplifying complex queries and providing a layer of abstraction.

## Data Manipulation

### Inserting Data
Add new records to a table using the `INSERT INTO` statement.

- **Insert with All Columns**:
  ```sql
  INSERT INTO Table1 VALUES (1, 'ABC', '2023-01-01');
  ```
- **Insert Specific Columns**:
  ```sql
  INSERT INTO Table1 (ColA, ColB) VALUES (1, 'ABC');
  ```

### Updating Data
Modify existing records using the `UPDATE` statement.

```sql
UPDATE Table1
SET ColA = 'NewValue'
WHERE ColB = 'Condition';
```

### Deleting Data
Remove records from a table using the `DELETE` statement.

```sql
DELETE FROM Table1
WHERE ColA = 5;
```

### Transaction Control
Ensure data integrity with transaction commands.

- **Commit Changes**:
  ```sql
  COMMIT;
  ```
- **Rollback Changes**:
  ```sql
  ROLLBACK;
  ```

## Querying Data

### Selecting Data
Retrieve data from tables using the `SELECT` statement.

- **Basic Selection**:
  ```sql
  SELECT ColA, ColB
  FROM Table1;
  ```
- **Distinct Values**:
  ```sql
  SELECT DISTINCT ColA
  FROM Table1;
  ```
- **Aggregate Functions**:
  Perform calculations on data.
  ```sql
  SELECT ColA,
         AVG(ColB),
         SUM(ColB),
         MIN(ColB),
         MAX(ColB)
  FROM Table1
  GROUP BY ColA
  HAVING ColA > 10;
  ```

### Sorting and Limiting
Control the order and number of results.

- **Order By**:
  Sort results in ascending (`ASC`) or descending (`DESC`) order.
  ```sql
  SELECT ColA
  FROM Table1
  ORDER BY ColA ASC;
  ```
- **Limit and Offset**:
  Restrict the number of rows returned and skip a specified number of rows.
  ```sql
  SELECT ColA
  FROM Table1
  LIMIT 10
  OFFSET 5;
  ```

### String Functions
Manipulate string data with built-in functions.

```sql
SELECT UPPER('text'),          -- Converts to uppercase
       LOWER('TEXT'),          -- Converts to lowercase
       LENGTH('text'),         -- Returns string length
       TRIM('  text  '),       -- Removes leading/trailing spaces
       SUBSTRING('text', 1, 3),-- Extracts substring
       CONCAT('text', 'more'), -- Concatenates strings
       REPLACE('text', 't', 'T') -- Replaces characters
FROM Table1;
```

### Date Filtering
Filter records based on date and time conditions.

```sql
SELECT ColA
FROM Table1
WHERE ColDate >= NOW() - INTERVAL '30 DAY';   -- Last 30 days
WHERE ColDate >= NOW() - INTERVAL '1 MONTH';  -- Last month
WHERE ColDate >= NOW() - INTERVAL '1 YEAR';   -- Last year
```

### Conditional Logic
Use `CASE` statements for conditional logic in queries.

```sql
SELECT ColA,
       CASE
           WHEN ColA > 5 THEN 'High'
           WHEN ColA = 5 THEN 'Medium'
           ELSE 'Low'
       END AS Category
FROM Table1;
```

### Joins
Combine data from multiple tables based on related columns.

- **Inner Join**:
  Returns only matching records from both tables.
  ```sql
  SELECT Table1.ColA, Table2.ColB
  FROM Table1
  INNER JOIN Table2 ON Table1.ColX = Table2.ColX;
  ```
- **Left Join**:
  Returns all records from the left table and matching records from the right table.
  ```sql
  SELECT Table1.ColA, Table2.ColB
  FROM Table1
  LEFT JOIN Table2 ON Table1.ColX = Table2.ColX;
  ```
- **Right Join**:
  Returns all records from the right table and matching records from the left table.
  ```sql
  SELECT Table1.ColA, Table2.ColB
  FROM Table1
  RIGHT JOIN Table2 ON Table1.ColX = Table2.ColX;
  ```
- **Full Join**:
  Returns all records when there is a match in either table.
  ```sql
  SELECT Table1.ColA, Table2.ColB
  FROM Table1
  FULL JOIN Table2 ON Table1.ColX = Table2.ColX;
  ```
- **Cross Join**:
  Returns the Cartesian product of both tables.
  ```sql
  SELECT Table1.ColA, Table2.ColB
  FROM Table1
  CROSS JOIN Table2;
  ```

### Window Functions
Perform calculations across a set of rows related to the current row.

```sql
SELECT ColA,
       ROW_NUMBER() OVER (PARTITION BY ColA ORDER BY ColB) AS RowNum,
       RANK() OVER (PARTITION BY ColA ORDER BY ColB) AS Rank,
       SUM(ColB) OVER (PARTITION BY ColA ORDER BY ColB) AS RunningTotal
FROM Table1;
```

- **Partition By**: Divides the result set into partitions.
- **Order By**: Specifies the order within each partition.

- ### Manual Row Selection:
- ```sql
  SELECT ColA,
       ROW_NUMBER() OVER (PARTITION BY ColA ORDER BY ColB) AS RowNum,
       RANK() OVER (PARTITION BY ColA ORDER BY ColB) AS Rank,
       SUM(ColB) OVER (
  PARTITION BY ColA
  ORDER BY ColB
  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS RunningTotal
  FROM Table1;
  ```
- For selecting only two rows before this: `` ROWS BETWEEN 2 PRECEDING AND CURRENT ROW``
- For selecting row after this: `` ROWS BETWEEN UNBOUNDED PRECEDING AND 2 FOLLOWING``

## Data Types

The following table lists common SQL data types, their categories, and descriptions.

| Category       | Data Type            | Description                                               |
|----------------|----------------------|-----------------------------------------------------------|
| **Numeric**    | `INT` / `INTEGER`    | Whole numbers (e.g., 42)                                  |
|                | `SMALLINT`           | Smaller range of whole numbers (e.g., -32768 to 32767)    |
|                | `BIGINT`             | Very large integers (e.g., -2^63 to 2^63-1)               |
|                | `DECIMAL(p, s)`      | Fixed precision and scale (e.g., 123.45 for money)        |
|                | `NUMERIC(p, s)`      | Same as DECIMAL, exact precision                          |
|                | `FLOAT` / `REAL`     | Approximate floating-point numbers (e.g., 3.14)           |
|                | `DOUBLE PRECISION`   | Higher precision floating-point numbers                   |
| **Character**  | `CHAR(n)`            | Fixed-length string, pads with spaces (e.g., 'abc  ')     |
|                | `VARCHAR(n)`         | Variable-length string with max length (e.g., 'abc')      |
|                | `TEXT`               | Large variable-length text                                |
| **Date/Time**  | `DATE`               | Calendar date (e.g., 2023-01-01)                          |
|                | `TIME`               | Time of day (e.g., 14:30:00)                              |
|                | `DATETIME`           | Date and time (e.g., 2023-01-01 14:30:00)                 |
|                | `TIMESTAMP`          | Date and time with timezone/epoch support                  |
|                | `INTERVAL`           | A span of time (e.g., 2 days, used in PostgreSQL)         |
| **Boolean**    | `BOOLEAN`            | TRUE or FALSE values                                      |
| **Binary/Other** | `BLOB`             | Binary Large Object (e.g., images, files)                 |
|                | `BYTEA`              | Binary data (PostgreSQL-specific)                         |
|                | `JSON` / `JSONB`     | Stores JSON data (JSONB for binary JSON in PostgreSQL)    |
|                | `UUID`               | Universally unique identifier (e.g., 123e4567-e89b-...)  |
|                | `XML`                | Stores XML data                                           |