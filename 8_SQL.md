# SQL

<span style="color:orange">Table1</span>
<span style="color:yellow">Table1</span>
<span style="color:green">Table1</span>
<span style="color:aqua">Table1</span>
<span style="color:lightblue">Table1</span>
<span style="color:red">Table1</span>

DISTINCT (<span style="color:yellow">ColA</span>)
COUNT(<span style="color:yellow">ColA</span>)

ORDER BY <span style="color:yellow">ColA</span> ASC|DESC
LIMIT N
OFFSET N

ALTER TABLE <span style="color:orange">Table1</span> ADD <span style="color:yellow">ColA</span> dtype  
ALTER TABLE <span style="color:orange">Table1</span> MODIFY <span style="color:yellow">ColA</span> dtype  
ALTER TABLE <span style="color:orange">Table1</span> Drop <span style="color:yellow">ColA</span>

DROP TABLE <span style="color:orange">Table1</span>

## Index:
- Fast retrieval, slow insertion
- Uses more spce
- CREATE INDEX Index1 on <span style="color:orange">Table1</span>(<span style="color:yellow">ColA</span>)
- DROP INDEX Index1

## View
- View is a virtual table based on the result-set of an SQL statement.  
- A view always shows up-to-date data! The database engine recreates the view, every time a user queries it.
CREATE VIEW ViewName AS SELECT <span style="color:yellow">ColA</span>, <span style="color:yellow">ColB</span> FROM <span style="color:orange">Table1</span>;

## Insertion:
INSERT INTO <span style="color:orange">Table1</span> values (1,'ABC',...)  
INSERT INTO <span style="color:orange">Table1</span>(<span style="color:yellow">ColA</span>, <span style="color:yellow">ColB</span>) values (1,'ABC',...)

## Update:
UPDATE <span style="color:orange">Table1</span> SET <span style="color:yellow">ColA</span> = 'SomeValue' where <span style="color:yellow">ColB</span> = 'This'

## Deletion
DELETE FROM <span style="color:orange">Table1</span> where ...

COMMIT;
ROLLBACK;

SELECT <span style="color:yellow">ColA</span>,  
    AVG(<span style="color:yellow">ColB</span>),  
    MEAN(<span style="color:yellow">ColB</span>),  
    SUM(<span style="color:yellow">ColB</span>),  
    MIN(<span style="color:yellow">ColB</span>),  
    MAX(<span style="color:yellow">ColB</span>)  
FROM <span style="color:orange">Table1</span> group by <span style="color:yellow">ColA</span> having <span style="color:yellow">ColA</span> > ...;

## String Utils
SELECT  
UPPER('...'),  
LOWER('...'),  
LENGTH('...'),  
TRIM('...'),
SUBSTRING('...', start, end),  
CONCAT('...','...'),  
REPLACE('...','...')  
from ....

### Date Selection
WHERE ColDate >= NOW() - INTERVAL 30 DAY  
WHERE ColDate >= NOW() - INTERVAL 1 MONTH  
WHERE ColDate >= NOW() - INTERVAL 1 YEAR  

### Case
SELECT CASE  
    WHEN <span style="color:yellow">ColA</span> > 5 THEN 'Something'  
    WHEN <span style="color:yellow">ColA</span> = 5 THEN 'SomethingElse'  
    ELSE 'Something else again'  
FROM ...

### JOINS
SELECT <span style="color:orange">Table1</span>.<span style="color:yellow">ColA</span>, <span style="color:orange">Table2</span>.<span style="color:yellow">ColB</span> FROM T1 as <span style="color:orange">Table1</span> <span style="color:red">INNER JOIN</span> T2 as <span style="color:orange">Table2</span> ON <span style="color:orange">Table1</span>.<span style="color:yellow">ColX</span> = <span style="color:orange">Table2</span>.<span style="color:yellow">ColX</span>;   
SELECT <span style="color:orange">Table1</span>.<span style="color:yellow">ColA</span>, <span style="color:orange">Table2</span>.<span style="color:yellow">ColB</span> FROM T1 as <span style="color:orange">Table1</span> <span style="color:red">LEFT JOIN</span> T2 as <span style="color:orange">Table2</span> ON <span style="color:orange">Table1</span>.<span style="color:yellow">ColX</span> = <span style="color:orange">Table2</span>.<span style="color:yellow">ColX</span>;   
SELECT <span style="color:orange">Table1</span>.<span style="color:yellow">ColA</span>, <span style="color:orange">Table2</span>.<span style="color:yellow">ColB</span> FROM T1 as <span style="color:orange">Table1</span> <span style="color:red">RIGHT JOIN</span> T2 as <span style="color:orange">Table2</span> ON <span style="color:orange">Table1</span>.<span style="color:yellow">ColX</span> = <span style="color:orange">Table2</span>.<span style="color:yellow">ColX</span>;   
SELECT <span style="color:orange">Table1</span>.<span style="color:yellow">ColA</span>, <span style="color:orange">Table2</span>.<span style="color:yellow">ColB</span> FROM T1 as <span style="color:orange">Table1</span> <span style="color:red">FULL JOIN</span> T2 as <span style="color:orange">Table2</span> ON <span style="color:orange">Table1</span>.<span style="color:yellow">ColX</span> = <span style="color:orange">Table2</span>.<span style="color:yellow">ColX</span>;   
SELECT <span style="color:orange">Table1</span>.<span style="color:yellow">ColA</span>, <span style="color:orange">Table2</span>.<span style="color:yellow">ColB</span> FROM T1 as <span style="color:orange">Table1</span> <span style="color:red">CROSS JOIN</span> T2 as <span style="color:orange">Table2</span> ON <span style="color:orange">Table1</span>.<span style="color:yellow">ColX</span> = <span style="color:orange">Table2</span>.<span style="color:yellow">ColX</span>;   

### Window Functions
- Partitioning and order are optional
- But highly recommended to order since data can be unorderly stored
  
SELECT ROW_NUMBER()  
<span style="color:red">RANK()  
SUM()  
OVER ([PARTITION BY <span style="color:yellow">ColA</span>] [ORDER BY <span style="color:yellow">ColB</span>]) </span>AS NewCol from <span style="color:orange">Table1</span>



# Data Types

| Category       | Data Type            | Description                                               |
|----------------|----------------------|-----------------------------------------------------------|
| **Numeric**    | `INT` / `INTEGER`    | Whole numbers                                             |
|                | `SMALLINT`           | Smaller range of whole numbers                            |
|                | `BIGINT`             | Very large integers                                       |
|                | `DECIMAL(p, s)`      | Fixed precision and scale (exact decimals, e.g. money)    |
|                | `NUMERIC(p, s)`      | Same as DECIMAL, exact precision                          |
|                | `FLOAT` / `REAL`     | Approximate floating-point numbers                        |
|                | `DOUBLE PRECISION`   | Higher precision floating-point numbers                   |
| **Character**  | `CHAR(n)`            | Fixed-length string (pads with spaces if shorter)         |
|                | `VARCHAR(n)`         | Variable-length string with a maximum length              |
|                | `TEXT`               | Large variable-length text                                |
| **Date/Time**  | `DATE`               | Calendar date (YYYY-MM-DD)                                |
|                | `TIME`               | Time of day (HH:MM:SS)                                    |
|                | `DATETIME`           | Date and time (YYYY-MM-DD HH:MM:SS)                       |
|                | `TIMESTAMP`          | Date and time, often with timezone/epoch support          |
|                | `INTERVAL`           | A span of time (PostgreSQL, Oracle)                       |
| **Boolean**    | `BOOLEAN`            | TRUE / FALSE values                                       |
| **Binary/Other** | `BLOB`             | Binary Large Object (e.g., images, files)                 |
|                | `BYTEA`              | Binary data (PostgreSQL)                                  |
|                | `JSON` / `JSONB`     | Stores JSON data (PostgreSQL, MySQL)                      |
|                | `UUID`               | Universally unique identifier                             |
|                | `XML`                | Stores XML data                                           |

