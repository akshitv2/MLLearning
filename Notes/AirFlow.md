---
title: Apache AirFlow
nav_order: 13
parent: Notes
layout: default
---

### **Apache Airflow: A Quick-Start Guide**

Apache Airflow is a platform for programmatically authoring, scheduling, and monitoring workflows. It defines workflows
as **Directed Acyclic Graphs (DAGs)**, a collection of tasks with a defined order and dependencies.

---

### **1. Core Concepts** 🛠️

* **DAGs**: The full workflow, defined in a single Python file.
* **Tasks**: A single unit of work within a DAG.
* **Operators**: Classes that define a type of task, such as running a Python function or a Bash command.
* **Task Instances**: A specific run of a task at a given time.

---

### **2. Your First DAG** 📝

A DAG is defined in a Python file placed in Airflow's `dags/` folder. Here's a simple example:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

with DAG(
        dag_id='my_first_dag',
        start_date=datetime(2025, 1, 1),
        schedule_interval=timedelta(days=1),
        catchup=False
) as dag:
    # Task 1: A simple bash command
    task1 = BashOperator(
        task_id='print_date',
        bash_command='date'
    )

    # Task 2: Another bash command, but it runs after task1
    task2 = BashOperator(
        task_id='sleep_task',
        bash_command='sleep 5'
    )

    # Setting the dependency: task2 runs only after task1 succeeds
    task1 >> task2
```

- Operators: The code uses BashOperator to execute shell commands.
- Dependencies: The >> operator sets the order of execution.

3. Passing Data Between Tasks (XComs) 🤝
   For passing small amounts of data between tasks, you use XComs. A task's return value is automatically pushed to
   XCom, and another task can pull it.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


# This function pushes data to XCom by returning a value
def push_data():
    return "Hello, Airflow!"


# This function pulls data from XCom using the 'ti' object
def pull_data(ti):
    pulled_value = ti.xcom_pull(task_ids='push_task')
    print(f"The pulled value is: {pulled_value}")


with DAG(
        dag_id='xcom_example',
        start_date=datetime(2025, 1, 1),
        schedule_interval=None,
        catchup=False
) as dag:
    push_task = PythonOperator(
        task_id='push_task',
        python_callable=push_data
    )

    pull_task = PythonOperator(
        task_id='pull_task',
        python_callable=pull_data
    )

    push_task >> pull_task
```

- `ti` (TaskInstance): This object is implicitly passed to the Python function and is used to call methods like
  `xcom_pull()`.
- task_ids: The argument to `xcom_pull()` specifies which task to get the data from.

4. Error Handling 🚨
   Airflow's default behavior is to stop downstream tasks if an upstream task fails. This is a crucial feature for data
   integrity.

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
        'error_handling_dag',
        start_date=datetime(2025, 1, 1),
        schedule_interval=None
) as dag:
    failing_task = BashOperator(
        task_id='failing_task',
        bash_command='exit 1'  # This command will cause the task to fail
    )

    downstream_task = BashOperator(
        task_id='downstream_task',
        bash_command='echo "This task will not run."'
    )

    failing_task >> downstream_task
```

- If failing_task fails, downstream_task will be marked as upstream_failed and will not run.

5. Conditional Branching 🚦
   To run a specific set of tasks based on a condition, you use the BranchPythonOperator.

```python
from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from datetime import datetime


# Branching function that returns the task_id of the next task
def choose_path():
    value = 5
    if value % 2 == 0:
        return 'even_number_path'
    else:
        return 'odd_number_path'


with DAG(
        'branching_dag',
        start_date=datetime(2025, 1, 1),
        schedule_interval=None,
        catchup=False
) as dag:
    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=choose_path
    )

    even_task = BashOperator(
        task_id='even_number_path',
        bash_command='echo "The number is even."'
    )

    odd_task = BashOperator(
        task_id='odd_number_path',
        bash_command='echo "The number is odd."'
    )

    # Connect the branch task to all possible paths
    branch_task >> [even_task, odd_task]
```

- The BranchPythonOperator returns the task_id of the next task to run.
- All other tasks connected to the branch are automatically skipped.

6. Python vs. Bash Operators 🐍🖥️

- PythonOperator: Runs a Python function. Ideal for custom data processing, API calls, and complex logic.
- BashOperator: Runs a shell command. Perfect for running existing shell scripts or command-line utilities.

7. Asynchronous Tasks 🏃‍♂️
   While Airflow itself is a sequential orchestrator, you can run async code within a task. This requires using specific
   async-aware operators or wrapping your async code in a blocking function.
```python
# A simple example of how to wrap an async function
import asyncio
from airflow.operators.python import PythonOperator

def run_async_task():
    async def my_async_function():
        await asyncio.sleep(1)
        return "Async task complete!"
    return asyncio.run(my_async_function())
```
- The run_async_task function is synchronous, but it handles the asyncio event loop internally.

## Real-World Examples
1. Data Ingestion 📊
A common real-world use case is a large-scale data ingestion pipeline. A single Airflow DAG might coordinate the following tasks:
   1. Task 1: A S3Sensor waits for a new data file to land in an S3 bucket.
   2. Task 2: A SparkSubmitOperator triggers a Spark job on a cluster to process and transform the raw data.
   3. Task 3: A MySqlOperator runs a query to load the transformed data into a data warehouse.
   4. Task 4: A SlackOperator sends a notification to a team channel once the entire process is complete. 
   - In this scenario, Airflow isn't running the Spark job itself; it's simply submitting the job and waiting for a success or failure signal.

2. Machine Learning Pipelines 🤖
Airflow is often used to manage the full lifecycle of a machine learning model. A single DAG might include:
   1. Task 1: A PythonOperator retrieves and preprocesses data.
   2. Task 2: A DockerOperator runs a containerized training script on a powerful machine.
   3. Task 3: A KubernetesPodOperator deploys the newly trained model to a Kubernetes cluster for serving.
   4. Task 4: An EmailOperator sends an email to the MLOps team with a link to the deployed model.

3. ETL for Big Data 📈
For large-scale Extract, Transform, Load (ETL) jobs, Airflow can orchestrate tasks across different platforms:
   1. Task 1: A HiveOperator extracts data from a Hive table on a Hadoop cluster.
   2. Task 2: A DataflowOperator triggers a Google Cloud Dataflow job to perform complex transformations.
   3. Task 3: A PostgresOperator loads the final, transformed data into a PostgreSQL database.
- How It Works at Scale
The key to Airflow's scalability is the separation of its components:
1. Scheduler: The scheduler is the core component that decides when tasks should run.
2. Executor: The executor is the mechanism for running those tasks. Airflow supports various executors, from a simple SequentialExecutor for testing to a KubernetesExecutor that can scale up and down dynamically.
3. Workers: The workers are the processes that actually run the tasks.