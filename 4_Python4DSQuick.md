<h1>Tensorflow Syntax and Code Help</h1>

1. **tf.data.Dataset**
    Basic Dataset type

    - Ways to create
    ```
    # From a Python list
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])

    # From a generator
    dataset = tf.data.Dataset.from_generator(my_generator, output_signature=...)

    # From TFRecord files
    dataset = tf.data.TFRecordDataset(["file1.tfrecord", "file2.tfrecord"])
    ```

    - Common Methods
        - .map: Transforms each element in the dataset using a function.
            ```
            dataset = dataset.map(lambda x: x * 2)
            ```
        - batch(batch_size)
            ```
            dataset = dataset.batch(2)
            ```
        - shuffle(buffer_size) Randomly shuffles the elements of the dataset.
            ```
            dataset = dataset.shuffle(buffer_size=100)
            ```
        - filter() Filters elements based on a condition.
            ```
            dataset = dataset.filter(lambda x: x % 2 == 0)
            ```




2. **PrefetchDataset (tensorflow.python.data.ops.prefetch_op._PrefetchDataset
)**
    - Wraps another dataset and prefetches elements from it asynchronously to improve performance (not used directly)
    ```
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    ```
2. s