import tensorflow as tf

tf.compat.v1.enable_eager_execution()

d = tf.data.Dataset.list_files("/data/part*")

d = d.interleave(lambda f: tf.data.TFRecordDataset(f), num_parallel_calls=tf.data.experimental.AUTOTUNE)

print("Finding the number of records")
count = 0
for rec in d:
    count = count + 1
    if (count % 10000):
        print(count)
print("Finished")
print(count)
print("Num records: " + count)
