import tensorflow as tf

# 查看可用的设备
gpus = tf.config.list_physical_devices('GPU')
print("GPUs: ", gpus)

# 检查 TensorFlow 是否使用了 GPU
if gpus:
    print("Using GPU for training")
else:
    print("No GPU detected, using CPU instead")