# import tensorflow as tf
# print("TF version:", tf.__version__)
# print("GPUs:", tf.config.list_physical_devices('GPU'))

# import tensorflow as tf
# from tensorflow.python.client import device_lib

# print("TF version:", tf.__version__)
# print("Local devices:\n", device_lib.list_local_devices())

# import os

# # Add both CUDA and cuDNN paths to PATH manually
# os.environ["PATH"] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
# os.environ["PATH"] += r";C:\Tools\CUDA\bin"


# from tensorflow.python.client import device_lib


# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]


# print(get_available_devices())


# from tensorflow.python.client import device_lib
# import tensorflow as tf
# import os
# os.environ["PATH"] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
# os.environ["PATH"] += r";C:\Tools\CUDA\bin"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # force full log output


# print("TensorFlow version:", tf.__version__)
# print("\n--- Full device list ---")
# print(device_lib.list_local_devices())


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs Available:", gpus)
    # Try setting GPU memory growth to avoid allocation issues
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("Logical GPUs:", logical_gpus)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs Available")
