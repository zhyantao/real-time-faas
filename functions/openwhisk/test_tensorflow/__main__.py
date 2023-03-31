import tensorflow as tf

def main():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    return {'msg: ': sess.run(hello)}
