import tensorflow as tf
from  autoencoderTrain import  train as AT
from train import run as RUN



tf.app.flags.DEFINE_string('process', 'final','process')
tf.app.flags.DEFINE_integer('origin_len', '60','length of architecture vector')
tf.app.flags.DEFINE_integer('embedding', '20','length of architecture-embedding vector')
tf.app.flags.DEFINE_integer('addFilter', '0','more filters')




FLAGS = tf.app.flags.FLAGS
 
origin_len = FLAGS.origin_len
embedding = FLAGS.embedding
process = FLAGS.process
addFilter = FLAGS.addFilter


def main(_):  
    if process == 'simulating':
        AT(embedding, origin_len)        
    elif process == 'searching' or process == 'final' :
        RUN(process,embedding, origin_len, addFilter)
        
        
 
if __name__ == '__main__':
    tf.app.run()
