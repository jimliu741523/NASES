import tensorflow as tf
from configparser import ConfigParser


tf.set_random_seed(1)

def weight_variable(shape,name):
    initial =  tf.orthogonal_initializer(0.5)
    return tf.get_variable(name = name, shape = shape, initializer = initial)


def bias_variable(shape,name):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable(name = name, shape = shape ,initializer = initial)

class DNNcoder:
  def __init__(
            self,
            lr=0.00001,
            embedding=20,
            origin_len=60
         ):  
      self.origin_len = origin_len
      self.embedding = embedding
      self.encoder_inputs = tf.placeholder(shape=[None,self.origin_len],dtype=tf.float32,name='encoder_in')
      self.de_in = tf.placeholder(shape=[None,embedding],dtype=tf.float32,name='de_in')
      self.decoder_targets = tf.placeholder(shape=[None,self.origin_len], dtype=tf.float32, name='decoder_tar')
      self.condition = tf.placeholder(tf.int32, shape=[], name="condition")

   
      
      self.encoder = self.encoder()      
      self.decoder = self.decoder()
      
      self.losses = tf.losses.mean_squared_error(labels=self.decoder_targets,
                                                 predictions=self.decoder)
      
      self.train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.losses)

      self.saver = tf.train.Saver()
    
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())   
     
  def encoder(self):
      x1 = self.encoder_inputs
      self.w1 = weight_variable(shape = [self.origin_len,self.origin_len],name = 'encoder_w1')
      self.b1 = bias_variable([self.origin_len],name = 'encoder_b1') 
      l1 = tf.nn.relu(tf.matmul(x1,self.w1)+self.b1)
#       l1 = tf.layers.batch_normalization(l1)
      self.w2 = weight_variable(shape = [self.origin_len,(self.origin_len*90)],name = 'encoder_w2')
      self.b2 = bias_variable([(self.origin_len*90)],name = 'encoder_b2') 
      l2 = tf.nn.tanh(tf.matmul(l1,self.w2)+self.b2)
#       l2 = tf.layers.batch_normalization(l2)

      self.w3 = weight_variable(shape = [(self.origin_len*90),(self.origin_len*30)],name = 'encoder_w3')
      self.b3 = bias_variable([(self.origin_len*30)],name = 'encoder_b3') 
      l3 = tf.nn.tanh(tf.matmul(l2,self.w3)+self.b3)
#       l3 = tf.layers.batch_normalization(l3)
      
      self.w4 = weight_variable([(self.origin_len*30),self.embedding],name = 'encoder_w4')
      self.b4 = bias_variable([self.embedding],name = 'encoder_b4') 
      self.en_outputs = tf.nn.sigmoid(tf.matmul(l3,self.w4)+self.b4)

      
      return self.en_outputs

  def show_weight(self,x):
      fd={self.encoder_inputs: x
            }   
      return self.sess.run([self.w1,self.b1,self.w2,self.b2,self.w3,self.b3,self.w4,self.b4])
  
  def decoder(self):
 
      de_in = tf.cond(self.condition > 0,lambda: self.en_outputs,lambda: self.de_in)

      w5 = weight_variable(shape = [self.embedding,(self.origin_len*30)],name = 'decoder_w5')
      b5 = bias_variable([(self.origin_len*30)],name = 'decoder_b5') 
      l5 = tf.nn.tanh(tf.matmul(de_in,w5)+b5)
#       l5 = tf.layers.batch_normalization(l5)  
      
      w6 = weight_variable(shape = [(self.origin_len*30),(self.origin_len*90)],name = 'decoder_w6')
      b6 = bias_variable([(self.origin_len*90)],name = 'decoder_b6') 
      l6 = tf.nn.tanh(tf.matmul(l5,w6)+b6)
#       l6 = tf.layers.batch_normalization(l6)
      
      w7 = weight_variable([(self.origin_len*90),self.origin_len],name = 'decoder_w7')
      b7 = bias_variable([self.origin_len],name = 'decoder_b7') 
      self.output = tf.nn.relu(tf.matmul(l6,w7)+b7)
      

      return self.output


  def code(self,x): 
      fd={self.encoder_inputs: x
            }

      return self.sess.run(self.en_outputs,fd)
    
  
  def loss(self,x,y,z,c): 
      fd={self.encoder_inputs: x,
          self.decoder_targets: y,          
          self.de_in: z,
          self.condition: c
            }

      return self.sess.run(self.losses,fd)
          
  def train(self,x,y,z,c):
      fd={self.encoder_inputs: x,
          self.decoder_targets: y,
          self.de_in: z,
          self.condition: c
            }

      self.sess.run(self.train_step,fd)
  
  def pred(self,x,z,c):
      fd={self.encoder_inputs: x,
          self.de_in: z,
          self.condition: c
            }

      return self.sess.run(self.output,fd)
  
  def save(self):
      self.saver.save(self.sess, "../scr/cifar10/autoencoderModel/DNNmodel.ckpt")
      print("Model Save.")
    
  def restore(self):
      self.saver.restore(self.sess, "../scr/cifar10/autoencoderModel/DNNmodel.ckpt")
      print("Model restored.")
    
if __name__ == '__main__':   
    DNNcoder()

