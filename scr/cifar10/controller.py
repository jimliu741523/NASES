import tensorflow as tf

tf.set_random_seed(1)

 
    
class controller:
  def __init__(
            self,
            pre_init,
            lr=0.00001,
            origin_len=60, 
            embedding=20
            
         ):    
      self.pre_init =  pre_init
      self.code_inputs = tf.placeholder(shape=[None,origin_len],dtype=tf.float32,name='code_inputs')
      self.tf_vt = tf.placeholder(tf.float32,[None,1], name="actions_value")
      self.tf_act = tf.placeholder(tf.float32, [None,embedding], name="actions_value")
      self.model_output = self.build_policy_network(code_inputs = self.code_inputs)
      
      self.neg_log_prob = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.model_output,labels = self.tf_act)

      self.pg_loss = tf.reduce_mean(self.neg_log_prob* self.tf_vt)

        
      self.l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()  ]) 
    
      self.losses = self.pg_loss + 0.002 * self.l2_loss
      self.train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.losses) 

        
      self.saver = tf.train.Saver()
    
      self.sess = tf.Session()
      init = tf.global_variables_initializer()
      self.sess.run(init)      
    


      
  def build_policy_network(self,code_inputs):   
      
      x1 = self.code_inputs 
      w1 = tf.get_variable(name = 'controller-w1',initializer=self.pre_init[0])
      b1 = tf.get_variable(name = 'controller-b1',initializer=self.pre_init[1])
      l1 = tf.nn.relu(tf.matmul(x1,w1)+b1)
      
      w2 =  tf.get_variable(name = 'controller-w2',initializer=self.pre_init[2])
      b2 =  tf.get_variable(name = 'controller-b2',initializer=self.pre_init[3])
      l2 =  tf.nn.tanh(tf.matmul(l1,w2)+b2)

      w3 =  tf.get_variable(name = 'controller-w3',initializer=self.pre_init[4])
      b3 =  tf.get_variable(name = 'controller-b3',initializer=self.pre_init[5])
      l3 = tf.nn.tanh(tf.matmul(l2,w3)+b3)
            
      w4 = tf.get_variable(name = 'controller-w4',initializer=self.pre_init[6])
      b4 = tf.get_variable(name = 'controller-b4',initializer=self.pre_init[7])
      self.l4 = tf.nn.sigmoid(tf.matmul(l3,w4)+b4)
    
    

      return self.l4
    
      
  def train(self,state,reward, action):
      self.sess.run(self.train_op, feed_dict={
             self.code_inputs: state, 
             self.tf_vt: reward,
             self.tf_act: action})
        
  def loss(self,state,reward, action):
      output = self.sess.run(self.losses, feed_dict={
             self.code_inputs: state, 
             self.tf_vt: reward,
             self.tf_act: action})
      
      return output
      
  def preds(self,state):
      
      output = self.sess.run(self.l4, feed_dict={
                  self.code_inputs: state})
      
      return output
              
  def save(self):
      self.saver.save(self.sess, "../scr/cifar10/autoencoderModel/Controller.ckpt")
      print("Model Save - Controller.")
    
  def restore(self):
      self.saver.restore(self.sess, "../scr/cifar10/autoencoderModel/Controller.ckpt")
      print("Model Restored - Controller.")
    
if __name__ == '__main__':
    main()