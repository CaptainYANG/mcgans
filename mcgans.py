import tensorflow as tf
import numpy as np
import input_data
from tensorflow.python.training import moving_averages

flags = tf.flags
flags.DEFINE_integer("max_steps", 5000,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 64,'Number of steps to run trainer.')
flags.DEFINE_integer("input_size", 784,'Number of steps to run trainer.')
flags.DEFINE_float("G_learning_rate", 5e-5,'Initial learning rate')
flags.DEFINE_float("D_learning_rate", 5e-5,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.') 
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_gan_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_boolean("save_model", True,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_gan_model','Checkpoint dir')
flags.DEFINE_string("critic_iters", 1,'number of critic iters per gen iter')
FLAGS = flags.FLAGS
CLIP_BOUNDS = [-.01, .01]
BN_DECAY = 0.999
# UPDATE_OPS_COLLECTION = 'update_ops'
class Utils():
    def __init__(self, session, checkpoint_dir=None, name="utils"):
        self.name = name
        self.session = session
        self.checkpoint_dir = checkpoint_dir
        self.saver = tf.train.Saver()
        

    def reload_model(self):
        if self.checkpoint_dir is not None:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Reloading from -- '+self.checkpoint_dir+'/model.ckpt')
                self.saver.restore(self.session, ckpt.model_checkpoint_path)

    def save_model(self, step=0):
        import os
        if not os.path.exists(self.checkpoint_dir):
            os.system('mkdir '+self.checkpoint_dir)
        save_path = self.saver.save(self.session, self.checkpoint_dir+'/model.ckpt',write_meta_graph=False)

def batchnorm(Ylogits, name,is_test = False):
  is_test = tf.convert_to_tensor(is_test, dtype='bool',name='is_test')
  bnepsilon = 1e-5
  mean, variance = tf.nn.moments(Ylogits, [0])
  shape = mean.get_shape().as_list()
  moving_mean = tf.get_variable('moving_mean'+name,
                                initializer=tf.zeros(shape),
                                trainable=False)
  moving_variance = tf.get_variable('moving_variance'+name,
                                initializer=tf.ones(shape),
                                trainable=False)
  update_moving_mean = moving_averages.assign_moving_average(moving_mean,mean,BN_DECAY)
  update_moving_variance = moving_averages.assign_moving_average(moving_variance,variance,BN_DECAY)
  # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
  # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

  # https://github.com/tensorflow/tensorflow/issues/5827
  # exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,name=name)
  # update_moving_everages = exp_moving_avg.apply([mean, variance])
  # m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
  # v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
  m = tf.cond(is_test, lambda: moving_mean, lambda: mean)
  v = tf.cond(is_test, lambda: moving_variance, lambda: variance)
  scale = tf.get_variable('scale_'+name, initializer=tf.ones(shape))
  shift = tf.get_variable('shift_'+name, initializer=tf.zeros(shape))
  Ybn = tf.nn.batch_normalization(Ylogits, m, v, shift, scale, bnepsilon)
  return Ybn,[m,v]

def get_var(name):
  return [v for v in tf.trainable_variables() if name in v.name]

def add_layer(inputs, in_size, out_size, name, activation_function = tf.nn.relu):
  with tf.variable_scope('layer'):
    with tf.variable_scope('weights'):
      Weights = tf.get_variable(initializer=tf.random_normal([in_size, out_size]), name='W'+name)
    with tf.variable_scope('biases'):
      biases = tf.get_variable(initializer=tf.zeros([1, out_size]) + 0.1, name='b'+name)
    with tf.variable_scope('Wx_plus_b'):
      Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
      outputs = Wx_plus_b
    else:
      outputs = activation_function(Wx_plus_b)
    return outputs

def generator(inputs, is_test = False):
  bn,_ = batchnorm(inputs,'bn0',is_test)
  gl1 = add_layer(bn,784,800,'gl1')
  bn,_ = batchnorm(gl1,'bn1',is_test)
  gl2 = add_layer(bn,800,800,'gl2')
  bn,_ = batchnorm(gl2,'bn2',is_test)
  gl3 = add_layer(bn,800,800,'gl3')
  bn,_ = batchnorm(gl3,'bn3',is_test)
  gl4 = add_layer(bn,800,10,'gl4',tf.nn.sigmoid)
  return gl4

def discriminator(inputs,is_test = False):
  bn,_ = batchnorm(inputs,'bn0',is_test)
  dl1 = add_layer(bn,10,200,'dl1')
  bn,_ = batchnorm(dl1,'bn1',is_test)
  dl2 = add_layer(bn,200,200,'dl2')
  bn,_ = batchnorm(dl2,'bn2',is_test)
  dl3 = add_layer(bn,200,200,'dl3')
  bn,_ = batchnorm(dl3,'bn3',is_test)
  dl4 = add_layer(bn,200,1, 'dl4',None)
  return dl4

# def compute_D_loss(D1, D2):
#     return tf.nn.sigmoid_cross_entropy_with_logits(logits=D1, labels=tf.ones(tf.shape(D1))) , tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.zeros(tf.shape(D2)))

# def compute_G_loss(D2):
#     return tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.ones(tf.shape(D2)))

def train():
  mnist = input_data.read_data_sets("data_dir", one_hot = True)
  with tf.Session() as sess:
    with tf.name_scope('input'):
      x = tf.placeholder(tf.float32,[None,784], name='x-input')
      y = tf.placeholder(tf.float32,[None,10], name='y-input')
    with tf.variable_scope('model') as model_scope:
      with tf.variable_scope('generator'):
        G = generator(x)
        G_params_num = len(tf.trainable_variables())
      with tf.variable_scope('discriminator') as disc_scope:
        D1 = discriminator(y)
        disc_scope.reuse_variables()
        D2 = discriminator(G)

    total_params = tf.trainable_variables()
    G_params = total_params[:G_params_num]
    D_params = total_params[G_params_num:]
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(G,1),tf.argmax(y,1)), tf.float32))
    tf.summary.scalar('acc', acc)
    with tf.variable_scope('Loss'):
      # D1_loss, D2_loss = compute_D_loss(D1, D2)
      # D_loss = tf.reduce_mean(D1_loss + D2_loss)
      # G_loss = compute_G_loss(D2)
      # tf.summary.scalar('D_real', tf.reduce_mean(D1_loss))
      # tf.summary.scalar('D_fake', tf.reduce_mean(D2_loss))
      # tf.summary.scalar('D_loss', tf.reduce_mean(D_loss))
      # tf.summary.scalar('G_loss', tf.reduce_mean(G_loss))

      #wgan loss
      D_loss = tf.reduce_mean(D2) - tf.reduce_mean(D1)
      G_loss = -tf.reduce_mean(D2)
      tf.summary.scalar('D_loss', tf.reduce_mean(D_loss))
      tf.summary.scalar('G_loss', tf.reduce_mean(G_loss))

    with tf.variable_scope('Trainer'):
      D_trainer = tf.train.RMSPropOptimizer(FLAGS.D_learning_rate).minimize(D_loss)
      G_trainer = tf.train.RMSPropOptimizer(FLAGS.G_learning_rate).minimize(G_loss)
      #clip weights and bias
      clip_ops = [var.assign(tf.clip_by_value(var,CLIP_BOUNDS[0],CLIP_BOUNDS[1])) for var in get_var('discriminator/layer')]
      D_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/D', sess.graph)
      G_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/G', sess.graph)
      merged = tf.summary.merge_all()
      tf.global_variables_initializer().run()
      utils = Utils(sess, FLAGS.checkpoint_dir)
      if FLAGS.reload_model:
        utils.reload_model()

      xt,yt = mnist.test.next_batch(200)
      test = {x:xt, y:yt}
      for i in range(FLAGS.max_steps):
        for j in range(FLAGS.critic_iters):
          xs, ys = mnist.train.next_batch(FLAGS.batch_size)
          D_summary, _ , dloss = sess.run([ merged, D_trainer, D_loss], feed_dict={x:xs, y:ys})
          sess.run(clip_ops)
        G_summary, _ , gloss = sess.run([merged, G_trainer, G_loss], feed_dict={x:xs, y:ys})
        if i%100==0:
          accuracy = sess.run(acc, test)
          print(gloss.mean(), dloss.mean(),accuracy)
        D_writer.add_summary(D_summary, i)
        G_writer.add_summary(G_summary, i)

      if FLAGS.save_model:
        utils.save_model()
      D_writer.close()
      G_writer.close()
      xt,yt = mnist.test.next_batch(200)
      test = {x:xt, y:yt}
      accuracy = sess.run(acc, test)
      print("After training, the accuracy is %g" % accuracy)

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    tf.app.run()