import tensorflow as tf
import numpy as np
import input_data

flags = tf.flags
flags.DEFINE_integer("max_steps", 5000,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("input_size", 784,'Number of steps to run trainer.')
flags.DEFINE_float("G_learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("D_learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.') 
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_gan_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_boolean("save_model", True,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_gan_model','Checkpoint dir')
FLAGS = flags.FLAGS

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

def add_layer(inputs, in_size, out_size, activation_function = None):
  with tf.name_scope('layer'):
    with tf.name_scope('weights'):
      Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    with tf.name_scope('biases'):
      biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
    with tf.name_scope('Wx_plus_b'):
      Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
      outputs = Wx_plus_b
    else:
      outputs = activation_function(Wx_plus_b)
    return outputs

def generator(inputs):
  gl1 = add_layer(inputs,784,800)
  gl2 = add_layer(gl1,800,800)
  gl3 = add_layer(gl2,800,800)
  gl4 = add_layer(gl3,800,10)
  return gl4

def discriminator(inputs):
  dl1 = add_layer(inputs,10,200)
  dl2 = add_layer(dl1,200,200)
  dl3 = add_layer(dl2,200,200)
  dl4 = add_layer(dl3,200,1)
  return dl4

def compute_D_loss(D1, D2):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=D1, labels=tf.ones(tf.shape(D1))) , tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.zeros(tf.shape(D2)))

def compute_G_loss(D2):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.ones(tf.shape(D2)))

def train():
  mnist = input_data.read_data_sets("data_dir", one_hot = True)
  with tf.Session() as sess:
    with tf.name_scope('input'):
      x = tf.placeholder(tf.float32,[None,784], name='x-input')
      y = tf.placeholder(tf.float32,[None,10], name='y-input')
    with tf.variable_scope('model'):
      with tf.variable_scope('generator'):
        G = generator(x)
        G_params_num = len(tf.trainable_variables())
      with tf.variable_scope('discriminator'):
        D1 = discriminator(y)
      with tf.variable_scope('discriminator') as scope:
        scope.reuse_variables()
        D2 = discriminator(G)

    total_params = tf.trainable_variables()
    G_params = total_params[:G_params_num]
    D_params = total_params[G_params_num:]
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(G,1),tf.argmax(y,1)), tf.float32))
    tf.summary.scalar('acc', acc)
    with tf.variable_scope('Loss'):
      D1_loss, D2_loss = compute_D_loss(D1, D2)
      D_loss = tf.reduce_mean(D1_loss + D2_loss)
      G_loss = compute_G_loss(D2)
      tf.summary.scalar('D_real', tf.reduce_mean(D1_loss))
      tf.summary.scalar('D_fake', tf.reduce_mean(D2_loss))
      tf.summary.scalar('D_loss', tf.reduce_mean(D_loss))
      tf.summary.scalar('G_loss', tf.reduce_mean(G_loss))

    with tf.variable_scope('Trainer'):
      D_trainer = tf.train.GradientDescentOptimizer(FLAGS.D_learning_rate).minimize(D_loss)
      G_trainer = tf.train.GradientDescentOptimizer(FLAGS.G_learning_rate).minimize(G_loss)

      D_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/D', sess.graph)
      G_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/G', sess.graph)
      merged = tf.summary.merge_all()
      tf.global_variables_initializer().run()
      utils = Utils(sess, FLAGS.checkpoint_dir)
      if FLAGS.reload_model:
        utils.reload_model()

      xt,yt = mnist.test.next_batch(200)
      test = {x:2*xt-1, y:yt}
      for i in range(FLAGS.max_steps):
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        D_summary, _ , dloss, dd1 ,dd2 = sess.run([ merged, D_trainer, D_loss, D1_loss,D2_loss], feed_dict={x:2*xs-1, y:ys})
        G_summary, _ , gloss = sess.run([merged, G_trainer, G_loss], feed_dict={x:2*xs-1, y:ys})
        G_summary, _ , gloss = sess.run([merged, G_trainer, G_loss], feed_dict={x:2*xs-1, y:ys})
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
      test = {x:2*xt-1, y:yt}
      accuracy = sess.run(acc, test)
      print("After training, the accuracy is %g" % accuracy)

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    tf.app.run()