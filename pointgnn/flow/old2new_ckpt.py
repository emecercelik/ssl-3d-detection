import tensorflow as tf

path_name = "/kitti_root_tracking/flow_log_train_pretrained/model.ckpt"
new_path_name = "/kitti_root_tracking/flow_log_train_pretrained_new/model"

vars = tf.contrib.framework.list_variables(path_name)
with tf.Graph().as_default(), tf.Session().as_default() as sess:

  new_vars = []
  for name, shape in vars:
    v = tf.contrib.framework.load_variable(path_name, name)
    new_vars.append(tf.Variable(v, name="flownet3d/"+name))

  saver = tf.train.Saver(new_vars)
  sess.run(tf.global_variables_initializer())
  saver.save(sess, new_path_name)
