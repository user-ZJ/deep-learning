import tensorflow as tf
# tf.name_scope()主要是用来管理命名空间的，这样子让我们的整个模型更加有条理
# 不会影响用 get_variable()创建的变量，但会影响通过 Variable()创建的变量
with tf.name_scope('nsc1'):
    var_1 = tf.Variable(initial_value=[0], name='var_1') # nsc1/var_1:0
    var_1 = tf.Variable(initial_value=[0], name='var_1') # nsc1/var_1_1:0
    var_2 = tf.get_variable(name='var_2', shape=[1, ]) # var_2:0
# tf.variable_scope() 的作用是为了实现变量共享，它和 tf.get_variable() 来完成变量共享的功能
with tf.variable_scope('vsc1'):
    var_3 = tf.Variable(initial_value=[0], name='var_3') # vsc1/var_3:0
    var_4 = tf.get_variable(name='var_4', shape=[1, ]) # vsc1/var_4:0
    # var_4 = tf.get_variable(name='var_4', shape=[1, ]) #报错：Variable vsc1/var_4 already exists, disallowed
#当 reuse 设置为 True 或者 tf.AUTO_REUSE 时，表示这个scope下的变量是重用的或者共享的，
#当 reuse 设置为 True时，说明这个变量以前就已经创建好了。但如果这个变量以前没有被创建过，则在tf.variable_scope下调用tf.get_variable创建这个变量会报错
#当 reuse 设置为tf.AUTO_REUSE 时，不会报错。
with tf.variable_scope('vsc1',reuse=True):
    var_4 = tf.get_variable(name='var_4', shape=[1, ])
    # var_5 = tf.get_variable(name='var_5', shape=[1, ]) #报错：Variable vsc1/var_5 does not exist, or was not created with tf.get_variable()
with tf.variable_scope('vsc1',reuse=tf.AUTO_REUSE):
    var_5 = tf.get_variable(name='var_5', shape=[1, ]) # vsc1/var_5:0
with tf.variable_scope('vsc1') as scope:
    scope.reuse_variables() #相当于reuse=True
    # var_6 = tf.get_variable(name='var_6', shape=[1, ]) #报错：Variable vsc1/var_5 does not exist, or was not created with tf.get_variable()
# 打印所有变量
vs = tf.trainable_variables()
for v in vs:
    print(v)
