import tensorflow as tf
'''
tf.InteractiveSession()是一种交互式的session方式，
它让自己成为了默认的session，也就是说用户在不需要指明用哪个session运行的情况下，就可以运行起来, 
这样的话就是run()和eval()函数可以不指明session
'''

# InteractiveSession
a=tf.constant([[1., 2., 3.],[4., 5., 6.]])
b=tf.Variable(tf.random_normal(shape=[3,2]))
c=tf.matmul(a,b)
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
print(c.eval())

# Session
a=tf.constant([[1., 2., 3.],[4., 5., 6.]])
b=tf.Variable(tf.random_normal(shape=[3,2]))
c=tf.matmul(a,b)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(c))
# 或者
tf.global_variables_initializer().run(session=sess)
print(c.eval(session=sess))
