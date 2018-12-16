import tensorflow as tf

# # 创建一个随机队列，队列最大长度为10，出队后最小长度为2
# q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")
# run_options = tf.RunOptions(timeout_in_ms = 10000)  # 等待10秒
# sess = tf.Session()
# for i in range(0, 10): #12次入队,会产生阻塞
#   try:
#     sess.run(q.enqueue(i),options=run_options)
#   except tf.errors.DeadlineExceededError:
#     print('out of range')
#
# for i in range(0, 8): # 8次出队
#   print(sess.run(q.dequeue()))

# # 创建一个含有队列的图
# q = tf.FIFOQueue(1000, "float")
# counter = tf.Variable(0.0)    # 计数器
# increment_op = tf.assign_add(counter, tf.constant(1.0))    # 操作：给计数器加1
# enqueue_op = q.enqueue([counter]) # 操作：计数器值加入队列
# # 创建一个队列管理器 QueueRunner，用这两个操作向队列 q 中添加元素。
# qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)
# #主线程
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   enqueue_threads = qr.create_threads(sess, start=True)  # 启动入队线程
#   #主线程
#   for i in range(10):
#     print (sess.run(q.dequeue()))



# 创建一个含有队列的图
q = tf.FIFOQueue(1000, "float")
counter = tf.Variable(0.0)    # 计数器
increment_op = tf.assign_add(counter, tf.constant(1.0))    # 操作：给计数器加1
enqueue_op = q.enqueue([counter]) # 操作：计数器值加入队列
# 创建一个队列管理器 QueueRunner，用这两个操作向队列 q 中添加元素。
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)

# 主线程
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Coordinator：协调器，协调线程间的关系可以视为一种信号量，用来做同步
coord = tf.train.Coordinator()
# 启动入队线程，协调器是线程的参数
enqueue_threads = qr.create_threads(sess, coord = coord,start=True)

# 使用方式1
# 主线程
for i in range(0, 10):
  print(sess.run(q.dequeue()))
# coord.request_stop()# 通知其他线程关闭
# coord.join(enqueue_threads) # join操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回

# 使用方式2
coord.request_stop()# 通知其他线程关闭
# 主线程
for i in range(0, 10):
  try:
    print(sess.run(q.dequeue()))
  except tf.errors.OutOfRangeError:
    break
coord.join(enqueue_threads) # join操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回
