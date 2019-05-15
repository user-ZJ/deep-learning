# 神经网络训练问题记录

### 1.训练集loss下降，验证集loss上升
问题原因：使用data_generator.generator生产数据，在通过feeddict方式送入到神经网络训练，data_generator.generator在for step in totalstep之外调用，导致数据只生成了一次，一直用相同的数据进行训练，过拟合使训练集loss下降，验证集loss上升。

解决方法：在每个step调用data_generator.generator重新生成数据







## 参考
[训练神经网络的方法分享-Andrej Karpathy](https://mp.weixin.qq.com/s?__biz=MzIxNDgzNDg3NQ==&mid=2247485845&idx=1&sn=13620bb17dc0fd75d71100dd84cce59d&chksm=97a0c241a0d74b574645c0ed0d78d43f3ab921997020c45d2067422f3774ad7dc2b5d5e699b1&mpshare=1&scene=1&srcid=0515LTjlKVIcTa8sx2wpGePW&key=aba27b4d9f74947f0bdcea9e42c8111d94aa1f39e29bd917aea8513d1ee2bb092ddfff8ece4c3b5eb82790347899a24b28ee50d5f907284a0451b1b7b3f2940f5c27007195bce3ca4bc4a8be0a731a58&ascene=1&uin=MTAzNzg3MTgyMg%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=REzGp85uqicWco8SqVlEMdVWqSiu4chDpZil9UH%2BekmufbhYageGk%2B0lkTxF5hI3)   
