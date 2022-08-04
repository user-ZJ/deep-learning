# 模型压缩

## 1. 模型压缩技术

1. 模型小型化：使用一个小模型处理相同的任务。使用的技术有模型蒸馏和模型结构搜索。

2. 算子分解：3x3卷积可以分解为3x1和1x3的卷积。

3. 量化：int8量化等

4. 模型压缩

5. 参数共享：

6. 稀疏化



## 2. 数据无关的模型压缩

a. 根据参数重要性进行剪枝

    训练模型 -> 裁剪靠近0的参数 -> 重新训练模型







facebook [bit goes down](https://www.toutiao.com/a6718568007041286660/?tt_from=weixin&utm_campaign=client_share&wxshare_count=1&timestamp=1564355223&app=news_article&utm_source=weixin&utm_medium=toutiao_android&req_id=201907290707020100250660727577982&group_id=6718568007041286660)  
[Mix net](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247490670&idx=2&sn=c6e261c8f55f22f8307251c2eb4e963b&chksm=f9a26ee1ced5e7f7f4ac9bc0d78db30ba1e46bc77c8bba5db07dccdc7ffadbf8d8210f81d570&mpshare=1&scene=1&srcid=&sharer_sharetime=1564362843068&sharer_shareid=42a896371dfe6ebe8cc4cd474d9b747c&key=7e5df62132e2da6fef592f387df08beb00e5554b83e32a4a7cba34944ccc4fa9ddb9d786cd847795dfbec4192d996b2bbef91658a211b5a83f2fd5c452c439dfd42559ab0610b153f427c096b041b11a&ascene=1&uin=MTAzNzg3MTgyMg%3D%3D&devicetype=Windows+10&version=62060834&lang=zh_CN&pass_ticket=QhSpPlCmib%2BdHWizDHxp4GdgNFD5n0xU5wmcY4MuBMBqlONOO8gh7BJitUpjAbcS)  

## 参考

https://arxiv.org/abs/2102.00554