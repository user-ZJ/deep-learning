neuralmagic使用笔记
=========================

bert模型压缩示例
-----------------------

https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/tutorials/sparsifying_bert_using_recipes.md

模型大小：
+----------------------------+----------+
|            模型            | 大小(MB) |
+============================+==========+
| bert-base-12layers         | 416      |
+----------------------------+----------+
| bert-base-12layers_prune80 | 416      |
+----------------------------+----------+

benchmark结果：
+----------------------------+----------+
|            模型            | 性能(ms) |
+============================+==========+
| bert-base-12layers         | 4427     |
+----------------------------+----------+
| bert-base-12layers_prune80 | 1291     |
+----------------------------+----------+

benchmark
-------------
deepsparse.benchmark zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none

