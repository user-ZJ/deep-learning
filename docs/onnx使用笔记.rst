onnx使用笔记
==================

onnx保存数据到metadata
-----------------------------

.. code-block:: python 

    import onnx

    assert eval(str(cfg)) == cfg # yours metadata dict, class, whatever

    # storing
    model = onnx.load(onnx_path)
    meta = model.metadata_props.add()
    meta.key = "cfg"
    meta.value = str(cfg)
    onnx.save(onnx_model, onnx_path)

    # loading
    cfg = eval(onnx.load(onnx_path).metadata_props[0].value)