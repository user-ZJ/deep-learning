Espnet使用笔记
=========================

使用提取的特征进行训练
--------------------------
本节以aishell使用hubert large预提取特征进行训练。`hubert预训练模型下载 <https://github.com/TencentGameMate/chinese_speech_pretrain>`_


hubert large特征提取
^^^^^^^^^^^^^^^^^^^^^^

使用以下脚本提取hubert特征

.. code-block:: python

    # compute_hubert_feature.py
    import torch
    import torch.nn.functional as F
    import soundfile as sf
    from fairseq import checkpoint_utils
    from s3prl.upstream.interfaces import Featurizer
    import numpy as np
    from pathlib import Path
    import os
    from kaldiio import WriteHelper
    import kaldiio
    try:
        import s3prl  # noqa
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
                    "s3prl is not installed, please git clone s3prl"
                    " (DO NOT USE PIP or CONDA) "
                    "and install it from Github repo, "
                    "by cloning it locally."
                )


    class hubert_feature:
        def __init__(self):
            model_path="/home/zhanjie/hubert-large/chinese-hubert-large.pt"
            self.device = 'cuda'
            s3prl_path = Path(os.path.abspath(s3prl.__file__)).parent.parent
            if not os.path.exists(os.path.join(s3prl_path, "hubconf.py")):
                raise RuntimeError(
                    "You probably have s3prl installed as a pip package, please uninstall it and then install it from")
            print("loading model(s) from {}".format(model_path))
            self.s3prl_upstream = torch.hub.load(
                s3prl_path,
                "hubert_local",
                ckpt=model_path,
                model_config=None,
                refresh=None,
                source="local",
                )
            self.s3prl_upstream = self.s3prl_upstream.to(self.device)
            self.s3prl_upstream.eval()
            feature_selection = "hidden_states"
            self.s3prl_featurizer = Featurizer(
                upstream=self.s3prl_upstream,
                feature_selection=feature_selection,
                upstream_device=self.device,
                )
            self.s3prl_featurizer = self.s3prl_featurizer.to(self.device)

        def get_hubert_feature(self,wav_path):
            wav, sr = sf.read(wav_path)
            feat = torch.from_numpy(wav).float()
            feats = feat.view(1, -1)
            print("feats.size:",feats.size())
            if feats.size()[1] == 0:
                return None

            with torch.no_grad():
                feats = feats.to(self.device)
                s3prl_feats = self.s3prl_upstream(feats)
                s3prl_feats = self.s3prl_featurizer(feats,s3prl_feats)
                return s3prl_feats[0].cpu().numpy()

        def get_hubert_feature_batch(self,wav_list):
            wav_datas = []
            for wav_path in wav_list:
                wav, sr = sf.read(wav_path)
                feat = torch.from_numpy(wav).float()
                feat = feat.to(self.device)
                wav_datas.append(feat)
                

            with torch.no_grad():
                s3prl_feats = self.s3prl_upstream(wav_datas)
                s3prl_feats = self.s3prl_featurizer(wav_datas,s3prl_feats)
                extract_feats = [ex.cpu().numpy() for ex in s3prl_feats]
                return extract_feats


    if __name__ == '__main__':
        hubert = hubert_feature()
        feat = hubert.get_hubert_feature("/home/zhanjie/data_aishell/wav/train/S0121/BAC009S0121W0330.wav")
        print(feat.shape)
        np.save("feat.npy",feat)
        feat = np.load("feat.npy")
        print(feat.shape)
        #with WriteHelper('ark:feat.ark',compression_method=2) as writer:
        #    writer("test", feat)
        kaldiio.save_mat("feat.ark",feat,compression_method=2)
        feats = hubert.get_hubert_feature_batch(["/home/zhanjie/data_aishell/wav/train/S0121/BAC009S0121W0330.wav"])
        print(feats[0].shape)

.. code-block:: python

    # aishell_hubert_feat.py
    import numpy as np
    import shutil
    import os
    import kaldiio
    from compute_hubert_feature import hubert_feature

    hubert = hubert_feature()
    batch_size = 16

    wav_list = []
    tgt_list = []
    with open("aishell_wav.flist") as f:
        for line in f.readlines():
            line = line.strip()
            tgt = line.replace("data_aishell/wav","hubertfeat").replace(".wav",".ark")
            tgtdir = os.path.dirname(tgt)
            if not os.path.exists(tgtdir):
                os.makedirs(tgtdir)
            if len(wav_list)<batch_size:
                wav_list.append(line)
                tgt_list.append(tgt)
            else:
                feats = hubert.get_hubert_feature_batch(wav_list)
                for i,feat in enumerate(feats):
                    if feat.shape[0] !=0:
                        kaldiio.save_mat(tgt_list[i], feat,compression_method=2)
                        print(tgt_list[i])
                wav_list = []
                tgt_list = []
        if len(wav_list) > 0:
            feats = hubert.get_hubert_feature_batch(wav_list)
            for i,feat in enumerate(feats):
                if feat.shape[0] !=0:
                    kaldiio.save_mat(tgt_list[i], feat,compression_method=2)

.. note:: 

    需要先安装espnet环境，并且在espnet根目录下执行
    ./tools/installers/install_s3prl.sh;
    ./tools/installers/install_fairseq.sh
    之后导入环境变量：
    export PYTHONPATH=/espnet:/espnet/fairseq:/espnet/s3prl:$PYTHONPATH

将提取好的hubertfeat目录存放在源数据的wav同级目录下

修改数据准备脚本
^^^^^^^^^^^^^^^^^^^^^
aishell数据准备脚本为：/espnet/egs2/aishell/asr1/local/data.sh

1. 不再生成wav.scp,改为生成feats.scp
2. 不再查找wav文件，而是查找ark文件


修改asr.sh
^^^^^^^^^^^^^^^
:: 

    将552行
    data/"${dset}"/cmvn.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp
    修改为
    data/"${dset}"/feats.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp

修改run_ssl.sh参数
^^^^^^^^^^^^^^^^^^^^^^^

::
    
    --audio_format kaldi_ark   
    --feats_type extracted 

