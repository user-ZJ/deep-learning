
create_tfrecords_fbank:
	python data_generator/create_tfrecords.py --dir_path E:\训练数据包\VCTK-Corpus --num_labels 109 --features logmel --mels 64 --context 64 --stride 3 --processes 4
	
create_tfrecords_mfcc:
	python data_generator/create_tfrecords.py --dir_path E:\训练数据包\VCTK-Corpus --num_labels 109 --features mfcc --mfccs 20 --context 60 --stride 3 --processes 4