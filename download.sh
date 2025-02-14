# Download Hotpot Data
# wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O [Output file]
# wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json -O [Output file]
# wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O [Output file]

# Download GloVe
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
unzip glove.840B.300d.zip 

# Download Spacy language models
python3 -m spacy download en