wget -P data/vectors/ http://nlp.stanford.edu/data/glove.6B.zip
unzip -j data/vectors/glove.6B.zip data/vectors/glove.6B.200d.txt
echo -e "400000 200\n$(cat data/vectors/glove.6B.200d.txt)" > data/vectors/glove.6B.200d.txt
wget -O data/vectors/law2vec.200d.txt https://archive.org/download/Law2Vec/Law2Vec.200d.txt

wget -O data/datasets/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip data/datasets/datasets.zip -d data/datasets/EURLEX57K
rm data/datasets/datasets.zip
rm -rf data/datasets/EURLEX57K/__MACOSX
mv data/datasets/EURLEX57K/dataset/* data/datasets/EURLEX57K/
rm -rf data/datasets/EURLEX57K/dataset
wget -O data/datasets/EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
