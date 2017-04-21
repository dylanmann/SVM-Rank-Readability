cd /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09

mkdir -p ~/CIS530/test
mkdir -p ~/CIS530/train

find /home1/c/cis530/hw3/data/test -type f > ~/CIS530/filelistTest.txt
find /home1/c/cis530/hw3/data/train -type f > ~/CIS530/filelistTrain.txt

java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP-annotators tokenize,ssplit,pos,lemma,ner,parse -filelist ~/CIS530/filelistTest.txt -outputDirectory ~/CIS530/test
java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP-annotators tokenize,ssplit,pos,lemma,ner,parse -filelist ~/CIS530/filelistTrain.txt -outputDirectory ~/CIS530/train
