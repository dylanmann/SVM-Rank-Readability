cd /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09

find /mnt/castor/seas_home/m/mannd/CIS530/project/test -type f > ~/CIS530/project/filelistTest.txt
find /mnt/castor/seas_home/m/mannd/CIS530/project/train -type f > ~/CIS530/project/filelistTrain.txt

java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist ~/CIS530/project/filelistTest.txt -outputDirectory ~/CIS530/project/xml/test
java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist ~/CIS530/project/filelistTrain.txt -outputDirectory ~/CIS530/project/xml/train
