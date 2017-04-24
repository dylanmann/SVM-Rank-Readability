cd /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09

output_dir = <dirname>

mkdir -p "$output_dir/filelistTest.txt"
mkdir -p "$output_dir/filelistTrain.txt"
mkdir -p "$output_dir/xml/test"
mkdir -p "$output_dir/xml/train"

find /mnt/castor/seas_home/m/mannd/CIS530/project/test -type f > "$output_dir/filelistTest.txt"
find  /mnt/castor/seas_home/m/mannd/CIS530/project/train -type f > "$output_dir/filelistTrain.txt"

java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist "$output_dir/filelistTest.txt" -outputDirectory "$output_dir/xml/test"
java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist "$output_dir/filelistTrain.txt" -outputDirectory "$output_dir/xml/train"
