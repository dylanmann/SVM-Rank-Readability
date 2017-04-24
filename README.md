# CIS530 Final Project Spring 2017

Dylan Mann, Pia Kochar
mannd, pkochar

## [You can view the final report on our findings here](Report.pdf)

Instructions to run the code:

To run our code, you need to first convert your excerpts into one excerpt per file (this can be done using a variation of the function `convert_to_files()`, and then put them into the proper location and edit the output_dir variable in `run_corenlp.sh` to use the proper directories.  Then you run that script from anywhere on eniac or the biglab machines.  Then you must run the two functions.

generate_svm_rank_train() and
generate_svm_rank_test()

To run these, you need to edit a few parameters:

the location of your xml files (default "xml/train" and "xml/test")
the name of the output files (default "train.dat" and "test.dat")

Then you can run the python program using python3 to produce the test and train data files.

Afterwards you need to download or build a copy of svm_rank here:
https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html

Once you have the executable for your system, you train it using the command
`svm_rank_learn -c 10 train.dat model.dat`

and classify the test data with the command:
`svm_rank_classify test.dat model.dat predictions.txt`

and your 50 predicted values will be in the proper format in predictions.txt
