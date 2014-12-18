SVM Email Spam Filter
=====================

This is the top-level folder for a challenge problem dealing with the use of
support vector machine (SVM) algorithm used to implement a spam classifier.

The original idea for this code comes from a Machine Learning Coursera course
taught by Andrew Ng, accessed in 2014 at https://www.coursera.org/course/ml.
The training and test data are taken from his homework example in this course.
He wrote the code in Octave and I changed it to C as well as altered the CPU
and GPU algorithms to more closely align with the algorithm described in [1],
labeled as "Algorithm 1" on page 105, where the working set choice is using the
first order heuristic, also described in [1] and a linear kernel is used.  The
general algorithm is the SMO algorithm from Platt [2].

The training set is 4000 emails and there are 1899 features (keywords).  This
is admittedly a reduced training set and reduced feature size for illustration
purposes only.

[1] B. C. Catanzaro, N. Sundaram, K. Keutzer, "Fast Support Vector Machine
Training and Classification on Graphics Processors", Proceedings of the 25th
International Comference on Machine Learning, Helsinki, Finland, 2008.

[2] J. C. Platt, "Fast training of support vector machines using sequential
minimal optimization", Advances in kernel methods: support vector learning,
Cambridge, MA, USA: MIT Press.

Instructions
------------

To run the code which trains and then classifies email as spam please do the 
following steps.

1.) Build the code.  Ensure that NVCC is in your path.

> make

2.) Choose an email to be tested.  There is one genuine email and three spam
emails to choose from.  If you wish to test your own email (either genuine or
spam) put your email as a text file in this directory.  Copy/paste only the 
text of the email.  Please omit the header information as this spam 
classifier only cares about the text of the email.

3.) Process the email.  The email text needs to be processed by stripping out 
all non-text elements and then running a stemming algorithm on each resultant
word.  This leaves you with just a tokenized email of stemmed words which is 
easier to process.  When you run this command the stemmed email will be 
printed to the screen and a file called "emailVector.txt" will be created
which will be a vector of 0's and 1's depending on whether that specific 
feature (word) exists in the email or not.

> sh processEmail.sh <emailTextfile.txt>

4.) Train the SVM and classify your email.  In this step the SVM will be first
be trained against a training set of size 4000.  Then it will be tested for
accuracy against this set.  Then it will be tested against a test set of size
1000.  Both of these accuracies should be over 98%.  Finally the SVM will 
classify your input email and either classify it as spam (1) or NOT spam (0).

> ./x.train
Prediction success rate on training set is 99.750000
Prediction success rate on test set is 98.200000
Email test results 1 is SPAM 0 is NOT SPAM
File Name emailVector.txt, classification 0 NOT SPAM
