# Reuters21578 Classification

This project applies a Naive Bayes implementation from Spark at [the Reuters21578 dataset](https://datahub.io/dataset/reuters­21578). The program uses categories found in TOPICS and classifications found in LEWISSPLIT along with the article title and content, and it can be executed as follows:

    cd <path to the classifier directory>
    export JAVA_CLASSPATH=”<path to Java8 directory>”
    mvn clean package
    java -cp target/classifier-0.0.1-SNAPSHOT-allinone.jar com.simonj.nlp.classifier.TextClassifier <path to the reuters21578 dir>

## Implementation details
Under the hood, the program will do the following:

1.  Go through the SGML files in the Reuters directory and extract the split decision, topic annotation and the article content (that is the title and body text).
2.  Clean, normalize and tokenize the text, eliminate stop-words and stem the tokens to be used as classification features.
3.  For each topic that has both TRAIN and TEST data, build and evaluate a classification model, keeping only those with the F1 score above 0.1. Currently, the program only prints the evaluation results for the remaining models, but it can be easily modified to store these models, or extended to use them for further classification of other text.

In the remainder, I briefly explain the approach chosen for each step, the current limitations and possible improvements.

### Parsing the dataset
Parsing is implemented in `Reuters21578` and a test example can be found in `Reuters21578Test`, this step returns a set of `Document` instances. This is the part of the code I am least satisfied with. After spending some time looking for an existing Java library and trying to use a DOM parser, I went for the simplest solution inspired by Lucene’s `ExtractReuters`, that is using a set of regular expressions to pull out and clean chunks of text from the article body and title. So to say, the implementation has a huge potential for improvement. As well as it could be interesting to see whether using text from the other fields, such as the dateline, could improve the classification performance. This code can also be extended by for example providing a parser for any other dataset containing topics, split decisions and content.

### Tokenization
Tokenization is implemented in `Tokenizer` and a test example can be found in `TokenizerTest`. Here I use the Lucene’s `EnglishAnalyzer` with a hard-coded list of English stop-words. I am very happy with this decision, as it was quite easy to implement. This step takes the document content text and returns a list of fully processed tokens.

### Feature extraction and model
The remaining part is implemented in `TextClassifier` and `Model`. The former of these two first uses `Reuters21678` on the provided directory path, throws out the documents with the NOT USED decision, applies `Tokenizer` on the content of the remaining documents and uses Spark’s `HashingTF` with size 65536 to generate the feature vectors. My intuition tells that `HashingTF` simply hashes each token to a number in the specified range (the feature index) and uses its frequency in the document (the feature value).

Then, `TextClassifier` finds a subset of topics having both TRAIN and TEST decisions and create a `Model.Builder` for each. The latter is used to collect the `LaledPoint`’s for training. Thereafter, `JavaSparkContext` with a local master is used to train the models using `NaiveBayes` and collect evaluation metrics including precision, recall, accuracy and F1.

Finally, `TextClassifier` discards all models with F1 below 0.1 and print the evaluation metrics for the remaining models, which gives be the following:

|topic | precision | recall | accuracy | F1 |
|------|-----------|--------|----------|----|
| corn | 0.556 | 0.357 | 0.992 | 0.435 |
| ship | 0.778 | 0.236 | 0.988 | 0.362 |
| acq | 0.627 | 0.808 | 0.922 | 0.706 |
| livestock | 0.222 | 0.083 | 0.995 | 0.121 |
| grain | 0.785 | 0.685 | 0.988 | 0.731 |
| veg-oil | 0.667 | 0.162 | 0.995 | 0.261 |
| cocoa | 1.000 | 0.056 | 0.997 | 0.105 |
| sugar | 0.538 | 0.194 | 0.994 | 0.286 |
| crude | 0.842 | 0.677 | 0.986 | 0.751 |
| money-fx | 0.700 | 0.583 | 0.981 | 0.636 |
| interest | 0.812 | 0.293 | 0.983 | 0.431 |
| wheat | 0.704 | 0.535 | 0.992 | 0.608 |
| soybean | 0.769 | 0.303 | 0.996 | 0.435 |
| oilseed | 0.647 | 0.234 | 0.993 | 0.344 |
| dlr | 0.800 | 0.091 | 0.993 | 0.163 |
| trade | 0.596 | 0.265 | 0.983 | 0.367 |
| earn | 0.680 | 0.938 | 0.911 | 0.788 |
| coffee | 0.667 | 0.286 | 0.996 | 0.400 |

Note that `TextClassifier` currently lacks unit tests and an end-to-end test, which will be added at some point in future.
