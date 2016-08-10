package com.simonj.nlp.classifier;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.simonj.nlp.classifier.Document.Split;

/**
 * A simple text classifier based on the NaiveBayes implementation in Spark.
 */
public final class TextClassifier {

    private TextClassifier() { }

    /**
     * Returns a subset of topics that has both test and training data.
     */
    protected static Set<String> getTopicsForClassification(List<Document> documents) {
        ImmutableSet.Builder<String> training = ImmutableSet.builder();
        ImmutableSet.Builder<String> test = ImmutableSet.builder();
        for (Document document : documents) {
            if (document.split == Split.TRAIN) {
                training.addAll(document.topics);
            } else if (document.split == Split.TEST) {
                test.addAll(document.topics);
            }
        }
        return Sets.intersection(training.build(), test.build());
    }

    /**
     * Extracts the features and builds classification models, returning only those with F1 score above the threshold.
     * FIXME The method needs refactoring.
     */
    protected static List<Model> getClassificationModels(List<Document> documents, double f1Threshold) {
        // Remove all unused cases and create an empty set of model builders for selected topics.
        documents = documents.stream().filter(document -> document.split != Split.NOT_USED).collect(Collectors.toList());
        Set<String> topics = getTopicsForClassification(documents);
        Map<String, Model.Builder> modelBuilders = Maps.newHashMap();
        for (String topic : topics) {
            modelBuilders.put(topic, new Model.Builder());
        }

        // Tokenize data for building.
        HashingTF tf = new HashingTF(65536);
        for (Document doc : documents) {
            Vector vector = tf.transform(Tokenizer.getTokens(doc));
            for (String topic : topics) {
                Model.Builder modelBuilder = modelBuilders.get(topic);
                double label = doc.topics.contains(topic) ? 1.0 : 0.0;
                modelBuilder.add(doc.split, new LabeledPoint(label, vector));
            }
        }

        // Now build, evaluate and prune the models.
        Logger.getLogger("org").setLevel(Level.WARN);
        Logger.getLogger("akka").setLevel(Level.WARN);

        ImmutableList.Builder<Model> resultBuilder = ImmutableList.builder();
        try (JavaSparkContext context = new JavaSparkContext("local[2]", "TextClassifier")) {
            for (Entry<String, Model.Builder> e : modelBuilders.entrySet()) {
                Model model = e.getValue().build(context, e.getKey());
                if (model.f1 > f1Threshold) {
                    resultBuilder.add(model);
                    // FIXME If desired it is possible to store the models using the following line
                    // model.model.save(context.sc(), "target/tmp/" + model.topic + "_NB");
                }
            }
            context.stop();
        }

        return resultBuilder.build();
    }

    public static void main(String[] args) {
        Preconditions.checkArgument(args.length == 1, "Wrong number of arguments, usage: java -cp <...> com.simonj.nlp.classifier.TextClassifier <path to Reuters>");
        List<Document> documents = Reuters21578.extract(new File(args[0]));
        List<Model> models = getClassificationModels(documents, 0.1);

        for (Model m : models) {
            System.out.println(m);
        }
    }
}
