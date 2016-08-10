package com.simonj.nlp.classifier;

import java.text.DecimalFormat;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;

import com.google.common.collect.ImmutableList;
import com.simonj.nlp.classifier.Document.Split;

public final class Model {

    public final String topic;
    public final NaiveBayesModel model;
    public final double precision;
    public final double recall;
    public final double accuracy;
    public final double f1;

    public static final class Builder {

        private final ImmutableList.Builder<LabeledPoint> training = ImmutableList.builder();
        private final ImmutableList.Builder<LabeledPoint> test = ImmutableList.builder();

        /**
         * Adds a point to the model being built.
         */
        public void add(Split split, LabeledPoint point) {
            if (split == Split.TRAIN) {
                this.training.add(point);
            } else if (split == Split.TEST) {
                this.test.add(point);
            }
        }

        /**
         * Builds and evaluates the model.
         */
        public Model build(JavaSparkContext context, String topic) {
            NaiveBayesModel model = NaiveBayes.train(context.parallelize(training.build()).cache().rdd(), 1.0);

            int truePositives = 0, trueNegatives = 0, falsePositives = 0, falseNegatives = 0;
            for (LabeledPoint point : test.build()) {
                double predicted = model.predict(point.features());
                if (predicted == point.label()) {
                    if (predicted == 1.0) {
                        truePositives++;
                    } else {
                        trueNegatives++;
                    }
                } else {
                    if (predicted == 1.0) {
                        falsePositives++;
                    } else {
                        falseNegatives++;
                    }
                }
            }

            double precision = (double) truePositives / (truePositives + falsePositives);
            double recall = (double) truePositives / (truePositives + falseNegatives);
            double accuracy = (double) (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);
            double f1 = (double) 2 * truePositives / (2 * truePositives + falsePositives + falseNegatives);

            return new Model(topic, model, precision, recall, accuracy, f1);
        }
    }

    private Model(String topic, NaiveBayesModel model, double precision, double recall, double accuracy, double f1) {
        this.topic = topic;
        this.model = model;
        this.precision = precision;
        this.recall = recall;
        this.accuracy = accuracy;
        this.f1 = f1;
    }

    /**
     * Returns a string with model name and statistics for debugging and testing purposes.
     */
    public String toString() {
        return this.topic + " " + pft(this.precision) + " " + pft(this.recall) + " " + pft(this.accuracy) + " " + pft(this.f1);
    }

    private static final DecimalFormat DFT = new DecimalFormat("0.000");
    private static String pft(Double metric) {
        return DFT.format(metric);
    }
}
