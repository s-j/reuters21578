package com.simonj.nlp.classifier;

import java.util.Set;

/**
 * A simple classification document with a split decision, a set of topics and a content.
 */
public class Document {

    public static enum Split {
        TEST, TRAIN, NOT_USED
    }

    public Split split;
    public Set<String> topics;
    public String content;

    public Document(Split split, Set<String> topics, String content) {
        this.split = split;
        this.topics = topics;
        this.content = content;
    }
}
