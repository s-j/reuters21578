package com.simonj.nlp.classifier;

import java.util.List;

import org.testng.Assert;
import org.testng.annotations.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.simonj.nlp.classifier.Document.Split;

/**
 * Unit tests for Tokenizer.
 */
@Test
public class TokenizerTest {

    public void testTokenization() {
        Document document = new Document(Split.NOT_USED, ImmutableSet.<String>of(), "hello bar");
        List<String> tokens = Tokenizer.getTokens(document);
        Assert.assertEquals(tokens, ImmutableList.<String>of("bar"));
    }
}
