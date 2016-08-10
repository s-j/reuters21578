package com.simonj.nlp.classifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.simonj.nlp.classifier.Document.Split;

/**
 * A quick and dirty implementation partly stolen from org.apache.lucene.benchmark.utils.ExtractReuters.
 */
public final class Reuters21578 {

    private static final Pattern LEWIS_PATTERN = Pattern.compile("LEWISSPLIT=\"(.*?)\"");
    private static final Pattern TOPICS_PATTERN = Pattern.compile("<TOPICS>(.*?)</TOPICS>");
    private static final Pattern LISTING_PATTERN = Pattern.compile("<D>(.*?)</D>");
    private static final Pattern TEXT_PATTERN = Pattern.compile("<TITLE>(.*?)</TITLE>|<BODY>(.*?)</BODY>");
    private static final String[] META_CHARS = { "&", "<", ">", "\"", "'" };
    private static final String[] META_CHARS_SERIALIZATIONS = { "&amp;", "&lt;", "&gt;", "&quot;", "&apos;" };

    private Reuters21578() { }

    /**
     * Extracts all documents in the given directory.
     */
    public static List<Document> extract(File dir) {
        Preconditions.checkArgument(dir.exists(), "Path " + dir + " does not exist");
        File[] sgmFiles = dir.listFiles(new FileFilter() {
            public boolean accept(File file) {
                return file.getName().endsWith(".sgm");
            }
        });
        Preconditions.checkArgument(sgmFiles != null && sgmFiles.length > 0, "No .sgm files in " + dir);

        ImmutableList.Builder<Document> builder = ImmutableList.builder();
        StringBuilder buffer = new StringBuilder(1024);
        for (File sgmFile : sgmFiles) {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(sgmFile));
                String line = null;
                while ((line = reader.readLine()) != null) {
                    if (line.indexOf("</REUTERS") == -1) {
                        buffer.append(line).append(' ');
                    } else {
                        builder.add(extract(buffer));
                        buffer.setLength(0);
                    }
                }
                reader.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        return builder.build();
    }

    /**
     * Extracts the document from the given character sequence.
     */
    public static Document extract(CharSequence input) {
        String lewisSays = Iterables.getOnlyElement(getAllMatches(LEWIS_PATTERN, input));
        Split split = lewisSays.equals("TRAIN") ? Split.TRAIN : lewisSays.equals("TEST") ? Split.TEST : Split.NOT_USED;
        Set<String> topics = getAllMatches(LISTING_PATTERN,
                Iterables.getOnlyElement(getAllMatches(TOPICS_PATTERN, input), ""));
        String content = Joiner.on(". ").join(getAllMatches(TEXT_PATTERN, input));

        // Replace all meta characters, tabs and other noise.
        for (int i = 0; i < META_CHARS_SERIALIZATIONS.length; i++) {
            content = content.replaceAll(META_CHARS_SERIALIZATIONS[i], META_CHARS[i]);
        }
        content = content.replaceAll("\t", " ").replaceAll("( )+", " ").replace("REUTER &#3;", "").replace("Reuter &#3;", "");

        return new Document(split, topics, content);
    }

    private static Set<String> getAllMatches(Pattern pattern, CharSequence input) {
        ImmutableSet.Builder<String> builder = ImmutableSet.builder();
        Matcher matcher = pattern.matcher(input);
        while (matcher.find()) {
            for (int i = 1; i <= matcher.groupCount(); i++) {
                if (matcher.group(i) != null) {
                    builder.add(matcher.group(i));
                }
            }
        }
        return builder.build();
    }
}
