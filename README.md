# Implementing the fruit fly's similarity hashing

This is an implementation of the random indexing method described by Dasgupta et al (2017) in [A neural algorithm for a fundamental computing problem](http://science.sciencemag.org/content/358/6364/793/tab-figures-data).

### Description of the data

Your data/ directory contains two small semantic spaces:

- one from the British National Corpus, containing lemmas expressed in 4000 dimensions.
- one from a subset of Wikipedia, containing words expressed in 1000 dimensions.

The cells in each semantic space are normalised co-occurrence frequencies *without any additional weighting* (PMI, for instance).

The directory also contains test pairs from the [MEN similarity dataset](https://staff.fnwi.uva.nl/e.bruni/MEN), both in lemmatised and natural forms.

Finally, it contains a file *generic_pod.csv*, which is a compilation of around 2400 distributional web page signatures, in [PeARS](http://pearsearch.org) format. The web pages span various topics: Harry Potter, Star Wars, the Black Panther film, the Black Panther social rights movement, search engines and various small topics involving architecture.

### Running the fruit fly code

To run the code, you need to enter the corpus you would like to test on, and the number of Kenyon cells you are going to use for the experiment. For instance, for the BNC space:

    python3 projection.py bnc 8000 6 5

Or for the Wikipedia space:

    python3 projection.py wiki 4000 4 10

The program returns the Spearman correlation with the MEN similarity data, as calculated a) from the raw frequency space; and b) after running the fly's random projections.


### Tuning parameters

First, get a sense for which parameters give best results on the MEN dataset, for both BNC and Wikipedia data. If you know how to code, you can do a random parameter search automatically. If not, just try different values manually and write down what you observe.


### Analysing the results

Compare results for the BNC and the Wikipedia data. You should see that results on the BNC are much better than on Wikipedia. Why is that?

To help you with the analysis, you can print a verbose version of the random projections with the -v flag. E.g.:

    python3 projection.py bnc 8000 6 1 -v

This will print out the projection neurons that are most responsible for the activation in the Kenyon layer.


### Turning the fly into a search engine

You can test the capability of the the fly's algorithm to return web pages that are similar to a given one (and crucially, dimensionality-reduced), by typing:

    python3 searchfly.py data/generic_pod.csv 2000 6 5 https://en.wikipedia.org/wiki/Yahoo!
