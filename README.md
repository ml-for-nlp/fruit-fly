# Implementing the fruit fly's similarity hashing

This is an implementation of the random indexing method described by Dasgupta et al (2017) in [A neural algorithm for a fundamental computing problem](http://science.sciencemag.org/content/358/6364/793/tab-figures-data). The code takes a raw count matrix and outputs locality-sensitive binary hashes in place of the original vectors.

**Before you run any analysis, make sure you understand what the algorithm does to the input, referring to the paper.**

### Description of the data

Your data/ directory contains two small semantic spaces:

- one from the British National Corpus, containing lemmas expressed in 4000 dimensions.
- one from a subset of Wikipedia, containing words expressed in 1000 dimensions.

The cells in each semantic space are normalised co-occurrence frequencies *without any additional weighting* (PMI, for instance).

The directory also contains test pairs from the [MEN similarity dataset](https://staff.fnwi.uva.nl/e.bruni/MEN), both in lemmatised and natural forms.


### Running the fruit fly code

To run the code, you need to enter the corpus you would like to test on, and the number of Kenyon cells you are going to use for the experiment, as well as the projection size and WTA rate (see paper for details). For instance, for the BNC space:

    python3 -W ignore run_fly.py --dataset=bnc --kcs=500 --wta=5 --proj_size=10

Or for the Wikipedia space:

    python3 -W ignore run_fly.py --dataset=wiki --kcs=500 --wta=5 --proj_size=10

The program returns the Spearman correlation with the MEN similarity data, as calculated a) from the raw frequency space; and b) after running the fly's random projections.


### Initial analysis

**Tuning parameters:** First, get a sense for which parameters give best results on the MEN dataset, for both BNC and Wikipedia data. If you know how to code, you can do a random parameter search automatically. If not, just try different values manually and write down what you observe.

**Preliminary results:** Compare results for the BNC and the Wikipedia data. You should see that results on the BNC are better than on Wikipedia. Why is that? 


### Expanding the code

We have talked about the problems related to high-dimensional data. The PN layer of the fruit fly is rather large. Can we fix this? The *utils.py* file contains a function *run_PCA* to run PCA over a matrix and reduce its dimensionality to  *k*  principal components. Integrate it in the main fruit fly code to check its effect on the PN layer and the final results. Experiment with different numbers of components.


### Open-ended project

Can you use the fly to hash the content of Web documents? Use the code of the [search engine practical] to get document vectors from Wikipedia, and hash them with the fruit fly. Do you get similar hashes for documents of a given category?

PS: if you want to see how the fruit fly is used in a real search project, check out the [PeARS](https://github.com/PeARSearch/PeARS-fruit-fly/wiki) framework, which is currently integrating fruit fly hashing in its framework.
