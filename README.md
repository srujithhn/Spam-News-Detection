# Fake News Detection (TF-IDF + Naive Bayes)

A text classifier that labels a news article as real or fake, using classic NLP: cleaning, lemmatisation, TF-IDF features and a Multinomial Naive Bayes model.

## What I did

- **Data:** the Kaggle fake and real news dataset (`True.csv` and `Fake.csv`), 44,898 articles: 21,417 real and 23,481 fake. The CSVs aren't in this repo; download them from Kaggle.
- **Cleaning:** lowercased text, removed everything except letters, dropped English stopwords and lemmatised words (NLTK WordNet).
- **Features:** TF-IDF with single words and word pairs, top 5,000 features.
- **Model:** Multinomial Naive Bayes, trained on 24,000 articles with 6,000 held out (80/20 split of the first 30,000 shuffled articles).
- **Result:** 94.6% accuracy on the training set.

## What I'd fix now (honest review)

Looking back at this notebook with what I've since learned about evaluation:

1. **It never measures test accuracy.** It reports training accuracy (94.6%), which says how well the model memorised what it saw, not how well it works on new articles.
2. **The test set gets its own vocabulary.** The TF-IDF vectoriser is fitted again on the test data instead of reusing the one fitted on training data, so test features don't line up with what the model learned. The fix is `vectorize.transform(test_data)`.
3. **The model may be learning the source, not the truth.** If most real articles come from one outlet, a model can score well just by spotting that outlet's writing style. I'd check this by removing source names and datelines and testing on articles from other outlets.

These are the same lessons I applied later when I built and tested an AI app: measure on data the model hasn't seen, and check what it's really relying on. See [DocAgent Lite](https://github.com/srujithhn/docagent-lite).

## Run it

Open `Spam news detection.ipynb` in Jupyter or Google Colab with pandas, NumPy, NLTK and scikit-learn installed, and put `True.csv` and `Fake.csv` next to it.

Built by [Srujith H N](https://www.linkedin.com/in/srujith-h-n-b7b000267).
