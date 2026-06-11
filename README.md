# Reddit Political Discourse Analysis

Scrapes, labels, and visualises Reddit activity around the 2023 Polish parliamentary elections. Covers two subreddits (r/Polska, r/PolskaPolityka), runs a full NLP pipeline (sentiment, topic modelling, NER, hate-speech heuristics), and builds a user-interaction network with community detection.

## Pipeline overview

```
scraper/download.py          # collect comments + submissions via PRAW
        │
        └─ data/activity_today_dist2.csv
                │
                ├─ analysis/sentiment_pl.py   # Polish BERT labelling
                │       └─ data/sentiment_analysis_big.csv
                │
                ├─ dashboards/activity.py     # activity + engagement overview
                └─ dashboards/nlp.py          # NER / hate-speech / linguistic patterns

analysis/topic_modeling.py   # LDA on labelled corpus, saves HTML to output/
network/build.py             # co-commenter graph → data/reddit_network.gml
network/topic_by_community.py  # per-community LDA
network/summary.py           # community overview dashboard
```

## Setup

```bash
# dashboards only
pip install -r requirements.txt

# full pipeline (scraper, sentiment labelling, topic modelling, network)
pip install -r requirements-dev.txt
```

The spaCy model (`pl_core_news_lg`) is installed automatically via the wheel URL in `requirements.txt`.

Copy `.env.example` to `.env` and fill in your Reddit API credentials:

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=reddit-political-discourse/1.0
```

## Running

**Collect data** — edit `scraper/subreddits.txt` and `scraper/users.txt`, then:

```bash
cd scraper && python download.py
```

The scraper is resumable: it skips users already present in `data/comments.csv`.

**Label sentiment** (required before the NLP dashboard):

```bash
cd analysis && streamlit run sentiment_pl.py
```

**Dashboards** (run from their own directory so `../data/` resolves correctly):

```bash
cd dashboards && streamlit run activity.py
cd dashboards && streamlit run nlp.py
```

**Network analysis** (run scripts in order):

```bash
cd network
python build.py              # builds reddit_network.gml
python communities.py        # community detection
python topic_by_community.py # per-community LDA, writes lda_topics_*.html
streamlit run summary.py
```

## NLP dashboard

Three-tab Streamlit app built on `data/sentiment_analysis_big.csv`:

- **NER** - named entity extraction with `pl_core_news_lg`, entity co-occurrence matrix. Manual lemma overrides handle foreign surnames the Polish model gets wrong (e.g. genitive "Tuska" → "Tusk").
- **Hate Speech & Disinformation** - regex heuristics: emotional lemma density, clickbait patterns, propaganda technique detection (generalisation, false dichotomy, appeal to fear). Aggregated into a `risk_score` per post.
- **Linguistic patterns** - MATTR (Moving Average TTR) for vocabulary richness, word-length distribution, Pearson correlation with upvote/reply counts, monthly trend lines.

Heavy computation is cached with `@st.cache_data` keyed on sample size.

## Network analysis

Builds an undirected graph where an edge connects a submission author to each user who commented on that post. Nodes with degree < 8 are pruned. Community detection runs on the pruned graph; each community then gets its own LDA model to surface distinct discussion themes.

## Limitations

The sentiment model (`eevvgg/bert-polish-sentiment-politics`) was fine-tuned on political text but has no neutral class — every post gets a positive or negative label, which inflates apparent polarity. The hate-speech detector is regex-based and has no ground-truth evaluation; treat its scores as a rough signal, not a classifier output. Network edges represent co-presence on a thread, not directed replies, so influence direction is lost.
