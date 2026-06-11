"""
Dashboard: NER · Hate Speech & Disinformation · Linguistic patterns
Data: Polish subreddits (r/Polska, r/PolskaPolityka)

Run:
    .venv/bin/streamlit run nlp.py
"""
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.express as px
import spacy
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Base (lemma) forms - spaCy lemmatises "idiotą" → "idiota" before matching
EMOTIONAL_WORDS = {
    "idiota", "głupek", "kłamca", "złodziej", "bandyta", "zdrajca",
    "szaleniec", "manipulator", "oszust", "hipokryta", "kreatura",
    "skandal", "katastrofa", "tragedia", "nienawiść", "wstyd",
    "hańba", "żenada", "patologia", "bezczelny", "perfidny", "debil",
    "imbecyl", "głupota", "bzdura", "brednia", "propaganda", "kłamstwo",
}

CLICKBAIT_PATTERNS = [
    r"nie uwierzysz", r"prawda o", r"szokuj\w*", r"pilne[!:\s]",
    r"tylko u nas", r"koniec \w+", r"ujawniamy", r"wiemy już",
    r"musisz wiedzieć", r"skandaliczne", r"\bszok\b", r"ekskluzywnie",
    r"tajemnica \w+", r"ukrywają przed",
]
_clickbait_re = re.compile("|".join(CLICKBAIT_PATTERNS), re.IGNORECASE)

PROPAGANDA_PATTERNS = {
    "Fear (appeal to fear)": [
        r"zagrożen\w+", r"katastrofa\w*", r"koniec \w+",
        r"zniszczy\w*", r"zagłada\w*", r"niebezpiecz\w+",
    ],
    "Whataboutism": [
        r"a co z\b", r"a wy to", r"a poprzedni\w*",
        r"za poprzedni\w*", r"też tak robili",
    ],
    "Generalisation": [
        r"\bwszyscy\b", r"\bzawsze\b", r"\bnigdy\b",
        r"\bkażdy\b", r"\bwszystkie\b", r"\bżaden\b",
    ],
    "False dichotomy": [
        r"albo .{1,30}? albo", r"kto nie jest z nami",
        r"z nami albo", r"tylko dwie opcje",
    ],
}
_propaganda_res = {
    name: re.compile("|".join(pats), re.IGNORECASE)
    for name, pats in PROPAGANDA_PATTERNS.items()
}

LABEL_NAMES = {
    "persName":  "Person",
    "orgName":   "Organisation",
    "placeName": "Place",
    "geogName":  "Geography",
    "date":      "Date",
    "time":      "Time",
}

SENTIMENT_LABELS = {"POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral"}

# Polish acronyms spaCy lowercases - restore them for display
_KNOWN_UPPER = {"ue", "pis", "tvp", "psl", "nbp", "pko", "krs", "msz", "msp", "tk", "nato"}

# spaCy's Polish model fails to lemmatize foreign surnames and some place names -
# map bad lemmas to their correct base (nominative) form.
_ENTITY_NORMALIZE: dict[str, str] = {
    "tuska": "tusk", "tuskiem": "tusk", "tuski": "tusk", "tuskie": "tusk",
    "polskę": "polska", "polskiej": "polska", "polskim": "polska", "polskie": "polska",
    "polaków": "polak", "polakami": "polak", "polakom": "polak",
    "niemiec": "niemcy", "niemcom": "niemcy", "niemcami": "niemcy",
    "unii": "unia", "unię": "unia",
    "sejmie": "sejm", "sejmem": "sejm",
    "rządu": "rząd", "rządowi": "rząd", "rządem": "rząd",
}

def _normalize_entity(text: str) -> str:
    return _ENTITY_NORMALIZE.get(text, text)

def _fmt_entity(name: str) -> str:
    return name.upper() if name.lower() in _KNOWN_UPPER else name.title()

# ─────────────────────────────────────────────────────────────────────────────
# spaCy MODEL - loaded once, kept in memory (@cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Downloading Polish NLP model (first run only)...")
def get_nlp():
    try:
        return spacy.load("pl_core_news_lg")
    except OSError:
        spacy.cli.download("pl_core_news_lg")
        return spacy.load("pl_core_news_lg")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING AND CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip Reddit noise before NLP analysis."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[ur]/\w+", " ", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^>.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_data
def load_data(n: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sample_data.csv").dropna(subset=["content"])
    df = df.drop_duplicates(subset=["content"])
    df = df[df["content"].str.len() > 15]
    df["content"] = df["content"].apply(clean_text)
    df = df[df["content"].str.len() > 15]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.head(n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED spaCy PROCESSING - runs once, results shared across tabs
# Returns serialisable lists (not Doc objects) for @cache_data compatibility
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running spaCy NLP (pl_core_news_lg)...")
def process_docs(n: int) -> tuple:
    """
    For n documents returns two lists (indexed the same as load_data(n)):
      - lemmas:   list[list[str]]  - token lemmas (alpha, len>=3)
      - entities: list[list[dict]] - entities [{text, label}, ...]
    """
    nlp = get_nlp()
    data = load_data(n)
    texts = [str(t)[:500] for t in data["content"]]

    all_lemmas: list[list[str]] = []
    all_entities: list[list[dict]] = []

    for doc in nlp.pipe(texts, batch_size=128):
        lemmas = [
            tok.lemma_.lower()
            for tok in doc
            if tok.is_alpha and len(tok.lemma_) >= 3
        ]
        entities = [
            {"text": _normalize_entity(ent.root.lemma_.lower()), "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in LABEL_NAMES
        ]
        all_lemmas.append(lemmas)
        all_entities.append(entities)

    return all_lemmas, all_entities


# ─────────────────────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="NLP Dashboard PL")
st.title("NLP Dashboard - Reddit PL")

st.sidebar.header("Global parameters")
sample_size = st.sidebar.slider("Documents", 500, 10_000, 2_000, step=500)

with st.status("Loading...", expanded=True) as _status:
    st.write("Reading dataset...")
    df_global = load_data(sample_size)
    st.write("Loading Polish NLP model...")
    get_nlp()
    st.write("Running NLP pipeline...")
    process_docs(sample_size)
    _status.update(label="Ready", state="complete", expanded=False)

st.sidebar.caption(f"Subreddits: {', '.join(df_global['subreddit'].unique())}")
st.sidebar.caption(f"Documents after filtering: {len(df_global)}")

tab_ner, tab_hate, tab_patterns = st.tabs([
    "🔍 NER - Named entities",
    "⚠️  Hate Speech & Disinformation",
    "📊 Linguistic patterns",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 - NER + CO-OCCURRENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab_ner:
    st.header("Named Entity Recognition (NER)")
    st.info("""
**What is NER?**
The model scans each text and labels spans as named entities of a specific type:
- **persName** → people (*Tusk, Kaczyński, Duda*)
- **orgName** → organisations (*PiS, Sejm, NBP, UE*)
- **placeName / geogName** → places (*Warsaw, Poland, Brussels*)

Using `pl_core_news_lg` - better accuracy than `_sm` at the cost of speed.
500-character limit per text - a speed/accuracy trade-off.
    """)

    ner_n = st.slider(
        "Documents for NER",
        100, min(3_000, sample_size), min(500, sample_size), step=100,
    )

    lemmas_ner, entities_ner = process_docs(ner_n)
    df_ner_base = load_data(ner_n)

    @st.cache_data
    def build_ner_df(n: int) -> pd.DataFrame:
        _, entities = process_docs(n)
        data = load_data(n)
        rows = []
        for i, ents in enumerate(entities):
            for e in ents:
                rows.append({
                    "entity":    e["text"],
                    "label":     e["label"],
                    "label_en":  LABEL_NAMES[e["label"]],
                    "subreddit": data.at[i, "subreddit"],
                    "type":      data.at[i, "type"],
                    "sentiment": data.at[i, "sentiment"],
                    "replies":   data.at[i, "replies"],
                })
        return pd.DataFrame(rows)

    df_ner = build_ner_df(ner_n)

    if df_ner.empty:
        st.warning("No entities found - increase the sample size.")
    else:
        st.caption(f"Found **{len(df_ner)}** entities in {ner_n} documents")

        col_f1, col_f2 = st.columns(2)
        label_filter = col_f1.multiselect(
            "Entity type", sorted(df_ner["label_en"].unique()),
            default=sorted(df_ner["label_en"].unique()),
        )
        sub_filter = col_f2.multiselect(
            "Subreddit", sorted(df_ner["subreddit"].unique()),
            default=sorted(df_ner["subreddit"].unique()),
        )
        df_f = df_ner[
            df_ner["label_en"].isin(label_filter) &
            df_ner["subreddit"].isin(sub_filter)
        ]

        top_n = st.slider("Entities to show", 5, 40, 20)
        top_ents = df_f["entity"].value_counts().head(top_n).reset_index()
        top_ents.columns = ["entity", "count"]
        top_ents["entity"] = top_ents["entity"].apply(_fmt_entity)
        fig = px.bar(
            top_ents, x="count", y="entity", orientation="h",
            color="count", color_continuous_scale="Blues",
            title=f"Top {top_n} most frequent entities",
            labels={"count": "Occurrences", "entity": "Entity"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Subreddit comparison - who/what appears where?")
        top20 = df_f["entity"].value_counts().head(20).index.tolist()
        pivot = (
            df_f[df_f["entity"].isin(top20)]
            .groupby(["entity", "subreddit"]).size().reset_index(name="count")
        )
        pivot["entity"] = pivot["entity"].apply(_fmt_entity)
        fig2 = px.bar(
            pivot, x="entity", y="count", color="subreddit", barmode="group",
            title="Top entities per subreddit",
            labels={"count": "Count", "entity": "Entity"},
        )
        fig2.update_xaxes(tickangle=40)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Entities vs post sentiment")
        st.write("Do the same people/organisations appear in positive and negative contexts?")
        top8 = df_f["entity"].value_counts().head(8).index.tolist()
        sent_pivot = (
            df_f[df_f["entity"].isin(top8)]
            .groupby(["entity", "sentiment"]).size().reset_index(name="count")
        )
        sent_pivot["entity"] = sent_pivot["entity"].apply(_fmt_entity)
        sent_pivot["sentiment"] = sent_pivot["sentiment"].map(SENTIMENT_LABELS)
        fig3 = px.bar(
            sent_pivot, x="entity", y="count", color="sentiment", barmode="stack",
            title="Post sentiment by entity",
            labels={"entity": "Entity", "count": "Occurrences", "sentiment": "Sentiment"},
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Entity type distribution")
        type_counts = df_f["label_en"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig4 = px.pie(type_counts, names="type", values="count", title="Entity type breakdown")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Entity co-occurrence")
        st.write("""
Which entities appear **together in the same post** most often?
Uses `itertools.combinations` - counts all entity pairs per document.
Shows which politicians/parties are mentioned together.
        """)

        @st.cache_data
        def build_cooccurrence(n: int, label_types: tuple) -> pd.DataFrame:
            _, entities = process_docs(n)
            counter: dict[tuple, int] = {}
            for ents in entities:
                doc_ents = list({
                    e["text"] for e in ents
                    if LABEL_NAMES.get(e["label"]) in label_types
                })
                for a, b in combinations(sorted(doc_ents), 2):
                    key = (a, b)
                    counter[key] = counter.get(key, 0) + 1
            if not counter:
                return pd.DataFrame(columns=["pair", "count"])
            pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:15]
            return pd.DataFrame(
                [{"pair": f"{a}  ↔  {b}", "count": c} for (a, b), c in pairs]
            )

        cooc_types = tuple(sorted(label_filter))
        df_cooc = build_cooccurrence(ner_n, cooc_types)

        if df_cooc.empty:
            st.info("Not enough entities to compute co-occurrence.")
        else:
            fig5 = px.bar(
                df_cooc, x="count", y="pair", orientation="h",
                title="Top 15 entity pairs co-occurring in the same post",
                labels={"count": "Posts", "pair": "Entity pair"},
                color="count", color_continuous_scale="Oranges",
            )
            fig5.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig5, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 - HATE SPEECH & DISINFORMATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_hate:
    st.header("Hate Speech & Disinformation")
    st.info("""
**Approach: linguistic heuristics (explainable AI)**

The data is pre-normalised (lowercase, no `!?`) - metrics based on capitalisation
and punctuation don't apply. We use purely textual features:

| Feature | What it measures | Why it matters |
|---|---|---|
| **Emotional density** | share of emotional **lemmas** | handles Polish morphological inflection |
| **Clickbait** | regex patterns on raw text | disinformation, sensationalism |
| **Propaganda score** | matched propaganda techniques | manipulation patterns |
| **Risk score** | emotional (50%) + propaganda (30%) + clickbait (20%) | aggregated signal |

`emotional_density` operates on **spaCy lemmas** - "idiotą", "idiotom" → "idiota" ✓
    """)

    @st.cache_data(show_spinner="Computing features (with spaCy lemmatisation)...")
    def compute_hate_features(n: int) -> pd.DataFrame:
        data = load_data(n).copy()
        all_lemmas, _ = process_docs(n)

        data["is_clickbait"] = data["content"].apply(
            lambda t: int(bool(_clickbait_re.search(str(t))))
        )
        data["propaganda_score"] = data["content"].apply(
            lambda t: sum(1 for re_ in _propaganda_res.values() if re_.search(str(t)))
        )

        # emotional_density on LEMMAS - correct for Polish inflection
        data["emotional_density"] = [
            sum(1 for lem in lemmas if lem in EMOTIONAL_WORDS) / len(lemmas)
            if lemmas else 0.0
            for lemmas in all_lemmas
        ]

        mx_em   = data["emotional_density"].max()
        mx_prop = data["propaganda_score"].max()
        data["emotional_density_norm"] = data["emotional_density"] / mx_em   if mx_em   > 0 else 0.0
        data["propaganda_score_norm"]  = data["propaganda_score"]  / mx_prop if mx_prop > 0 else 0.0

        data["risk_score"] = (
            data["emotional_density_norm"] * 0.5 +
            data["propaganda_score_norm"]  * 0.3 +
            data["is_clickbait"]           * 0.2
        )
        return data

    df_hate = compute_hate_features(sample_size)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg emotional density",  f"{df_hate['emotional_density'].mean():.4f}")
    c2.metric("Clickbait posts",
              f"{df_hate['is_clickbait'].sum()}",
              f"{df_hate['is_clickbait'].mean():.1%} of sample")
    c3.metric("Avg propaganda score", f"{df_hate['propaganda_score'].mean():.2f} / 4")

    st.divider()

    st.subheader("Emotional density per subreddit and post type")
    st.write("Share of emotional lemmas in text - correctly handles Polish inflection.")
    em_agg = df_hate.groupby(["subreddit", "type"])["emotional_density"].mean().reset_index()
    fig = px.bar(
        em_agg, x="subreddit", y="emotional_density", color="type", barmode="group",
        title="Mean emotional density per subreddit and post type",
        labels={"emotional_density": "Emotional density (lemmas)", "type": "Type"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Emotional density vs sentiment")
    em_sent = df_hate.groupby(["subreddit", "sentiment"])["emotional_density"].mean().reset_index()
    em_sent["sentiment"] = em_sent["sentiment"].map(SENTIMENT_LABELS)
    fig2 = px.bar(
        em_sent, x="subreddit", y="emotional_density", color="sentiment", barmode="group",
        title="Emotional density per subreddit and sentiment",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
        labels={"emotional_density": "Emotional density", "sentiment": "Sentiment"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Risk map: emotional density vs propaganda score")
    st.write("Top-right = posts combining emotional language with propaganda patterns.")
    sample_sc = df_hate.sample(min(800, len(df_hate)), random_state=42).copy()
    sample_sc["sentiment"] = sample_sc["sentiment"].map(SENTIMENT_LABELS)
    fig3 = px.scatter(
        sample_sc, x="emotional_density", y="propaganda_score",
        size=sample_sc["replies"].clip(lower=1),
        color="sentiment",
        hover_data={"content": True, "subreddit": True, "risk_score": ":.3f"},
        title="Emotional density vs propaganda score (bubble size = replies)",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
        opacity=0.65,
        labels={"emotional_density": "Emotional density (lemmas)", "propaganda_score": "Propaganda score", "sentiment": "Sentiment"},
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 20 posts by risk score")
    cols_show = ["content", "subreddit", "type", "sentiment",
                 "emotional_density", "propaganda_score", "is_clickbait", "risk_score", "replies"]
    st.dataframe(
        df_hate.nlargest(20, "risk_score")[cols_show]
        .style.background_gradient(subset=["risk_score"], cmap="Reds"),
        use_container_width=True,
    )

    st.subheader("Posts with clickbait patterns")
    st.caption(f"Patterns: `{'` | `'.join(CLICKBAIT_PATTERNS[:5])}` ...")
    cb = df_hate[df_hate["is_clickbait"] == 1][
        ["content", "subreddit", "sentiment", "replies", "score", "risk_score"]
    ]
    if cb.empty:
        st.info("No clickbait patterns detected in this sample.")
    else:
        st.caption(f"Found {len(cb)} posts ({len(cb)/len(df_hate):.1%} of sample)")
        st.dataframe(cb, use_container_width=True)

    st.subheader("Propaganda patterns")
    st.write("""
Each technique is a separate set of regex patterns - the score counts **distinct techniques** matched per post.

| Technique | Example patterns |
|---|---|
| Fear (appeal to fear) | "threat", "catastrophe", "will destroy" |
| Whataboutism | "but what about", "the previous government" |
| Generalisation | "everyone", "always", "never", "every" |
| False dichotomy | "either X or Y", "who is not with us" |
    """)

    prop_agg = df_hate.groupby(["subreddit", "propaganda_score"]).size().reset_index(name="count")
    fig_prop = px.bar(
        prop_agg, x="propaganda_score", y="count", color="subreddit", barmode="group",
        title="Distribution of propaganda technique count per subreddit",
        labels={"propaganda_score": "Techniques matched (0–4)", "count": "Posts"},
    )
    st.plotly_chart(fig_prop, use_container_width=True)

    st.write("How often does each technique appear per subreddit?")
    prop_rows = []
    for name, re_ in _propaganda_res.items():
        df_hate[f"_prop_{name}"] = df_hate["content"].apply(
            lambda t: int(bool(re_.search(str(t))))
        )
        for sub, grp in df_hate.groupby("subreddit"):
            prop_rows.append({
                "technique": name,
                "subreddit": sub,
                "proportion": grp[f"_prop_{name}"].mean(),
            })
    df_prop = pd.DataFrame(prop_rows)
    fig_prop2 = px.bar(
        df_prop, x="technique", y="proportion", color="subreddit", barmode="group",
        title="Share of posts matching each propaganda technique",
        labels={"proportion": "Share of posts", "technique": "Technique"},
    )
    fig_prop2.update_yaxes(tickformat=".0%")
    fig_prop2.update_xaxes(tickangle=15)
    st.plotly_chart(fig_prop2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 - LINGUISTIC PATTERNS + TEMPORAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_patterns:
    st.header("Linguistic patterns")
    st.info("""
| Metric | What it measures |
|---|---|
| **Text length** | word count |
| **MATTR** (Moving Average TTR) | vocabulary richness - independent of text length |
| **Avg word length** | language formality |

Data is normalised (lowercase, no `!?`) - punctuation-based metrics are excluded.

**Why MATTR instead of TTR?**
TTR (unique/total words) inflates for short texts - a 10-word post always scores higher than a 200-word post,
even if the longer one is linguistically richer.
MATTR computes TTR over a sliding window of n=20 words and averages - eliminating this bias.
    """)

    @st.cache_data(show_spinner="Computing linguistic features...")
    def compute_linguistic(n: int) -> pd.DataFrame:
        data = load_data(n).copy()

        def mattr(text: str, window: int = 20) -> float:
            """Moving Average TTR - independent of text length.
            Falls back to plain TTR for texts shorter than the window."""
            words = str(text).lower().split()
            if len(words) < window:
                return len(set(words)) / len(words) if words else 0.0
            ttrs = [
                len(set(words[i : i + window])) / window
                for i in range(len(words) - window + 1)
            ]
            return sum(ttrs) / len(ttrs)

        def avg_word_len(text: str) -> float:
            words = [w for w in str(text).split() if w.isalpha()]
            return sum(len(w) for w in words) / len(words) if words else 0.0

        data["word_count"]   = data["content"].apply(lambda t: len(str(t).split()))
        data["mattr"]        = data["content"].apply(mattr)
        data["avg_word_len"] = data["content"].apply(avg_word_len)
        return data

    df_ling = compute_linguistic(sample_size)

    st.subheader("Post length distribution (words)")
    st.write(
        "Two independent Y axes - left for comments (higher volume), right for submissions. "
        "Keeps both distributions visually comparable."
    )
    import plotly.graph_objects as go
    nbins = 60
    max_words = 300

    df_comm = df_ling[(df_ling["type"] == "comment")  & (df_ling["word_count"] <= max_words)]
    df_sub  = df_ling[(df_ling["type"] == "submission") & (df_ling["word_count"] <= max_words)]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_comm["word_count"], nbinsx=nbins,
        name="comment", marker_color="#636EFA", opacity=0.7,
        yaxis="y1",
    ))
    fig.add_trace(go.Histogram(
        x=df_sub["word_count"], nbinsx=nbins,
        name="submission", marker_color="#EF553B", opacity=0.7,
        yaxis="y2",
    ))
    fig.update_layout(
        title="Text length distribution (dual Y axes)",
        xaxis=dict(title="Word count", range=[0, max_words]),
        yaxis=dict(title=dict(text="Comments", font=dict(color="#636EFA"))),
        yaxis2=dict(
            title=dict(text="Submissions", font=dict(color="#EF553B")),
            overlaying="y", side="right",
        ),
        barmode="overlay",
        legend=dict(x=0.75, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("MATTR - vocabulary richness (Moving Average TTR)")
    mattr_agg = df_ling.groupby(["subreddit", "type"])["mattr"].mean().reset_index()
    fig2 = px.bar(
        mattr_agg, x="subreddit", y="mattr", color="type", barmode="group",
        title="Mean MATTR per subreddit (higher = richer vocabulary)",
        labels={"mattr": "MATTR (window=20)"},
    )
    fig2.update_yaxes(range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Vocabulary richness vs sentiment")
    st.write("Do negative posts use simpler language?")
    mattr_sent = df_ling.groupby(["sentiment", "subreddit"])["mattr"].mean().reset_index()
    fig3 = px.bar(
        mattr_sent, x="sentiment", y="mattr", color="subreddit", barmode="group",
        title="Mean MATTR per sentiment",
        labels={"mattr": "MATTR"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation with engagement (replies)")
    st.write("""
**Pearson correlation**: +1 → more of this feature = more replies | -1 → inverse | ~0 → no relationship
    """)
    corr_cols = ["word_count", "mattr", "avg_word_len", "score", "ratio"]
    corr = df_ling[corr_cols + ["replies"]].corr()["replies"].drop("replies").sort_values()
    fig4 = px.bar(
        x=corr.values, y=corr.index, orientation="h",
        title="Pearson correlation with reply count",
        labels={"x": "Correlation", "y": "Feature"},
        color=corr.values, color_continuous_scale="RdBu", range_color=[-1, 1],
    )
    fig4.add_vline(x=0, line_dash="dash", line_color="black")
    fig4.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Text length vs reply count")
    df_sub = df_ling[df_ling["type"] == "submission"]
    fig5 = px.scatter(
        df_sub.sample(min(600, len(df_sub)), random_state=1),
        x="word_count", y="replies", color="subreddit",
        title="Do longer posts get more replies?",
        labels={"word_count": "Word count", "replies": "Replies"},
        opacity=0.6,
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Temporal analysis - metric trends over time")
    st.write("""
The `date` column lets us check whether emotional intensity and risk changed over the election period.
Aggregated by **month**, compared across subreddits.
    """)

    @st.cache_data
    def compute_temporal(n: int) -> pd.DataFrame:
        hate = compute_hate_features(n)
        hate["month"] = hate["date"].dt.to_period("M").dt.to_timestamp()
        return (
            hate.groupby(["month", "subreddit"])[["risk_score", "emotional_density", "propaganda_score"]]
            .mean()
            .reset_index()
            .dropna(subset=["month"])
        )

    df_temp = compute_temporal(sample_size)

    if df_temp.empty or df_temp["month"].isna().all():
        st.info("No date data available - check the 'date' column in the CSV.")
    else:
        metric_choice = st.selectbox(
            "Metric to analyse",
            ["risk_score", "emotional_density", "propaganda_score"],
            format_func=lambda x: {
                "risk_score": "Risk score",
                "emotional_density": "Emotional density",
                "propaganda_score": "Propaganda score",
            }[x],
        )
        fig7 = px.line(
            df_temp, x="month", y=metric_choice, color="subreddit",
            title=f"{metric_choice} - monthly trend per subreddit",
            labels={"month": "Month", metric_choice: metric_choice},
            markers=True,
        )
        st.plotly_chart(fig7, use_container_width=True)
