"""rag_common.py: shared code for the M5 inventory agent notebooks (IDS 560, Group 4).

Notebook 1 builds the evidence library, notebook 2 tests retrieval, notebook 3 runs the agent, notebook 4 evaluates orders.
Each notebook runs:
    import rag_common as rc
    rc.setup(m5=..., library=..., forecasts=...)
and can change any setting before calling functions, e.g. rc.OPENROUTER_MODEL = "...". The functions here are the
same ones the Streamlit app can import.
"""
import os, re, io, json, time, math, pickle, hashlib, warnings
from statistics import NormalDist
import numpy as np
import pandas as pd
import requests
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================== 1. Configuration
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parent) + "/"
M5_DIR = PROJECT_DIR + "M5/datasets/"
RAG_DIR = PROJECT_DIR + "RAG/"
RAW_DIR = RAG_DIR + "raw/"
FORECAST_DIR = PROJECT_DIR + "forecasting/"

CUBE_FILE = FORECAST_DIR + "forecast_cube_lightgbm_50.npz"
SEGMENTS_FILE = FORECAST_DIR + "series_segments_50.csv"
SEG_ERROR_FILE = FORECAST_DIR + "lightgbm_error_by_segment_50.csv"
WINDOW_START, WINDOW_END = "2011-01-01", "2016-06-30"
STATES = {"CA": "CALIFORNIA", "TX": "TEXAS", "WI": "WISCONSIN"}
STATE_NAMES = {"CA": "California", "TX": "Texas", "WI": "Wisconsin", "US": "United States"}
CATS = ["FOODS", "HOBBIES", "HOUSEHOLD"]

EMBED_MODEL = "BAAI/bge-small-en-v1.5"       # text embedding (Hugging Face)
RERANK_MODEL = "BAAI/bge-reranker-base"      # re-scores the top passages for each question (Hugging Face)
OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl:free"   # any model id ending in ":free" (see openrouter.ai/models)
OPENROUTER_URL = "https://openrouter.ai/api/v1"
LLM_TIMEOUT = 90
SEC_USER_AGENT = "IDS560 Group 4 student project"   # notebook 1 sets the full name and email (the SEC requires one)
CHUNK_WORDS, CHUNK_OVERLAP = 300, 40
# How far back each document type is useful for a decision on a given date (None = no limit)
MAX_AGE_DAYS = {"storm_event": 45, "beige_book": 120, "sec_filing": 200, "walmart_doc": 400, "policy_doc": 400,
                "evidence_card": None, "reference": None, "extra": None}

RUN_TAG = "v4"                               # added to every saved file name, so a new run never overwrites an earlier one
def out_path(name):
    base, ext = os.path.splitext(name)
    return RAG_DIR + f"{base}_{RUN_TAG}{ext}"

SETTINGS_FILE = RAG_DIR + f"settings_{RUN_TAG}.json"   # values one notebook chooses and the others use
def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_setting(**kw):
    s = load_settings(); s.update(kw)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=1)

# ============================================================== 2. Helpers: downloads, cleaning, chunking
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "IDS560-Group4-RAG/1.0 (student project)", "Accept-Encoding": "gzip, deflate"})

def cache_name(url):
    base = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://", 1)[-1])[-90:]
    return hashlib.md5(url.encode()).hexdigest()[:10] + "_" + base

def fetch(url, pause=0.0, tries=4):
    """Download once, then read from the Drive cache. Returns bytes."""
    path = os.path.join(RAW_DIR, cache_name(url))
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    for attempt in range(tries):
        try:
            extra = {"User-Agent": SEC_USER_AGENT} if "sec.gov" in url else None    # contact email only for the SEC
            r = SESSION.get(url, timeout=120, headers=extra)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    with open(path, "wb") as f:
        f.write(r.content)
    if pause:
        time.sleep(pause)
    return r.content

def clean(text):
    text = re.sub(r"[ \t\r\f\v\xa0]+", " ", str(text))
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def chunk_text(text, max_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split on paragraphs, then pack paragraphs into chunks of about max_words words."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks, cur = [], []
    for p in paras:
        words = p.split()
        while len(words) > max_words:                     # very long paragraph: hard split
            if cur:
                chunks.append(" ".join(cur)); cur = cur[-overlap:]
            chunks.append(" ".join(words[:max_words])); words = words[max_words - overlap:]
        if len(cur) + len(words) > max_words and cur:
            chunks.append(" ".join(cur)); cur = cur[-overlap:]
        cur += words
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def new_doc(doc_type, source, title, url, published, state, text, period_start=None, period_end=None, series_key=None, category="ALL"):
    return {"doc_type": doc_type, "source": source, "title": title, "url": url,
            "published": pd.Timestamp(published).normalize(), "state": state, "category": category,
            "period_start": pd.Timestamp(period_start) if period_start is not None else pd.NaT,
            "period_end": pd.Timestamp(period_end) if period_end is not None else pd.NaT,
            "series_key": series_key, "text": clean(text)}


# ============================================================== 3. M5 data (sales, calendar, prices)
def readable(name):
    """'LaborDay' -> 'Labor Day', 'NBAFinalsStart' -> 'NBA Finals Start'."""
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", re.sub(r"([a-z])([A-Z])", r"\1 \2", name))

def load_m5():
    """Loads sales, calendar and weekly prices into module variables: cal, day_cols, vals (series x day), ids, dates,
    day_index, WEEKS, WEEK_END, WEEK_UNITS (series x week) and PRICE (series x week, NaN = not on sale)."""
    global cal, day_cols, vals, ids, dates, day_index, _ROW, WEEKS, WEEK_END, WEEK_UNITS, PRICE
    cal = pd.read_csv(M5_DIR + "calendar.csv", parse_dates=["date"])
    hdr = pd.read_csv(M5_DIR + "sales_train_evaluation.csv", nrows=1).columns
    day_cols = [c for c in hdr if c.startswith("d_")]
    sales = pd.read_csv(M5_DIR + "sales_train_evaluation.csv",
                        dtype={**{c: "int16" for c in day_cols}, **{c: "category" for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]}})
    vals = sales[day_cols].to_numpy(np.int16)
    ids = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].astype(str)
    del sales
    dates = pd.DatetimeIndex(cal.set_index("d").loc[day_cols, "date"])
    day_index = pd.Series(np.arange(len(dates)), index=dates)
    _ROW = {(i, s): k for k, (i, s) in enumerate(zip(ids["item_id"], ids["store_id"]))}
    day_week = cal.set_index("d").loc[day_cols, "wm_yr_wk"].to_numpy()
    week_starts = np.r_[0, np.flatnonzero(np.diff(day_week)) + 1]
    WEEKS = day_week[week_starts]
    WEEK_END = dates[np.r_[week_starts[1:] - 1, len(dates) - 1]]
    WEEK_UNITS = np.add.reduceat(vals, week_starts, axis=1, dtype=np.int32)
    PRICE = np.full(WEEK_UNITS.shape, np.nan, dtype=np.float32)
    pr = pd.read_csv(M5_DIR + "sell_prices.csv", dtype={"store_id": str, "item_id": str})
    pr = pr[pr["wm_yr_wk"].isin(WEEKS)]
    r = pd.Series(range(len(ids)), index=pd.MultiIndex.from_frame(ids[["item_id", "store_id"]])).reindex(
        pd.MultiIndex.from_frame(pr[["item_id", "store_id"]])).to_numpy()
    ok = ~np.isnan(r)
    PRICE[r[ok].astype(int), pd.Series(range(len(WEEKS)), index=WEEKS).reindex(pr["wm_yr_wk"]).to_numpy()[ok]] = pr["sell_price"].to_numpy()[ok]
    print(f"M5 data: {vals.shape[0]:,} series x {vals.shape[1]:,} days | prices for {np.isfinite(PRICE).mean():.0%} of series-weeks")

def has_series(item_id, store_id):
    return (item_id, store_id) in _ROW

# ============================================================== 4. Evidence library and retrieval
class Embedder:
    def __init__(self, model_name):
        self.name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.kind = "dense"
        except Exception as e:
            print("Embedding model unavailable, using TF-IDF instead:", str(e)[:200])
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
            self.kind = "tfidf"
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""

    def fit_encode(self, texts):
        if self.kind == "dense":
            return self.model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        return self.model.fit_transform(texts)

    def encode_query(self, q):
        if self.kind == "dense":
            return self.model.encode([self.query_prefix + q], normalize_embeddings=True)[0].astype(np.float32)
        return self.model.transform([q])

    def scores(self, qv, matrix):
        if self.kind == "dense":
            return matrix @ qv
        return np.asarray((matrix @ qv.T).todense()).ravel()


RELEVANCE_MIN = 0.015       # passages the reranker scores below this count as "no relevant evidence" (notebook 2 updates it)
CAT_WORDS = {"FOODS": ("food", "grocer", "snap"), "HOBBIES": ("hobb", "toy", "craft"), "HOUSEHOLD": ("household", "cleaning", "home")}
chunks = EMB = embedder = KEYWORDS = KW = RERANKER = None

def load_library(reranker=True):
    """Loads the passages and embeddings saved by notebook 1, builds the keyword index and loads the reranker."""
    global chunks, EMB, embedder, KEYWORDS, KW, RERANKER, RELEVANCE_MIN
    chunks = pd.read_pickle(out_path("chunks.pkl"))
    with open(out_path("index.pkl"), "rb") as f:
        saved = pickle.load(f)
    assert saved["chunk_ids"] == chunks["chunk_id"].tolist(), "Passages and index don't match: rerun notebook 1."
    embedder = Embedder(saved["model"])
    EMB = saved["emb"]
    if embedder.kind == "tfidf":
        embedder.model = saved["tfidf"]
    from sklearn.feature_extraction.text import TfidfVectorizer
    KEYWORDS = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
    KW = KEYWORDS.fit_transform(chunks["embed_text"])
    RERANKER = None
    if reranker:
        try:
            from sentence_transformers import CrossEncoder
            RERANKER = CrossEncoder(RERANK_MODEL, max_length=512)
        except Exception as e:
            print("Reranker unavailable, retrieval runs without it:", str(e)[:200])
    RELEVANCE_MIN = load_settings().get("RELEVANCE_MIN", RELEVANCE_MIN)
    print(f"Library: {len(chunks):,} passages from {chunks['doc_id'].nunique():,} documents | reranker "
          f"{'on' if RERANKER is not None else 'off'} | RELEVANCE_MIN {RELEVANCE_MIN:.3f}")

def category_of(text):
    """The category a question is about, if it names exactly one."""
    t = text.lower()
    hits = [c for c, words in CAT_WORDS.items() if any(w in t for w in words)]
    return hits[0] if len(hits) == 1 else None

def eligible_passages(as_of, state=None, category=None, doc_types=None):
    """Passages a decision on as_of may use: published by then, the state or US-wide, the category or all categories,
    within each document type's age limit, and only the latest version of each evidence card."""
    as_of = pd.Timestamp(as_of)
    m = (chunks["published"] <= as_of).to_numpy().copy()
    if state:
        m &= chunks["state"].isin([state, "US"]).to_numpy()
    if category:
        m &= chunks["category"].isin([category, "ALL"]).to_numpy()
    if doc_types:
        m &= chunks["doc_type"].isin(doc_types).to_numpy()
    age = (as_of - chunks["published"]).dt.days.to_numpy()
    for t, days in MAX_AGE_DAYS.items():
        if days is not None:
            m &= ~((chunks["doc_type"] == t).to_numpy() & (age > days))
    cand = chunks[m]
    keyed = cand["series_key"].notna()
    latest = cand.loc[keyed].groupby("series_key")["published"].transform("max")
    return cand.drop(index=latest.index[cand.loc[keyed, "published"] < latest])

def retrieve(query, as_of, state=None, k=6, category=None, method="full", doc_types=None, per_doc=1, per_type=3, pool=30):
    """Best passages for a question as of a date. method: "full" (hybrid + reranker, the agent's), "hybrid+category",
    "dense+category", "dense" (meaning only, no category filter: the v1-v2 search), "keyword", "random" (benchmark baselines)."""
    cand = eligible_passages(as_of, state, category if method != "dense" else None, doc_types)
    if cand.empty:
        return cand.assign(score=[], relevance=[])
    idx = cand.index.to_numpy()
    kw = np.asarray((KW[idx] @ KEYWORDS.transform([query]).T).todense()).ravel()
    dense = embedder.scores(embedder.encode_query(query), EMB[idx])
    if method in ("dense", "dense+category"):
        score = dense
    elif method == "keyword":
        score = kw
    elif method == "random":
        score = np.random.default_rng(int(hashlib.md5(f"{query}|{as_of}".encode()).hexdigest()[:8], 16)).random(len(idx))   # same every run
    else:                                                                # reciprocal rank fusion of the two rankings
        rd = pd.Series(-dense).rank(method="first").to_numpy(); rk = pd.Series(-kw).rank(method="first").to_numpy()
        score = 1 / (60 + rd) + 1 / (60 + rk)
    cand = cand.assign(score=score, relevance=np.nan).sort_values("score", ascending=False)
    if method == "full" and RERANKER is not None:
        top = cand.head(pool)
        rel = RERANKER.predict([(query, t[:2000]) for t in top["embed_text"]], show_progress_bar=False)
        cand = top.assign(relevance=np.asarray(rel, dtype=float)).sort_values("relevance", ascending=False)
    out, n_doc, n_type = [], {}, {}
    for i, r in cand.iterrows():
        if n_doc.get(r["doc_id"], 0) >= per_doc or n_type.get(r["doc_type"], 0) >= per_type:
            continue
        out.append(i); n_doc[r["doc_id"]] = n_doc.get(r["doc_id"], 0) + 1; n_type[r["doc_type"]] = n_type.get(r["doc_type"], 0) + 1
        if len(out) == k:
            break
    return cand.loc[out]

def show(res, width=200):
    for _, r in res.iterrows():
        rel = "" if pd.isna(r.get("relevance")) else f" rel {r['relevance']:.2f}"
        print(f"[{r['score']:.3f}{rel}] {r['doc_type']:13s} {r['state']:2s} {r['published']:%Y-%m-%d}  {r['title']}")
        print("        " + r["text"][:width].replace("\n", " ") + "...\n")


# ============================================================== 5. Forecast and ordering rule
FORECAST_MODEL_NAME = "Recent Mean (7-day) with demand-type P10-P90 intervals"
LGB_MODEL_NAME = "LightGBM, the M5 forecasting model (retrained quarterly + bias correction), P10-P90 from errors known at the time"
ORDER_DAYS = 14                              # lead time 7 + review period 7
ADI_CUT, CV2_CUT = 1.32, 0.49
Z80 = NormalDist().inv_cdf(0.90)            # P10 to P90 spans 2 x 1.2816 standard deviations under a normal curve

# Demand-type interval offsets (q10, q90 per lead 1..7) for the Recent Mean fallback, from the forecasting outputs if present
_OFFSETS_FALLBACK = {
    "Erratic":      [(-3.285714, 6.142858), (-3.428571, 6.142857), (-4.571428, 4.000000), (-4.857143, 3.714286), (-5.000000, 3.714286), (-5.285714, 3.428571), (-4.571429, 4.857142)],
    "Intermittent": [(-0.714286, 1.142857), (-0.714286, 1.285714), (-0.857143, 1.000000), (-0.857143, 0.857143), (-0.857143, 0.857143), (-1.000000, 0.857143), (-0.857143, 1.000000)],
    "Lumpy":        [(-1.571429, 2.428571), (-1.714286, 2.428571), (-1.857143, 1.857143), (-2.000000, 1.714286), (-2.000000, 1.714286), (-2.142857, 1.428571), (-1.857143, 1.857143)],
    "No sales":     [(0.0, 0.0)] * 7,
    "Smooth":       [(-2.571429, 5.285714), (-2.714286, 5.571429), (-3.571429, 3.571428), (-4.142857, 3.000000), (-4.142857, 2.857143), (-4.571428, 2.714286), (-3.714286, 3.857143)],
}

def series_row(item_id, store_id):
    if (item_id, store_id) not in _ROW:
        raise ValueError(f"No series for {item_id} at {store_id}")
    return _ROW[(item_id, store_id)]

def split_series_id(sid):
    """'HOBBIES_1_268_WI_2_evaluation' -> ('HOBBIES_1_268', 'WI_2')"""
    p = sid.replace("_evaluation", "").replace("_validation", "").split("_")
    return "_".join(p[:3]), "_".join(p[3:5])

def demand_type_of(history):
    """Same ADI / CV2 rule as the forecasting notebook (classify_demand_types), for one series."""
    x = np.asarray(history, dtype=np.float64)
    pos = x > 0
    k = int(pos.sum())
    if k == 0:
        return "No sales", np.nan, np.nan
    first = int(pos.argmax())
    adi = (len(x) - first) / k
    mean = x.sum() / k
    cv2 = max((x ** 2).sum() / k - mean ** 2, 0) / mean ** 2
    if adi < ADI_CUT:
        return ("Smooth" if cv2 < CV2_CUT else "Erratic"), adi, cv2
    return ("Intermittent" if cv2 < CV2_CUT else "Lumpy"), adi, cv2

def team_forecast(item_id, store_id, as_of, horizon=ORDER_DAYS):
    """Recent Mean (7-day) point forecast with demand-type P10-P90 intervals (the fallback). Uses sales up to and including as_of."""
    row = series_row(item_id, store_id)
    end = int(day_index[pd.Timestamp(as_of)])
    if end < 6:
        raise ValueError("Need at least 7 days of history")
    point = float(vals[row, end - 6: end + 1].mean())
    dtype_, adi, cv2 = demand_type_of(vals[row, : end + 1])
    off = OFFSETS[dtype_][np.minimum(np.arange(horizon), 6)]          # leads beyond 7 reuse the lead-7 interval
    p10 = np.maximum(0, point + off[:, 0])
    p90 = np.maximum(p10, point + off[:, 1])
    mean = np.full(horizon, point)
    history = vals[row, : end + 1]
    zero_run = int(np.argmax(history[::-1] > 0)) if history.any() else end + 1
    return {"model": FORECAST_MODEL_NAME, "dates": pd.date_range(pd.Timestamp(as_of) + pd.Timedelta(days=1), periods=horizon),
            "mean": mean, "p10": p10, "p90": p90, "sd": (p90 - p10) / (2 * Z80), "demand_type": dtype_, "adi": adi, "cv2": cv2,
            "zero_run": zero_run, "last_28_units": int(vals[row, max(0, end - 27): end + 1].sum())}

def lightgbm_forecast(item_id, store_id, as_of, horizon=ORDER_DAYS):
    """The latest weekly LightGBM forecast made on or before as_of, for the days after as_of; None if the cube doesn't have it."""
    if CUBE is None:
        return None
    j = CUBE["row"].get(ids["id"].iloc[series_row(item_id, store_id)])
    as_of = pd.Timestamp(as_of)
    made = np.flatnonzero(CUBE["origins"] <= as_of)
    if j is None or len(made) == 0 or (as_of - CUBE["origins"][made[-1]]).days > 6:
        return None
    t = made[-1]
    lag = (as_of - CUBE["origins"][t]).days                        # days of this forecast already past on the decision date
    point, p10, p90 = (CUBE[k][t, j].astype(float) for k in ("point", "p10", "p90"))
    keep = np.isfinite(point) & (np.arange(len(point)) >= lag)
    point, p10, p90 = point[keep], p10[keep], p90[keep]
    n = len(point)
    if n == 0:
        return None
    if horizon > n:                                                  # extend: average of the model's days, last day's range
        lo, hi = p10[-1] - point[-1], p90[-1] - point[-1]
        point = np.r_[point, np.full(horizon - n, point.mean())]
        p10 = np.r_[p10, np.maximum(0, point[n:] + lo)]
        p90 = np.r_[p90, np.maximum(p10[n:], point[n:] + hi)]
    point, p10, p90 = point[:horizon], p10[:horizon], p90[:horizon]
    name = LGB_MODEL_NAME + f"; forecast made on {CUBE['origins'][t]:%Y-%m-%d}" + ("" if n >= horizon else f", days {n + 1}-{horizon} extended")
    base = team_forecast(item_id, store_id, as_of, horizon)          # zero run and recent sales, from sales up to as_of
    return {**base, "model": name, "mean": point, "p10": p10, "p90": p90, "sd": (p90 - p10) / (2 * Z80), "model_days": n}

_warned = set()
def baseline_forecast(item_id, store_id, as_of, horizon=ORDER_DAYS, quiet=False):
    """The forecast the rest of the notebook uses: LightGBM from the cube when it has it, otherwise Recent Mean (labelled)."""
    fc = lightgbm_forecast(item_id, store_id, as_of, horizon)
    if fc is not None:
        return fc
    if not quiet and (item_id, store_id, str(as_of)) not in _warned:
        print(f"  No LightGBM forecast for {item_id} at {store_id} on {pd.Timestamp(as_of):%Y-%m-%d}: using Recent Mean instead.")
        _warned.add((item_id, store_id, str(as_of)))
    fc = team_forecast(item_id, store_id, as_of, horizon)
    fc["model"] = FORECAST_MODEL_NAME + " (fallback: no LightGBM forecast for this item and date)"
    return fc

def item_safety_stock(item_id, store_id, as_of, protection_days, service_level=0.95, lookback_weeks=52, min_origins=12):
    """Safety stock from this item's own past forecast errors, using only sales up to as_of.
    For each weekly origin in the last year: forecast = 7-day recent mean x protection days, error = actual total - forecast.
    Safety stock = the service-level quantile of those errors (floored at 0). Returns None if there is too little history."""
    row = series_row(item_id, store_id)
    end = int(day_index[pd.Timestamp(as_of)])
    errs = []
    for k in range(1, lookback_weeks + 1):
        o = end - protection_days - 7 * (k - 1)              # origin: the forecast window o+1 .. o+P ends on or before as_of
        if o < 6:
            break
        point = vals[row, o - 6: o + 1].mean()
        errs.append(vals[row, o + 1: o + 1 + protection_days].sum() - point * protection_days)
    if len(errs) < min_origins:
        return None
    return {"safety_stock": max(0.0, float(np.quantile(errs, service_level))), "n_origins": len(errs),
            "typical_error": float(np.median(np.abs(errs)))}

def order_quantity(mean, sd, on_hand, on_order=0, lead_time_days=7, review_days=7, service_level=0.95, safety=None):
    """Order-up-to rule. safety: a safety-stock dict from item_safety_stock(); if None, falls back to the interval-based estimate."""
    p = lead_time_days + review_days
    mu = float(np.sum(mean[:p]))
    if safety is not None:
        ss, method = safety["safety_stock"], safety.get("method", f"item's own past {p}-day forecast errors ({safety['n_origins']} weekly origins)")
    else:
        ss = NormalDist().inv_cdf(service_level) * float(np.sqrt(np.sum(np.square(sd[:p]))))
        method = "P10-P90 interval (too little item history)"
    target = mu + ss
    return {"protection_days": p, "expected_demand": round(mu, 1), "safety_stock": round(ss, 1), "safety_stock_method": method,
            "order_up_to": round(target, 1), "on_hand": on_hand, "on_order": on_order,
            "order_qty": int(max(0, math.ceil(target - on_hand - on_order))), "service_level": service_level}


# ============================================================== 6. Segments, track record, flags, new items, confidence
TRAIN_END = pd.Timestamp("2014-12-31")       # segments use 2011-2014 sales only
TRACK_MIN_WEEKS = 4                          # weeks of scored forecasts before a segment's error is quoted
_track_cache = {}
def track_record(as_of, horizon=ORDER_DAYS):
    """LightGBM's pooled error by segment over the weekly forecasts whose results were known by as_of (known_date <= as_of)."""
    key = (pd.Timestamp(as_of), horizon)
    if key not in _track_cache:
        out = {}
        if SEG_ERR is not None:
            e = SEG_ERR[(SEG_ERR["horizon"] == horizon) & (SEG_ERR["known_date"] <= key[0])]
            if len(e):
                g = e.groupby(["segment_type", "segment"]).agg(actual=("actual_units", "sum"), abs_err=("abs_error", "sum"),
                                                               signed=("signed_error", "sum"), weeks=("origin_date", "nunique"))
                for k, r in g.iterrows():
                    if r["actual"] > 0:
                        out[k] = {"wape": 100 * r["abs_err"] / r["actual"], "bias": 100 * r["signed"] / r["actual"],
                                  "weeks": int(r["weeks"]), "through": e["known_date"].max()}
        _track_cache[key] = out
    return _track_cache[key]

def segment_of(item_id, store_id):
    """Demand type and volume level from 2011-2014 sales, plus first sale and first price dates."""
    row = series_row(item_id, store_id)
    sid = ids["id"].iloc[row]
    if SEG is not None and sid in SEG.index:
        s = SEG.loc[sid]
        return {"demand_type": s["demand_type"], "volume_level": s["volume_level"], "adi": s["ADI"], "cv2": s["CV2"],
                "first_price": s["first_price_date"], "source": "forecasting notebook segments (2011-2014 sales)"}
    t, adi, cv2 = demand_type_of(vals[row, : int((dates <= TRAIN_END).sum())])
    return {"demand_type": t, "volume_level": None, "adi": adi, "cv2": cv2, "first_price": pd.NaT,
            "source": "computed here from 2011-2014 sales"}

def sales_history(item_id, store_id, as_of):
    """First and last sale up to as_of, days of history and the current zero-sale run. Uses sales up to as_of only."""
    row, end = series_row(item_id, store_id), int(day_index[pd.Timestamp(as_of)])
    sold = np.flatnonzero(vals[row, : end + 1] > 0)
    if len(sold) == 0:
        return {"row": row, "end": end, "first": None, "last": None, "days_selling": 0, "zero_run": end + 1}
    return {"row": row, "end": end, "first": int(sold[0]), "last": int(sold[-1]),
            "days_selling": end - int(sold[0]) + 1, "zero_run": end - int(sold[-1])}

NEW_ITEM_DAYS = 56            # days since first sale below which an item counts as new
DISCONTINUED_DAYS = 56        # zero run (days) from which an unlikely run reads as "possibly discontinued" rather than a stock-out
STOCKOUT_CHANCE = 0.001       # flag a zero run whose chance under the item's normal sale rate is below this (was 0.01)
STOCKOUT_MIN_DAYS = 7         # shortest zero run that can be flagged
HOLD_FLAGS = {"NEW_ITEM", "POSSIBLE_STOCKOUT", "POSSIBLY_DISCONTINUED"}

def item_flags(item_id, store_id, as_of, seg=None):
    """Flags for this order, from sales up to as_of. Each flag: code, hold (bool), message, check (what the manager should do)."""
    as_of = pd.Timestamp(as_of)
    seg = seg or segment_of(item_id, store_id)
    h = sales_history(item_id, store_id, as_of)
    row, end, flags = h["row"], h["end"], []
    if h["first"] is None:
        listed = pd.notna(seg["first_price"]) and seg["first_price"] <= as_of
        msg = (f"New item: listed with a price since {seg['first_price']:%Y-%m-%d} but no sales yet" if listed
               else "New item: no sales history at this store yet")
        return [{"code": "NEW_ITEM", "hold": True, "message": msg,
                 "check": "confirm the item is on the shelf and priced; set the first order by judgment or from similar items"}]
    if h["days_selling"] < NEW_ITEM_DAYS:
        flags.append({"code": "NEW_ITEM", "hold": True,
                      "message": f"New item: first sale {dates[h['first']]:%Y-%m-%d}, only {h['days_selling']} days of sales history",
                      "check": "confirm the item is stocked and check early sales against similar items before ordering"})
    elif h["zero_run"] >= STOCKOUT_MIN_DAYS:
        start = max(h["first"], h["last"] - 364)
        before = vals[row, start: h["last"] + 1]
        k = int((before > 0).sum())
        if k >= 10:
            p = k / len(before)
            chance = (1 - p) ** h["zero_run"]
            if chance < STOCKOUT_CHANCE:
                odds = f"about 1 in {1 / chance:,.0f}" if chance > 1e-9 else "less than 1 in a billion"
                base = (f"No sales for {h['zero_run']} days although the item sold on {k} of the {len(before)} days before "
                        f"({100 * p:.0f}% of days); a run this long has {odds} chance")
                if h["zero_run"] >= DISCONTINUED_DAYS:
                    flags.append({"code": "POSSIBLY_DISCONTINUED", "hold": True, "message": "Possibly discontinued: " + base,
                                  "check": "check whether the store still carries the item (shelf, price, planogram) before ordering"})
                else:
                    flags.append({"code": "POSSIBLE_STOCKOUT", "hold": True, "message": "Possible stock-out: " + base,
                                  "check": "check stock on hand, the shelf and the price; the zeros may be an empty shelf, not zero demand"})
    if seg["demand_type"] == "No sales" and h["days_selling"] >= NEW_ITEM_DAYS:
        flags.append({"code": "STALE_SEGMENT", "hold": False,
                      "message": "No sales in 2011-2014, so the item's demand segment ('No sales') is out of date",
                      "check": "treat the segment's error figures as a rough guide only"})
    return flags


NEW_ITEM_WEEKS = 8           # weeks over which the forecast shifts from similar items to the item's own sales
NEW_MIN_MATCHES = 5          # similar items needed before the cascade stops widening
NEW_MAX_AGE = 730            # similar items must have launched in the 2 years before the decision
GROUP_LEVELS = [("dept + store + price", ["dept", "store", "band"]), ("dept + state + price", ["dept", "state", "band"]),
                ("dept + state", ["dept", "state"]), ("category + state", ["cat", "state"]), ("category", ["cat"])]
AGE_STEPS = np.arange(0, 7 * (NEW_ITEM_WEEKS + 1), 7)        # ages (days since first sale) with stored 14-day totals
NEW_ITEM_STRATEGY, NEW_ITEM_BLEND = "cascade", True          # chosen by the simulation in notebook 4 (saved in the settings file)

def price_band(row, as_of_idx):
    """Low / mid / high third of the department's prices, from the first price listed by the decision week ('mid' if none yet)."""
    if _first_pw[row] < 0 or WEEK_END[_first_pw[row]] > dates[as_of_idx] + pd.Timedelta(days=7):
        return "mid"
    lo, hi = BAND_CUTS[ids["dept_id"].iloc[row]]
    p = _first_price[row]
    return "low" if p <= lo else "high" if p > hi else "mid"
def similar_totals(attrs, d_idx, age, strategy="cascade", exclude_row=None):
    """14-day sales at the same age of similar items launched in the 2 years before day d_idx, whose sales at that age
    were known by d_idx. Returns (group name, array of totals) or (None, empty)."""
    k = min(age // 7, len(AGE_STEPS) - 1)
    known = (L_FIRST + AGE_STEPS[k] + 14 <= d_idx + 1) & (L_FIRST >= d_idx - NEW_MAX_AGE) & np.isfinite(L_TOT[:, k])
    if exclude_row is not None:
        known &= L_ROWS != exclude_row
    levels = GROUP_LEVELS if strategy == "cascade" else [l for l in GROUP_LEVELS if l[0] == strategy] + GROUP_LEVELS[-1:]
    need = NEW_MIN_MATCHES if strategy == "cascade" else 1
    for name, keys in levels:
        m = known.copy()
        for key in keys:
            m &= L_ATTR[key] == attrs[key]
        if m.sum() >= need:
            return name, L_TOT[m, k]
    return None, np.array([])

def new_item_estimate(row, d_idx, strategy="cascade", blend=True):
    """14-day sales estimate for the series in `row` as a new item, decided on day d_idx. Uses sales up to d_idx only."""
    sold = np.flatnonzero(vals[row, : d_idx + 1] > 0)
    f = int(sold[0]) if len(sold) else d_idx + 1
    age = d_idx + 1 - f                                                # days the item has been selling (0 = not yet)
    attrs = {"dept": ids["dept_id"].iloc[row], "store": ids["store_id"].iloc[row], "state": ids["state_id"].iloc[row],
             "cat": ids["cat_id"].iloc[row], "band": price_band(row, d_idx)}
    group, tots = similar_totals(attrs, d_idx, age, strategy, exclude_row=row)
    if group is None:
        return None
    w = min(1.0, age / (7 * NEW_ITEM_WEEKS)) if blend and age > 0 else 0.0
    own = float(vals[row, f: d_idx + 1].mean()) * 14 if age > 0 else 0.0
    mix = lambda x: (1 - w) * x + w * own
    return {"total": mix(float(np.median(tots))), "p10": mix(float(np.quantile(tots, 0.10))), "p90": mix(float(np.quantile(tots, 0.90))),
            "p95": mix(float(np.quantile(tots, 0.95))), "group": group, "n_similar": len(tots), "own_weight": w, "age": age,
            "band": attrs["band"]}

def new_item_forecast(item_id, store_id, as_of, horizon=ORDER_DAYS):
    """Forecast for a new item with the chosen method, in the same format as the other forecasts (or None)."""
    row, d_idx = series_row(item_id, store_id), int(day_index[pd.Timestamp(as_of)])
    est = new_item_estimate(row, d_idx, NEW_ITEM_STRATEGY, NEW_ITEM_BLEND)
    if est is None:
        return None
    base = team_forecast(item_id, store_id, as_of, horizon)
    daily = lambda x: np.full(horizon, x / 14)
    p10, p90 = daily(est["p10"]), daily(est["p90"])
    name = (f"New-item forecast: median of {est['n_similar']} similar items ({est['group']}, {est['band']} price) at age {est['age']} days"
            + (f", {100 * est['own_weight']:.0f}% weight on its own sales" if est["own_weight"] > 0 else ""))
    return {**base, "model": name, "mean": daily(est["total"]), "p10": p10, "p90": p90, "sd": (p90 - p10) / (2 * Z80),
            "new_item": est, "safety": {"safety_stock": max(0.0, est["p95"] - est["total"]), "n_origins": est["n_similar"],
                                        "method": f"spread of {est['n_similar']} similar items' sales (95th percentile minus median)"}}
CONF_WAPE_CUTS = (65, 90)       # segment WAPE below the first = 2 points, below the second = 1 point
CONF_RANGE_MAX = 1.5            # P10-P90 width / forecast at or below this = 1 point
CONF_HISTORY_DAYS = 365         # days of sales history for 1 point

def forecast_confidence(fc, seg, flags, as_of, history=None):
    """Returns level (High / Medium / Low), points and the reasons, using only information known on as_of."""
    tr = track_record(as_of)
    pts, why = 0, []
    known = [(kind, key, tr[(kind, key)]) for kind, key in [("demand_type", seg["demand_type"]), ("volume_level", seg["volume_level"])]
             if key and (kind, key) in tr and tr[(kind, key)]["weeks"] >= TRACK_MIN_WEEKS]
    if known:
        kind, key, r = max(known, key=lambda x: x[2]["wape"])
        p = 2 if r["wape"] < CONF_WAPE_CUTS[0] else 1 if r["wape"] < CONF_WAPE_CUTS[1] else 0
        label = f"{key} items" if kind == "demand_type" else f"{key.lower()}-volume items"
        why.append(f"LightGBM's 14-day error so far for {label} is {r['wape']:.0f}% WAPE")
    else:
        p = {"Smooth": 2, "Erratic": 2, "Lumpy": 1}.get(seg["demand_type"], 0)
        why.append(f"{seg['demand_type']} demand, no LightGBM results known yet")
    pts += p
    n = len(fc["mean"])
    rel = (fc["p90"].sum() - fc["p10"].sum()) / max(fc["mean"].sum(), 0.1)
    if rel <= CONF_RANGE_MAX:
        pts += 1; why.append(f"narrow forecast range ({rel:.1f} times the {n}-day forecast)")
    else:
        why.append(f"wide forecast range ({rel:.1f} times the {n}-day forecast)")
    days = (history or {}).get("days_selling", 0)
    if days >= CONF_HISTORY_DAYS:
        pts += 1
    else:
        why.append(f"only {days} days of sales history")
    level = "High" if pts >= 3 else "Medium" if pts == 2 else "Low"
    if any(f["hold"] for f in flags):
        level = "Low"; why.insert(0, "order on hold until checked")
    if not str(fc.get("model", "")).startswith("LightGBM"):
        level = {"High": "Medium", "Medium": "Low", "Low": "Low"}[level]; why.append("Recent Mean fallback, not LightGBM")
    return {"level": level, "points": pts, "reasons": why}

def load_forecasts():
    """Loads the forecast cube, the segment files and the interval offsets, and builds the price levels and the table of
    past launches used by the new-item forecast."""
    global OFFSETS, CUBE, SAMPLE_IDS, SEG, SEG_ERR, _track_cache, _first_pw, _first_price, BAND_CUTS, L_ROWS, L_FIRST, L_TOT, L_ATTR
    global NEW_ITEM_STRATEGY, NEW_ITEM_BLEND
    off_path = FORECAST_DIR + "prediction_interval_demand_type_offsets_50.csv"
    if os.path.exists(off_path):
        o = pd.read_csv(off_path).sort_values(["Demand_Type", "lead"])
        OFFSETS = {t: np.array(g[["q10_offset", "q90_offset"]], dtype=float) for t, g in o.groupby("Demand_Type")}
    else:
        OFFSETS = {t: np.array(v, dtype=float) for t, v in _OFFSETS_FALLBACK.items()}
    CUBE = None
    if os.path.exists(CUBE_FILE):
        z = np.load(CUBE_FILE, allow_pickle=False)
        CUBE = {"point": z["point"], "p10": z["p10"], "p90": z["p90"], "origins": pd.DatetimeIndex(pd.to_datetime(z["origin_date"])),
                "ids": z["series_id"].astype(str), "row": {s: i for i, s in enumerate(z["series_id"].astype(str).tolist())}}
        print(f"Forecast cube: {len(CUBE['row']):,} series, {len(CUBE['origins'])} weekly forecasts "
              f"({CUBE['origins'][0]:%Y-%m-%d} to {CUBE['origins'][-1]:%Y-%m-%d}), {CUBE['point'].shape[2]} days each")
    else:
        print("forecast_cube_lightgbm_50.npz not found in", FORECAST_DIR, "- every forecast will be the Recent Mean fallback")
    SAMPLE_IDS = set(CUBE["row"]) if CUBE else set()
    SEG = pd.read_csv(SEGMENTS_FILE, parse_dates=["first_sale_date", "first_price_date"]).set_index("series_id") if os.path.exists(SEGMENTS_FILE) else None
    SEG_ERR = pd.read_csv(SEG_ERROR_FILE, parse_dates=["origin_date", "known_date"]) if os.path.exists(SEG_ERROR_FILE) else None
    _track_cache = {}
    print("Segments:", f"{len(SEG):,} series" if SEG is not None else "file not found (demand type computed here)",
          "| model error by segment:", f"{len(SEG_ERR):,} rows" if SEG_ERR is not None else "file not found")
    # Price level of each series: its first listed price against its department's usual 2011-2014 prices (thirds)
    has_p = np.isfinite(PRICE)
    _first_pw = np.where(has_p.any(1), has_p.argmax(1), -1)
    _first_price = np.where(_first_pw >= 0, PRICE[np.arange(len(ids)), np.maximum(_first_pw, 0)], np.nan)
    usual = np.nanmedian(np.where(has_p[:, WEEK_END <= TRAIN_END], PRICE[:, WEEK_END <= TRAIN_END], np.nan), axis=1)
    BAND_CUTS = {d: np.nanquantile(usual[(ids["dept_id"] == d).to_numpy()], [1 / 3, 2 / 3]) for d in ids["dept_id"].unique()}
    # Past launches: series first sold after day 60, with their 14-day sales at ages 0, 7, ..., 56 days
    pos = vals > 0
    first_sale = np.where(pos.any(1), pos.argmax(1), -1)
    del pos
    L_ROWS = np.flatnonzero(first_sale >= 60)
    L_FIRST = first_sale[L_ROWS]
    cs = np.concatenate([np.zeros((len(L_ROWS), 1), np.int32), np.cumsum(vals[L_ROWS], axis=1, dtype=np.int32)], axis=1)
    L_TOT = np.full((len(L_ROWS), len(AGE_STEPS)), np.nan, np.float32)
    for k, a in enumerate(AGE_STEPS):
        s = L_FIRST + a
        ok = s + 14 <= vals.shape[1]
        L_TOT[ok, k] = cs[ok, s[ok] + 14] - cs[ok, s[ok]]
    del cs
    L_ATTR = {"dept": ids["dept_id"].to_numpy()[L_ROWS], "store": ids["store_id"].to_numpy()[L_ROWS],
              "state": ids["state_id"].to_numpy()[L_ROWS], "cat": ids["cat_id"].to_numpy()[L_ROWS],
              "band": np.array([price_band(r, f) for r, f in zip(L_ROWS, L_FIRST)])}
    st = load_settings()
    NEW_ITEM_STRATEGY, NEW_ITEM_BLEND = st.get("NEW_ITEM_STRATEGY", NEW_ITEM_STRATEGY), st.get("NEW_ITEM_BLEND", NEW_ITEM_BLEND)
    print(f"Past launches: {len(L_ROWS):,} | new-item method: {NEW_ITEM_STRATEGY}" + (" + own sales" if NEW_ITEM_BLEND else ""))

# ============================================================== 7. Forecast-data lookup (questions about the forecasts themselves)
# Questions like "which store has the highest predicted demand?" are answered from the forecast cube, not from documents.
# The lookup computes the answer and gives it to the model as passage [E1]. M5 has no stock-on-hand data, so questions
# about inventory, shortages or excess use an ASSUMED stock: ASSUMED_STOCK_DAYS days of the item's average daily sales
# over the last 4 weeks. Every lookup that uses it says so.
ASSUMED_STOCK_DAYS = 7
LOOKUP_DAYS = 7              # "next 7 days" in the questions
_table_cache = {}

def cube_slot(as_of):
    """(forecast index, days already past) for the latest weekly forecast made on or before as_of, or None."""
    if CUBE is None:
        return None
    as_of = pd.Timestamp(as_of)
    made = np.flatnonzero(CUBE["origins"] <= as_of)
    if len(made) == 0 or (as_of - CUBE["origins"][made[-1]]).days > 6:
        return None
    return made[-1], (as_of - CUBE["origins"][made[-1]]).days

def forecast_table(as_of, days=LOOKUP_DAYS):
    """One row per series in the forecast cube: next `days` days of forecast, rough P10/P90 (sums of the daily values),
    assumed stock, shortage, excess and their value at the latest price. Also returns the daily forecasts (series x day)."""
    key = (pd.Timestamp(as_of), days)
    if key in _table_cache:
        return _table_cache[key]
    slot = cube_slot(as_of)
    if slot is None:
        raise ValueError(f"No weekly forecast covers {pd.Timestamp(as_of):%Y-%m-%d} (the cube runs from "
                         f"{CUBE['origins'][0]:%Y-%m-%d} to {CUBE['origins'][-1]:%Y-%m-%d})" if CUBE else "No forecast cube loaded")
    t, lag = slot
    daily = np.nan_to_num(CUBE["point"][t, :, lag:lag + days].astype(float))
    p10 = np.nan_to_num(CUBE["p10"][t, :, lag:lag + days].astype(float)).sum(1)
    p90 = np.nan_to_num(CUBE["p90"][t, :, lag:lag + days].astype(float)).sum(1)
    items = [split_series_id(s) for s in CUBE["ids"]]
    rows = np.array([_ROW.get(x, -1) for x in items])
    end = int(day_index[pd.Timestamp(as_of)])
    recent = np.where(rows >= 0, vals[np.maximum(rows, 0), max(0, end - 27): end + 1].mean(1), 0.0)
    wk = int(np.searchsorted(WEEK_END, pd.Timestamp(as_of)))                       # current week
    pw = PRICE[np.maximum(rows, 0), : min(wk + 1, PRICE.shape[1])]
    last_price = pd.DataFrame(pw).ffill(axis=1).iloc[:, -1].to_numpy() if pw.shape[1] else np.full(len(rows), np.nan)
    df = pd.DataFrame({"series_id": CUBE["ids"], "item_id": [i for i, _ in items], "store_id": [s for _, s in items]})
    df["dept_id"] = df["item_id"].str.rsplit("_", n=1).str[0]
    df["cat_id"] = df["item_id"].str.split("_").str[0]
    df["state_id"] = df["store_id"].str[:2]
    df["forecast"] = daily.sum(1); df["p10"] = p10; df["p90"] = p90
    df["recent_daily"] = recent
    df["assumed_stock"] = recent * ASSUMED_STOCK_DAYS
    df["shortage"] = (df["forecast"] - df["assumed_stock"]).clip(lower=0)
    df["excess"] = (df["assumed_stock"] - df["forecast"]).clip(lower=0)
    df["upside_risk"] = (df["p90"] - df["assumed_stock"]).clip(lower=0)
    df["price"] = last_price
    df["shortage_value"] = df["shortage"] * np.nan_to_num(last_price)
    df = df[rows >= 0].reset_index(drop=True)
    daily = daily[rows >= 0]
    days_idx = pd.date_range(pd.Timestamp(as_of) + pd.Timedelta(days=1), periods=days)
    _table_cache[key] = (df, daily, days_idx, CUBE["origins"][t])
    return _table_cache[key]

STOCK_NOTE = (f"M5 has no inventory data, so stock on hand is ASSUMED to be {ASSUMED_STOCK_DAYS} days of each item's average daily "
              "sales over the last 4 weeks; shortage = forecast minus assumed stock, excess = assumed stock minus forecast.")

def _fmt_items(d, col, n, unit="units"):
    return "; ".join(f"{r.item_id} at {r.store_id} {getattr(r, col):,.1f} {unit}" for r in d.head(n).itertuples())

def forecast_lookup(kind, as_of, days=LOOKUP_DAYS, n=10, stores=None):
    """Computes the answer to a forecast-data question. Returns {kind, text, table, keys}: `text` becomes passage [E1],
    `keys` are the strings a correct answer must contain (used by the validation matrix)."""
    df, daily, days_idx, made = forecast_table(as_of, days)
    span = f"{days_idx[0]:%Y-%m-%d} to {days_idx[-1]:%Y-%m-%d}"
    head = (f"Forecast data from the M5 forecasting model (LightGBM, forecast made on {made:%Y-%m-%d}) for the next {days} days, "
            f"{span}, {len(df):,} item-store series in the model's 50% sample. ")
    uses_stock = kind in ("stockout_risk", "risk_by_day", "replenish_items", "replenish_stores", "dept_shortage", "compare_stores", "excess_items")
    keys, table = [], None
    if kind in ("top_items", "low_items"):
        d = df.sort_values("forecast", ascending=(kind == "low_items"))
        table = d.head(n)[["item_id", "store_id", "forecast", "p10", "p90"]]
        text = f"{'Highest' if kind == 'top_items' else 'Lowest'} predicted demand: " + _fmt_items(d, "forecast", n) + "."
        keys = [d.iloc[0]["item_id"]]
    elif kind == "by_category":
        g = df.groupby("cat_id")["forecast"].sum().sort_values(ascending=False)
        table = g.to_frame("forecast").assign(share_pct=lambda x: 100 * x["forecast"] / x["forecast"].sum())
        text = "Predicted demand by category, all stores: " + "; ".join(f"{c.title()} {v:,.0f} units ({100 * v / g.sum():.0f}%)" for c, v in g.items()) + "."
        keys = [g.index[0], g.index[0].title()]
    elif kind == "by_store":
        g = df.groupby("store_id")["forecast"].sum().sort_values(ascending=False)
        table = g.to_frame("forecast")
        text = "Predicted demand by store: " + "; ".join(f"{s} {v:,.0f} units" for s, v in g.items()) + "."
        keys = [g.index[0]]
    elif kind == "daily_profile":
        tot = daily.sum(0)
        table = pd.DataFrame({"date": days_idx.date, "weekday": days_idx.day_name(), "forecast": tot})
        peak, low = int(np.argmax(tot)), int(np.argmin(tot))
        text = ("Total predicted demand by day: " + "; ".join(f"{d:%a %Y-%m-%d} {v:,.0f} units" for d, v in zip(days_idx, tot))
                + f". Highest on {days_idx[peak]:%A} ({tot[peak]:,.0f}), lowest on {days_idx[low]:%A} ({tot[low]:,.0f}).")
        keys = [days_idx[peak].day_name(), f"{days_idx[peak]:%Y-%m-%d}"]
    elif kind == "store_spread":
        g = df.groupby("item_id")["forecast"].agg(["max", "min", "mean", "count"])
        g = g[g["count"] >= 3].assign(spread=lambda x: x["max"] - x["min"]).sort_values("spread", ascending=False)
        top = g.head(n)
        best = {i: df[df["item_id"] == i].sort_values("forecast") for i in top.index}
        table = top
        text = ("Items with the largest difference in predicted demand across stores (items forecast in 3+ stores): "
                + "; ".join(f"{i} from {b.iloc[0]['forecast']:,.1f} units at {b.iloc[0]['store_id']} to {b.iloc[-1]['forecast']:,.1f} at {b.iloc[-1]['store_id']}"
                            for i, b in best.items()) + ".")
        keys = [top.index[0]] if len(top) else []
    elif kind == "stockout_risk":
        d = df.sort_values("upside_risk", ascending=False)
        table = d.head(n)[["item_id", "store_id", "forecast", "p90", "assumed_stock", "upside_risk"]]
        text = ("Items with the greatest stock-out risk: units by which demand at the high end of the forecast range (sum of daily P90) "
                "would exceed the assumed stock: " + _fmt_items(d, "upside_risk", n) + ".")
        keys = [d.iloc[0]["item_id"]]
    elif kind == "risk_by_day":
        cum = np.cumsum(daily, axis=1)
        at_risk = (cum > df["assumed_stock"].to_numpy()[:, None]).sum(0)
        table = pd.DataFrame({"date": days_idx.date, "items_out_of_stock": at_risk, "share_pct": 100 * at_risk / len(df)})
        text = ("Number of items whose cumulative forecast exceeds the assumed stock, by day (risk of a stock-out by that day): "
                + "; ".join(f"day {k + 1} ({d:%a %Y-%m-%d}) {v:,} items ({100 * v / len(df):.1f}%)" for k, (d, v) in enumerate(zip(days_idx, at_risk)))
                + f". The risk {'rises' if at_risk[-1] > at_risk[0] else 'does not rise'} over the {days} days.")
        keys = [f"{at_risk[-1]:,}", str(at_risk[-1])]
    elif kind == "replenish_items":
        d = df.sort_values("shortage_value", ascending=False)
        table = d.head(n)[["item_id", "store_id", "forecast", "assumed_stock", "shortage", "price", "shortage_value"]]
        text = ("Replenishment priority: items with the largest shortage value (shortage units x latest price): "
                + "; ".join(f"{r.item_id} at {r.store_id} short {r.shortage:,.1f} units, ${r.shortage_value:,.0f}" for r in d.head(n).itertuples()) + ".")
        keys = [d.iloc[0]["item_id"]]
    elif kind == "replenish_stores":
        g = df.groupby("store_id")[["shortage", "shortage_value"]].sum().sort_values("shortage_value", ascending=False)
        table = g
        text = ("Stores ranked for extra inventory by shortage value (shortage units x latest price): "
                + "; ".join(f"{s} {r.shortage:,.0f} units short, ${r.shortage_value:,.0f}" for s, r in g.iterrows()) + ".")
        keys = [g.index[0]]
    elif kind == "dept_shortage":
        g = df.groupby("dept_id")[["shortage", "shortage_value"]].sum().sort_values("shortage", ascending=False)
        table = g
        text = ("Predicted shortage by department, all stores: " + "; ".join(f"{dp} {r.shortage:,.0f} units (${r.shortage_value:,.0f})" for dp, r in g.iterrows())
                + ". Top items to prioritise in the largest one: "
                + _fmt_items(df[df["dept_id"] == g.index[0]].sort_values("shortage_value", ascending=False), "shortage", 5) + ".")
        keys = [g.index[0]]
    elif kind == "compare_stores":
        stores = stores or []
        g = df[df["store_id"].isin(stores)].groupby("store_id").agg(shortage=("shortage", "sum"), shortage_value=("shortage_value", "sum"),
                                                                    items_short=("shortage", lambda x: int((x > 0).sum())), forecast=("forecast", "sum"))
        table = g
        text = ("Predicted shortages compared: " + "; ".join(f"{s}: {r.shortage:,.0f} units short across {r.items_short} items "
                                                             f"(${r.shortage_value:,.0f}), total forecast {r.forecast:,.0f} units" for s, r in g.iterrows()) + ".")
        keys = [g["shortage"].idxmax()] if len(g) else []
    elif kind == "excess_items":
        d = df.sort_values("excess", ascending=False)
        table = d.head(n)[["item_id", "store_id", "forecast", "assumed_stock", "excess"]]
        text = "Items with the most excess stock (assumed stock minus forecast): " + _fmt_items(d, "excess", n) + "."
        keys = [d.iloc[0]["item_id"]]
    else:
        raise ValueError(f"unknown lookup kind {kind}")
    if uses_stock:
        text += " " + STOCK_NOTE
    return {"kind": kind, "text": head + text, "table": table, "keys": list(dict.fromkeys(str(k) for k in keys)), "uses_stock": uses_stock,
            "forecast_made": made}

def route_question(question):
    """Which forecast lookup (if any) a question needs. Returns (kind, kwargs) or (None, {})."""
    q = question.lower()
    stores = re.findall(r"\b(?:CA|TX|WI)_\d\b", question.upper())
    demand = "demand" in q
    if len(stores) >= 2 and "compare" in q:
        return "compare_stores", {"stores": stores}
    if "excess" in q:
        return "excess_items", {}
    if "department" in q and "shortage" in q:
        return "dept_shortage", {}
    if "store" in q and any(w in q for w in ("first", "prioritiz", "priority")):
        return "replenish_stores", {}
    if any(w in q for w in ("prioritiz", "priority")) and any(w in q for w in ("replenish", "product")):
        return "replenish_items", {}
    stockout = any(w in q for w in ("running out", "stockout", "stock-out", "out of stock"))
    if "risk" in q and stockout and "change" in q:
        return "risk_by_day", {}
    if "risk" in q and stockout:
        return "stockout_risk", {}
    if ("difference" in q or "multiple stores" in q) and "store" in q and demand:
        return "store_spread", {}
    if demand and "change" in q and ("day" in q or "week" in q):
        return "daily_profile", {}
    if "category" in q and demand and any(w in q for w in ("highest", "most")):
        return "by_category", {}
    if "store" in q and demand and any(w in q for w in ("highest", "most")):
        return "by_store", {}
    if any(w in q for w in ("product", "item")) and demand and "highest" in q:
        return "top_items", {}
    if any(w in q for w in ("product", "item")) and demand and "lowest" in q:
        return "low_items", {}
    return None, {}


# ============================================================== 8. The agent: recommend() for an order, ask() for a question
DISTRICT_OF = {"TX": "Dallas", "CA": "San Francisco", "WI": "Chicago and Minneapolis"}

def scenario_queries(item_id, store_id, as_of, fc, flags=()):
    st = store_id.split("_")[0]
    cat = item_id.split("_")[0]
    as_of = pd.Timestamp(as_of)
    window = cal[(cal["date"] > as_of) & (cal["date"] <= fc["dates"][-1])]
    upcoming = sorted(set(window["event_name_1"].dropna()) | set(window["event_name_2"].dropna()))
    q = [f"{cat.lower()} sales pattern by weekday and month in {STATE_NAMES[st]}",
         f"year over year sales trend in {STATE_NAMES[st]}",
         f"weather effect on {cat.lower()} sales in {STATE_NAMES[st]}",
         f"recent severe weather, storm, snow, heat or flood in {STATE_NAMES[st]}",
         f"consumer spending and retail sales in the {DISTRICT_OF[st]} Federal Reserve district",
         "Walmart U.S. comparable store sales, traffic and management commentary"]
    q += [f"sales effect of {e}" for e in upcoming]
    if cat == "FOODS":
        q += [f"SNAP benefit days and food sales in {STATE_NAMES[st]}", "changes to SNAP food assistance benefits"]
    q.append(f"forecast accuracy and prediction interval for {fc['demand_type'].lower()} demand items")
    if fc["demand_type"] in ("Intermittent", "Lumpy"):
        q.append("intermittent demand: how reliable is the forecast for an item that sells on few days")
    codes = {f["code"] for f in flags}
    if codes & {"POSSIBLE_STOCKOUT", "POSSIBLY_DISCONTINUED"} or fc["zero_run"] >= 14:
        q.append("long run of zero sales: item not on sale, missing price, or out of stock")
    if "NEW_ITEM" in codes:
        q.append("zero sales before an item's first sale or with no listed price")
    return st, cat, upcoming, q

def gather_evidence(queries, as_of, state, category=None, per_query=2, max_items=12):
    best = {}
    for q in queries:
        res = retrieve(q, as_of, state, k=per_query, category=category)
        res = res[~(res["relevance"] < RELEVANCE_MIN)]                    # drop passages the reranker scored as irrelevant
        for i, r in res.iterrows():
            if i not in best or r["score"] > best[i]["score"]:
                best[i] = r
    ev = pd.DataFrame(best.values())
    if ev.empty:
        return ev
    ev = ev.sort_values("score", ascending=False).drop_duplicates("doc_id")    # one passage per document
    return ev.head(max_items)

def item_profile(item_id, store_id, as_of, fc, seg, flags, conf, history):
    """Passage [E1]: the item's segment, LightGBM's track record known by as_of, sales history, flags and forecast confidence."""
    as_of = pd.Timestamp(as_of)
    st, cat = store_id.split("_")[0], item_id.split("_")[0]
    seg_txt = f"Demand segment from 2011-2014 sales: {seg['demand_type']}"
    if seg["volume_level"]:
        seg_txt += f", {seg['volume_level'].lower()} volume"
    if np.isfinite(seg["adi"]):
        seg_txt += f" (sells on average every {seg['adi']:.1f} days)"
    parts = [f"Item profile for {item_id} at store {store_id} ({STATE_NAMES[st]}), using data available on {as_of:%Y-%m-%d}.", seg_txt + "."]
    tr = track_record(as_of)
    lines = []
    for kind, key, label in [("demand_type", seg["demand_type"], f"{seg['demand_type']} items"),
                             ("volume_level", seg["volume_level"], f"{str(seg['volume_level']).lower()}-volume items"),
                             ("category", cat, f"{cat.title()} items")]:
        r = tr.get((kind, key))
        if key and r and r["weeks"] >= TRACK_MIN_WEEKS:
            lines.append(f"{label} {r['wape']:.1f}% WAPE, bias {r['bias']:+.1f}%")
    if lines:
        weeks = max(r["weeks"] for r in tr.values())
        parts.append(f"LightGBM's 14-day forecast error over the {weeks} weekly forecasts scored by {max(r['through'] for r in tr.values()):%Y-%m-%d} "
                     f"(50% sample): " + "; ".join(lines) + ". WAPE is total absolute error as a share of units sold; "
                     "negative bias means the model forecast too low.")
    else:
        parts.append("No LightGBM forecasts had been scored by this date, so its error for this segment is not yet known.")
    if history["first"] is None:
        parts.append("Sales history: no sales yet at this store.")
    else:
        parts.append(f"Sales history: first sale {dates[history['first']]:%Y-%m-%d} ({history['days_selling']} days ago); "
                     f"last sale {dates[history['last']]:%Y-%m-%d}; current run of zero-sale days: {history['zero_run']}.")
    parts += [f"Flag {f['code']}: {f['message']}. Check: {f['check']}." for f in flags] or ["Flags: none."]
    nf = fc.get("new_item_forecast")
    if nf:
        e = nf["new_item"]
        parts.append(f"New-item forecast: {e['total']:.1f} units over {ORDER_DAYS} days (P10 {e['p10']:.1f}, P90 {e['p90']:.1f}), "
                     f"from {e['n_similar']} similar items launched in the past two years ({e['group']}, {e['band']} price) at the same age"
                     + (f", {100 * e['own_weight']:.0f}% weight on the item's own sales so far" if e["own_weight"] > 0 else "") + ".")
    parts.append(f"Forecast confidence: {conf['level']} ({'; '.join(conf['reasons'])}).")
    return {"doc_type": "item_profile", "source": "Forecast, segment and order data", "title": f"Item profile: {item_id} at {store_id}",
            "url": "", "published": as_of, "state": st, "text": " ".join(parts), "doc_id": "item_profile", "chunk_id": "item_profile",
            "score": np.nan}

SYSTEM_PROMPT = """You are an inventory decision-support assistant for a Walmart store manager.
You receive a demand forecast, an order computed by a fixed ordering rule, and numbered evidence passages [E1], [E2], ...
[E1] is always the item profile: its demand segment, the forecast model's track record, sales history, flags and forecast confidence.
Rules:
- Use only the forecast numbers and the evidence passages. Do not use outside knowledge about what happened after the decision date.
- Cite the evidence number for every reason, e.g. [E3]. Use only numbers that appear in the prompt; do not calculate new ones.
- Order quantity: if "hold" is false, recommend "order_qty" units. You may suggest an adjustment only if a cited passage gives a concrete reason; state the adjusted number and why.
- If "hold" is true, the Recommendation line must say "Hold: order 0 until checked", and then give "if_confirmed_order_qty" as the number to order if the manager confirms the item is in stock and still sold.
- Restate the forecast confidence level exactly as given in [E1], with its main reason.
- In the first reason under "Why", name the store's state (California, Texas or Wisconsin) and the item's category (Foods, Hobbies or Household).
- If no passage besides [E1] is relevant to this item, state, or period, say "No supporting evidence found".
- List every flag's check under "Manager checks". Evidence about weather is state-level; store locations are unknown. Do not claim knowledge of future weather.
- Be brief and concrete. Format:
Recommendation: <one sentence with the quantity>
Forecast confidence: <High, Medium or Low> - <main reason> [E1]
Why:
- <reason with citation>
Risks and uncertainty:
- <point with citation where possible>
Manager checks: <each flag's check, or "none">
Evidence used: <list of E numbers>"""

def build_prompt(item_id, store_id, as_of, fc, qty, ev, upcoming):
    lines = [f"Decision date: {pd.Timestamp(as_of):%Y-%m-%d}. Item {item_id} at store {store_id}.",
             f"Forecast model: {fc.get('model', 'baseline')}. Next {len(fc['mean'])} days: total {fc['mean'].sum():.1f} units, "
             f"{fc['mean'][0]:.2f} units per day, 80% range per day {fc['p10'][0]:.2f} to {fc['p90'][0]:.2f} (P10 to P90). "
             f"Units sold in the last 28 days: {fc['last_28_units']}.",
             f"Calendar events in the forecast window: {', '.join(upcoming) if upcoming else 'none'}.",
             f"Ordering rule result: {json.dumps(qty)}", "", "Evidence:"]
    if ev is None or len(ev) == 0:
        lines.append("(none provided)")
        return "\n".join(lines)
    for n, (_, r) in enumerate(ev.iterrows(), 1):
        lines.append(f"[E{n}] ({r['doc_type']}, {r['source']}, published {r['published']:%Y-%m-%d}, {r['state']}) {r['title']}\n{r['text'][:1500]}")
    return "\n".join(lines)

def get_secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)

def generate_text(system, prompt, max_tokens=1200):
    """Send the prompt to the OpenRouter model. Returns None if no key is set; retries once on an empty reply."""
    key = get_secret("OPENROUTER_API_KEY")
    if not key:
        return None
    from openai import OpenAI                     # OpenRouter uses the same request format as OpenAI
    client = OpenAI(base_url=OPENROUTER_URL, api_key=key, timeout=LLM_TIMEOUT, max_retries=3)   # retries 429/5xx itself
    text = ""
    for attempt in range(2):
        # Many free models "think" before answering and that thinking counts toward max_tokens, so allow plenty of room
        resp = client.chat.completions.create(model=OPENROUTER_MODEL, max_tokens=max(max_tokens, 6000),
                                              messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        text = resp.choices[0].message.content or ""
        if text.strip():
            return text
        print(f"  {OPENROUTER_MODEL} returned no text (finish_reason: {resp.choices[0].finish_reason})"
              + ("; retrying in 10 seconds" if attempt == 0 else "; try another free model"))
        if attempt == 0:
            time.sleep(10)
    return text

def place_order(item_id, store_id, as_of, on_hand, on_order=0, lead_time_days=7, review_days=7, service_level=0.95, forecast=None):
    """Forecast + safety stock + ordering rule + flags + hold rule + forecast confidence."""
    fc = forecast or baseline_forecast(item_id, store_id, as_of)
    seg = segment_of(item_id, store_id)
    fc = {**fc, "demand_type": seg["demand_type"], "adi": seg["adi"], "cv2": seg["cv2"], "volume_level": seg["volume_level"]}
    history = sales_history(item_id, store_id, as_of)
    safety = item_safety_stock(item_id, store_id, as_of, lead_time_days + review_days, service_level)
    qty = order_quantity(fc["mean"], fc["sd"], on_hand, on_order, lead_time_days, review_days, service_level, safety)
    flags = item_flags(item_id, store_id, as_of, seg)
    computed = qty["order_qty"]
    qty["computed_order_qty"] = computed
    qty["hold"] = any(f["hold"] for f in flags)
    if qty["hold"]:
        confirmed = computed
        nf = new_item_forecast(item_id, store_id, as_of) if any(f["code"] == "NEW_ITEM" for f in flags) else None
        if nf:                                       # new item: order from the similar-items forecast instead
            fc["new_item_forecast"] = nf
            confirmed = order_quantity(nf["mean"], nf["sd"], on_hand, on_order, lead_time_days, review_days, service_level, nf["safety"])["order_qty"]
        qty["order_qty"] = 0
        qty["if_confirmed_order_qty"] = int(max(0, confirmed))
        qty["hold_reasons"] = [f["code"] for f in flags if f["hold"]]
    conf = forecast_confidence(fc, seg, flags, as_of, history)
    return {"forecast": fc, "order": qty, "segment": seg, "flags": flags, "confidence": conf, "history": history}

def recommend(item_id, store_id, as_of, on_hand, on_order=0, lead_time_days=7, review_days=7, service_level=0.95,
              forecast=None, call_llm=True, verbose=True):
    po = place_order(item_id, store_id, as_of, on_hand, on_order, lead_time_days, review_days, service_level, forecast)
    fc, qty = po["forecast"], po["order"]
    st, cat, upcoming, queries = scenario_queries(item_id, store_id, as_of, fc, po["flags"])
    profile = item_profile(item_id, store_id, as_of, fc, po["segment"], po["flags"], po["confidence"], po["history"])
    ev = pd.concat([pd.DataFrame([profile]), gather_evidence(queries, as_of, st, category=cat)], ignore_index=True)
    prompt = build_prompt(item_id, store_id, as_of, fc, qty, ev, upcoming)
    answer = generate_text(SYSTEM_PROMPT, prompt) if call_llm else None
    if call_llm and answer is None:
        print("No OPENROUTER_API_KEY found: showing the evidence only (the prompt is in result['prompt']).")
    if verbose:
        print(f"=== {item_id} @ {store_id}, as of {pd.Timestamp(as_of):%Y-%m-%d} ===")
        print(f"Forecast: {fc['model']}")
        print(f"Segment: {po['segment']['demand_type']}, {po['segment']['volume_level']} volume | confidence: {po['confidence']['level']}")
        if qty["hold"]:
            print(f"Order rule: HOLD, order 0 until checked ({', '.join(qty['hold_reasons'])}); if confirmed, order {qty['if_confirmed_order_qty']} units")
        else:
            print(f"Order rule: order {qty['order_qty']} units (expected demand {qty['expected_demand']} over {qty['protection_days']} days, "
                  f"safety stock {qty['safety_stock']}, on hand {on_hand})")
        for f in po["flags"]:
            print(f"FLAG {f['code']}: {f['message']}")
        print(f"Upcoming events: {', '.join(upcoming) or 'none'}\n")
        show(ev.iloc[1:], width=160)
        if answer:
            print(f"--- {OPENROUTER_MODEL} ---\n" + answer)
    return {**po, "evidence": ev, "prompt": prompt, "answer": answer}

ASK_PROMPT = """You answer questions from a Walmart store manager using numbered evidence passages [E1], [E2], ...
Rules:
- Use only the evidence passages. Do not use outside knowledge about what happened after the decision date.
- Cite the evidence number for every reason. Use only numbers that appear in the passages; do not calculate new ones.
- A passage of type forecast_data was computed from the M5 forecasting model: use it for "which", "how much" and "when" questions and
  name the top results it lists. If it says stock on hand is assumed, say that inventory is an assumption, not data.
- If the passages do not answer the question, start the answer with "No supporting evidence found" and say what data would be needed.
- Evidence about weather and the economy is state-level; store cities are unknown.
- Be brief and concrete. Format:
Answer: <two to four sentences>
Why:
- <reason with citation>
What the data cannot tell us:
- <limit, or "nothing important">
Evidence used: <list of E numbers>"""

def ask(question, as_of, state=None, category=None, k=8, call_llm=True, verbose=True):
    """Answer a manager's question with evidence published on or before as_of. Forecast questions also get the
    forecast-data lookup as passage [E1]."""
    category = category or category_of(question)
    kind, kw = route_question(question)
    lookup = forecast_lookup(kind, as_of, **kw) if kind else None
    ev = retrieve(question, as_of, state, k=(5 if lookup else k), category=category)
    ev = ev[~(ev["relevance"] < RELEVANCE_MIN)].reset_index(drop=True)
    if lookup:
        row = {"doc_type": "forecast_data", "source": "M5 forecasting model (forecast cube)", "title": f"Forecast data: {kind.replace('_', ' ')}",
               "url": "", "published": pd.Timestamp(as_of), "state": state or "US", "text": lookup["text"], "doc_id": "forecast_data",
               "chunk_id": "forecast_data", "score": np.nan, "relevance": np.nan, "category": "ALL"}
        ev = pd.concat([pd.DataFrame([row]), ev], ignore_index=True)
    prompt = (f"Decision date: {pd.Timestamp(as_of):%Y-%m-%d}. Store state: {STATE_NAMES.get(state, 'any')}. "
              f"Question: {question}\n\nEvidence:\n" + ("\n".join(
                  f"[E{n}] ({r['doc_type']}, {r['source']}, published {r['published']:%Y-%m-%d}, {r['state']}) {r['title']}\n{r['text'][:2000]}"
                  for n, (_, r) in enumerate(ev.iterrows(), 1)) or "(no relevant passages found)"))
    answer = generate_text(ASK_PROMPT, prompt) if call_llm else None
    if verbose:
        print(f"=== {question} (as of {pd.Timestamp(as_of):%Y-%m-%d}, {state or 'any state'}) ===")
        if lookup:
            print("Forecast lookup:", kind)
        show(ev, width=140)
        print(answer or "(no answer: no key, or call_llm=False)")
    return {"question": question, "as_of": as_of, "state": state, "evidence": ev, "prompt": prompt, "answer": answer, "lookup": lookup}

# ============================================================== 9. Test questions
# (id, question, decision date, state, expected document types, expected terms, kind, forecast lookup)
# kind: hit / fresh / no_future / no_evidence are retrieval tests (notebook 2); "data" questions are answered from the forecast
# cube and checked in notebook 3 against the lookup's answer. Tests 1-16 come from v1-v2, 17-36 and 37-56 from the team's lists.
TESTS = [
    (1,  "Why might Hobbies sales drop in Wisconsin stores this week?", "2014-01-06", "WI", ["evidence_card", "storm_event"], ["snow", "winter"], "hit", None),
    (2,  "Is unusually hot weather likely to change Foods demand in Texas?", "2011-08-01", "TX", ["storm_event"], ["heat"], "hit", None),
    (3,  "Does rain meaningfully affect sales in California?", "2015-02-01", "CA", ["evidence_card"], ["rain"], "hit", None),
    (4,  "What happens to store sales on Christmas Day?", "2014-12-15", None, ["evidence_card"], ["christmas"], "hit", None),
    (5,  "Which holidays lower sales and which raise them?", "2014-11-01", None, ["evidence_card"], ["event"], "hit", None),
    (6,  "Which weekday and month sell the most?", "2014-06-01", None, ["evidence_card"], ["weekday"], "hit", None),
    (7,  "How could changes to SNAP benefits affect Foods demand?", "2013-12-01", None, ["policy_doc", "evidence_card", "reference"], ["snap"], "hit", None),
    (8,  "What were retailers in the Dallas Fed district reporting about consumer spending?", "2015-03-15", "TX", ["beige_book"], ["dallas"], "hit", None),
    (9,  "This item sells on only a few days a month. How reliable is its forecast?", "2015-06-01", None, ["evidence_card"], ["intermittent"], "hit", None),
    (10, "The item shows zero sales for three weeks with no listed price. Is that zero demand?", "2015-06-01", None, ["reference"], ["price"], "hit", None),
    (11, "What did Walmart management say about weather and US sales last quarter?", "2014-06-01", None, ["sec_filing", "walmart_doc"], ["weather"], "hit", None),
    (12, "A major winter storm is forecast for Wisconsin. What should a store manager expect?", "2015-01-20", "WI", ["evidence_card", "storm_event"], ["snow"], "hit", None),
    (13, "What caused the Wisconsin sales dip in winter 2013-14?", "2013-06-01", "WI", None, [], "no_future", None),
    (14, "What did the latest Beige Book say about the San Francisco district?", "2012-05-01", "CA", ["beige_book"], ["san francisco"], "fresh", None),
    (15, "How did the COVID-19 pandemic affect demand?", "2016-05-22", None, None, [], "no_evidence", None),
    (16, "What is the weather forecast for CA_3's city next week?", "2016-05-22", "CA", ["reference"], ["location"], "hit", None),
    # Manager questions (team list 1, tests 17-26)
    (17, "My California store's total sales forecast looks reasonable, but some categories are selling more than expected. Could the overall forecast be hiding problems at the category level?", "2015-09-01", "CA", ["evidence_card"], ["forecast", "category"], "hit", None),
    (18, "Do Foods products show similar seasonal sales patterns in Texas and Wisconsin, or are there important regional differences?", "2015-07-01", None, ["evidence_card"], ["foods", "seasonal"], "hit", None),
    (19, "Some products at my California store are selling more units despite lower prices. Does higher sales volume necessarily mean higher revenue?", "2015-05-15", "CA", ["evidence_card", "reference"], ["price", "revenue"], "hit", None),
    (20, "My Wisconsin store occasionally sells much more than predicted. What might explain these unexpected demand spikes, and are they common in historical sales?", "2015-10-01", "WI", ["evidence_card"], ["spike", "unexpected"], "hit", None),
    (21, "I need to plan inventory for the next two weeks, but the forecast only covers seven days. What are the limitations of using that forecast for a longer ordering period?", "2015-11-01", None, ["reference"], ["horizon", "seven days"], "hit", None),
    (22, "Two products in my California store have similar expected sales, but one has much less predictable demand. Why might they require different amounts of safety stock?", "2015-08-01", "CA", ["evidence_card", "reference"], ["safety stock", "predictab"], "hit", None),
    (23, "My Texas store's sales are growing, but I'm unsure whether the improvement is happening across the region. How do Walmart's reported sales trends compare with the Dallas Fed's view of local consumer spending?", "2015-06-15", "TX", ["sec_filing", "walmart_doc", "beige_book"], ["dallas", "comparable"], "hit", None),
    (24, "Our supplier now takes longer to deliver Household products. How could a longer delivery time affect inventory risk even if customer demand stays the same?", "2015-09-15", None, ["reference"], ["lead time"], "hit", None),
    (25, "Can I determine which of two inventory strategies is more profitable using only historical sales and demand forecasts, without actual product margins or inventory costs?", "2015-12-01", None, None, [], "no_evidence", None),
    (26, "I'm reviewing a replenishment decision made at my Texas store in 2014. How can I distinguish what historical sales and business reports showed at the time from information published only afterward?", "2014-10-15", "TX", ["evidence_card", "walmart_doc"], ["sales"], "hit", None),
    # Manager questions (team list 2, tests 27-36)
    (27, "A product I order for a Wisconsin store sold well last year, but the forecast for it now looks low. What history should I check before I decide how much to order?", "2015-08-20", "WI", ["evidence_card"], ["bias", "forecast accuracy"], "hit", None),
    (28, "How much should I raise orders before Labor Day weekend?", "2015-08-24", None, ["evidence_card"], ["labor day"], "hit", None),
    (29, "Do the Federal Reserve and Walmart's own filings agree on how consumer spending is holding up?", "2016-03-01", None, ["beige_book", "sec_filing", "walmart_doc"], ["consumer spending"], "hit", None),
    (30, "Did price changes drive Household sales in California, or was it something else?", "2015-04-06", "CA", ["evidence_card"], ["price"], "hit", None),
    (31, "Why is the forecast for fast-selling Foods items more reliable than for slow-selling Hobbies items?", "2015-09-01", None, ["evidence_card"], ["demand type", "error"], "hit", None),
    (32, "I can't review every product in my California store. Which ones deserve the most attention when I place orders?", "2015-07-06", "CA", ["evidence_card"], ["revenue", "priority"], "hit", None),
    (33, "My plan is to order 40 units but the system recommends 25. Which one is safer and why?", "2015-06-08", None, ["evidence_card", "reference"], ["uncertainty", "safety stock", "range"], "hit", None),
    (34, "A storm report shows severe weather in Texas, but sales don't show a clear drop. How should I reconcile the two before changing my Foods order?", "2015-10-15", "TX", ["storm_event", "evidence_card"], ["storm", "weather"], "hit", None),
    (35, "Which city in California has the highest Hobbies demand, so I can send more stock there?", "2016-02-20", "CA", None, [], "no_evidence", None),
    (36, "What is the exact inventory on hand at each Wisconsin store, and how many units did we lose to theft last month?", "2016-01-15", "WI", None, [], "no_evidence", None),
    # Team question list 3 (tests 37-56). "data" = answered from the forecast cube (forecast-data lookup); retrieval isn't scored
    (37, "Which products are expected to have the highest demand over the next 7 days?", "2015-04-22", None, None, [], "data", "top_items"),
    (38, "Which product category has the highest predicted demand across all stores?", "2015-07-15", None, None, [], "data", "by_category"),
    (39, "Which Walmart store has the highest predicted demand over the next 7 days?", "2015-10-21", None, None, [], "data", "by_store"),
    (40, "How does predicted demand change throughout the next 7 days?", "2016-01-14", None, None, [], "data", "daily_profile"),
    (41, "Which products show the largest differences in predicted demand across stores?", "2015-02-11", None, None, [], "data", "store_spread"),
    (42, "Which products have the greatest risk of running out of stock based on current inventory and predicted demand?", "2015-04-22", None, None, [], "data", "stockout_risk"),
    (43, "How does the risk of stockouts change over the next 7 days?", "2015-07-15", None, None, [], "data", "risk_by_day"),
    (44, "Which products should we prioritize for replenishment, and why?", "2015-10-21", None, None, [], "data", "replenish_items"),
    (45, "Which stores should receive additional inventory first if supply is limited?", "2016-01-14", None, None, [], "data", "replenish_stores"),
    (46, "Which departments are expected to have the largest inventory shortages, and what should managers prioritize?", "2016-03-10", None, None, [], "data", "dept_shortage"),
    (47, "Compare the predicted inventory shortages between WI_3 and CA_4 over the next 7 days.", "2015-02-11", None, None, [], "data", "compare_stores"),
    (48, "Are there any products with excess inventory compared to their predicted demand?", "2016-03-10", None, None, [], "data", "excess_items"),
    (49, "Why might HOBBIES products have more unpredictable demand than FOODS products?", "2015-07-15", None, ["evidence_card"], ["demand type", "intermittent"], "hit", None),
    (50, "How do demand patterns differ between FOODS, HOUSEHOLD, and HOBBIES?", "2015-10-21", None, ["evidence_card"], ["seasonal", "demand type"], "hit", None),
    (51, "Which Walmart store generated the most profit last year?", "2016-01-14", None, None, [], "no_evidence", None),
    (52, "Which products received the most customer complaints last month?", "2015-10-21", None, None, [], "no_evidence", None),
    (53, "Which Walmart store currently has the most employees working?", "2015-07-15", None, None, [], "no_evidence", None),
    (54, "Which products had the highest return rates last month?", "2016-03-10", None, None, [], "no_evidence", None),
    (55, "Which Walmart store had the highest customer satisfaction score this year?", "2015-12-16", None, None, [], "no_evidence", None),
    (56, "Which suppliers had the longest delivery delays last quarter?", "2015-09-16", None, None, [], "no_evidence", None),
]
NO_DATA_CARDS = {"ref:data_limits", "ref:stores"}      # a no-answer question is handled if one of these comes first
def set_paths(project_dir, m5_dir=None):
    """Configure project paths for Streamlit or local Python."""
    global PROJECT_DIR, M5_DIR, RAG_DIR, RAW_DIR
    global FORECAST_DIR, CUBE_FILE, SEGMENTS_FILE
    global SEG_ERROR_FILE, SETTINGS_FILE

    PROJECT_DIR = str(Path(project_dir).resolve()) + "/"
    M5_DIR = str(Path(m5_dir).resolve()) + "/" if m5_dir else PROJECT_DIR + "M5/datasets/"
    RAG_DIR = PROJECT_DIR + "RAG/"
    RAW_DIR = RAG_DIR + "raw/"
    FORECAST_DIR = PROJECT_DIR + "forecasting/"

    CUBE_FILE = FORECAST_DIR + "forecast_cube_lightgbm_50.npz"
    SEGMENTS_FILE = FORECAST_DIR + "series_segments_50.csv"
    SEG_ERROR_FILE = FORECAST_DIR + "lightgbm_error_by_segment_50.csv"
    SETTINGS_FILE = RAG_DIR + f"settings_{RUN_TAG}.json"
    SEG_ERROR_FILE = FORECAST_DIR + "lightgbm_error_by_segment_50.csv"
    SETTINGS_FILE = RAG_DIR + f"settings_{RUN_TAG}.json"

def setup(m5=True, library=True, forecasts=True, reranker=True):
    """Loads what a notebook needs. m5: sales, calendar, prices. forecasts: forecast cube, segments, new-item table (needs m5).
    library: passages, embeddings, keyword index, reranker (built by notebook 1)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    t0 = time.time()
    if m5 or forecasts:
        load_m5()
    if forecasts:
        load_forecasts()
    if library:
        load_library(reranker)
    print(f"Setup done in {time.time() - t0:.0f}s | files are read from and saved to {RAG_DIR} (names end in _{RUN_TAG})")

