"""
LLM narrative generation utilities for SalesForecasting XAI V2.

RAG architecture: chunk XAI outputs → embed with Gemini → filter/retrieve → generate narratives.
RAGAS evaluation: optional faithfulness + answer_relevancy scoring (requires ragas package).

Public API
----------
build_knowledge_base(outputs_dir, manifest_df, top_n=10, client=None) -> KnowledgeBase
generate_narrative(narrative_type, context_chunks, params, client, persona, sleep_s=5) -> str
evaluate_narratives(records, client, api_key) -> pd.DataFrame
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

KB_VERSION = "2"   # bump whenever chunk schema changes to auto-invalidate stale caches

# ── Persona system prompts ─────────────────────────────────────────────────────

PERSONAS = {
    "Business Executive": (
        "You are an expert Management Consultant briefing C-level executives on AI model performance. "
        "Use no statistical jargon. Translate all findings into commercial impact, strategic risk, "
        "and clear actionable decisions. Write in plain English. Be concise, direct, and focus on "
        "what the business should do about these insights."
    ),
    "Data Scientist": (
        "You are a Lead Data Scientist briefing the engineering and modelling team. "
        "Focus on mathematical magnitudes, feature contribution scales, potential collinearity risks, "
        "model mechanics, distributional shifts, and any anomalies that may affect model integrity. "
        "Reference specific numbers and feature names from the provided data."
    ),
    "Risk Manager": (
        "You are an Enterprise Risk Officer specialising in AI Governance and model risk management. "
        "Assess model sensitivity to high-weighted features, over-reliance risks, temporal stability, "
        "regulatory implications of data drift, and the reliability of using this model for "
        "long-term forecasting. Flag any governance concerns explicitly."
    ),
    "Data Custodian": (
        "You are a Senior Data Engineer responsible for data pipeline robustness and quality. "
        "Focus on upstream data sources, ETL failure impact, null rates, encoding anomalies, "
        "distribution drift across pipeline stages, and data quality trends. "
        "Prioritise operational data reliability over model performance commentary."
    ),
}

NARRATIVE_PERSONAS = {
    "overview":    ["Business Executive", "Data Scientist", "Risk Manager"],
    "explanation": ["Business Executive", "Data Scientist", "Risk Manager"],
    "drift":       ["Business Executive", "Data Scientist", "Risk Manager"],
    "lineage":     ["Data Custodian"],
    "comparison":  ["Business Executive", "Data Scientist", "Risk Manager"],
    "consensus":   ["Business Executive", "Data Scientist", "Risk Manager"],
}


# ── Knowledge base ─────────────────────────────────────────────────────────────

@dataclass
class KnowledgeBase:
    chunks: list = field(default_factory=list)
    texts:  list = field(default_factory=list)
    embs:   np.ndarray = field(default_factory=lambda: np.empty((0, 768), dtype=np.float32))

    def filter(self, **kwargs) -> list:
        return [c for c in self.chunks
                if all(str(c.get(k, "")) == str(v) for k, v in kwargs.items())]

    def search(self, query_emb: np.ndarray, top_k: int = 3) -> list:
        if self.embs.shape[0] == 0:
            return []
        norms = np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-8
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        sims = (self.embs / norms) @ q_norm
        idxs = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in idxs]


# ── Chunk text serialisers (used for embedding) ────────────────────────────────

def _chunk_to_text(chunk: dict) -> str:
    t = chunk.get("type", "")
    if t == "manifest":
        champion_str = " [CHAMPION MODEL — organisational standard]" if chunk.get("champion") == "Y" else ""
        return (f"Model {chunk['model']}{champion_str} is a {chunk['family']} regression model "
                f"for {chunk['domain']}. Uses {chunk['n_features']} features. "
                f"Target variable: {chunk['target']}. Group variable: {chunk['group_vars']}.")
    if t == "performance":
        return (f"{chunk['model']} performance on {chunk['dataset']} "
                f"({chunk['group_var']}={chunk['group_value']}): "
                f"MAE={chunk['MAE']:.2f}, MAPE={chunk['MAPE']:.4f}, "
                f"Accuracy={chunk['Accuracy_pct']:.1f}%, R²={chunk['R2']:.4f}.")
    if t == "importance":
        top5 = ", ".join(
            f"{f['feature']}({f['importance']:.4f})"
            for f in chunk["top_features"][:5]
        )
        return (f"{chunk['model']} {chunk['framework']} importance on {chunk['dataset']} "
                f"({chunk['group_var']}={chunk['group_value']}): {top5}.")
    if t == "importance_pivot":
        features = chunk.get("features", [])[:5]
        return (f"{chunk['model']} {chunk['framework']} pivoted importance "
                f"({chunk['pivot_var']}) on {chunk['dataset']}: "
                f"Top features across groups: {features}.")
    if t == "drift":
        drifted = [f.get("Feature", f.get("feature", "?")) for f in chunk.get("top_drifted", [])][:3]
        return (f"{chunk['model']} drift DEV→{chunk['eval_dataset']}: "
                f"Model PSI={chunk['model_PSI']:.4f} ({chunk['model_status']}). "
                f"Features: {chunk['n_stable']} Stable, {chunk['n_monitor']} Monitor, "
                f"{chunk['n_drifted']} Significant Drift. Top drifted: {drifted}.")
    if t == "descriptive":
        return (f"{chunk['dataset']} dataset: {chunk['n_records']:,} records, "
                f"{chunk['n_continuous']} continuous + {chunk['n_categorical']} categorical features. "
                f"Max null rate: {chunk['null_pct_max']:.1f}%.")
    if t == "consensus":
        strong = [f["feature"] for f in chunk["features"]
                  if f["consensus"] == "Strong Consensus"][:5]
        divergent = [f["feature"] for f in chunk["features"]
                     if f["consensus"] == "Divergent"][:3]
        return (f"{chunk['model']} explanation consensus ({', '.join(chunk['frameworks'])}) "
                f"on {chunk['dataset']} ({chunk['group_var']}={chunk['group_value']}): "
                f"{chunk['n_strong_consensus']} strong consensus, "
                f"{chunk['n_partial']} partial, {chunk['n_divergent']} divergent. "
                f"Top consensus features: {strong}. Divergent: {divergent}.")
    return json.dumps(chunk)


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed_texts(texts: list, client) -> np.ndarray:
    BATCH = 50
    all_embs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
            )
            all_embs.extend([list(e.values) for e in response.embeddings])
        except Exception as e:
            print(f"  [WARN] Embedding batch {i//BATCH} failed: {e}. Using zeros.")
            all_embs.extend([[0.0] * 768] * len(batch))
    return np.array(all_embs, dtype=np.float32)


# ── Chunk builders ─────────────────────────────────────────────────────────────

def _build_manifest_chunks(manifest_df: pd.DataFrame) -> list:
    seen = set()
    chunks = []
    for _, row in manifest_df.iterrows():
        model = row["model_name"]
        if model in seen or row.get("dataset", "") != "DEV":
            continue
        # Only keep one manifest chunk per model (from DEV row which has champion flag)
        is_champ = str(row.get("champion", "N")).strip().upper() == "Y"
        chunks.append({
            "type":       "manifest",
            "model":      model,
            "family":     row.get("model_family", ""),
            "domain":     row.get("model_domain", "Sales Forecasting"),
            "n_features": int(row.get("n_features", 0)),
            "target":     row.get("target_col", "UnitSales"),
            "group_vars": row.get("group_vars", "Segment"),
            "champion":   "Y" if is_champ else "N",
        })
        seen.add(model)
    return chunks


def _build_performance_chunks(perf_df: pd.DataFrame) -> list:
    chunks = []
    for _, row in perf_df.iterrows():
        chunks.append({
            "type":        "performance",
            "model":       row["Model"],
            "dataset":     row["Dataset"],
            "group_var":   row["Group_Variable"],
            "group_value": row["Group_Value"],
            "Samples":     int(row.get("Samples", 0)),
            "MAE":         float(row.get("MAE", 0)),
            "RMSE":        float(row.get("RMSE", 0)),
            "MAPE":        float(row.get("MAPE", 0)),
            "Accuracy_pct":float(row.get("Accuracy %", 0)),
            "R2":          float(row.get("R-Squared", 0)),
            "Adj_R2":      float(row.get("Adjusted R-Squared", 0)),
            "High_Conf_pct": float(row.get("High Confidence %", 0)),
        })
    return chunks


def _build_importance_chunks(shap_df: pd.DataFrame, lime_df: pd.DataFrame,
                              surr_df: pd.DataFrame, top_n: int = 10) -> list:
    chunks = []
    fw_map = [
        ("SHAP",      shap_df, "Mean_Abs_SHAP",    None),
        ("LIME",      lime_df, "Mean_Abs_LIME",     None),
        ("Surrogate", surr_df, "Feature_Importance","Selected"),
    ]
    for fw_name, df, imp_col, sel_col in fw_map:
        if df.empty or imp_col not in df.columns:
            continue
        sub = df[df[sel_col] == True].copy() if sel_col and sel_col in df.columns else df.copy()
        for (model, dataset, gvar, gval), grp in sub.groupby(
                ["Model", "Dataset", "Group_Variable", "Group_Value"]):
            top = grp.nlargest(top_n, imp_col)[["Feature", imp_col]].reset_index(drop=True)
            total = float(top[imp_col].sum()) or 1e-8
            top_list = [
                {
                    "rank":        i + 1,
                    "feature":     r["Feature"],
                    "importance":  round(float(r[imp_col]), 6),
                    "pct_of_total": round(float(r[imp_col]) / total * 100, 1),
                }
                for i, r in top.iterrows()
            ]
            chunks.append({
                "type":        "importance",
                "model":       model,
                "dataset":     dataset,
                "group_var":   gvar,
                "group_value": gval,
                "framework":   fw_name,
                "top_features": top_list,
            })
    return chunks


def _build_importance_pivot_chunks(shap_df: pd.DataFrame, lime_df: pd.DataFrame,
                                    surr_df: pd.DataFrame, top_n: int = 10) -> list:
    """
    Build pivot chunks for Decile and Segment explanation narratives.
    One chunk per (model, dataset, framework, pivot_var) with importance
    tabulated across all group values (D1-D10 or segments).
    """
    chunks = []
    fw_map = [
        ("SHAP",      shap_df, "Mean_Abs_SHAP",    None),
        ("LIME",      lime_df, "Mean_Abs_LIME",     None),
        ("Surrogate", surr_df, "Feature_Importance","Selected"),
    ]
    for pivot_var in ["Decile", "Segment"]:
        for fw_name, df, imp_col, sel_col in fw_map:
            if df.empty or imp_col not in df.columns:
                continue
            sub = df[df[sel_col] == True].copy() if sel_col and sel_col in df.columns else df.copy()
            sub = sub[sub["Group_Variable"] == pivot_var]
            if sub.empty:
                continue
            for (model, dataset), grp in sub.groupby(["Model", "Dataset"]):
                pivot = grp.pivot_table(
                    index="Feature", columns="Group_Value",
                    values=imp_col, aggfunc="mean"
                ).fillna(0.0)
                pivot["_total"] = pivot.sum(axis=1)
                pivot = pivot.sort_values("_total", ascending=False).head(top_n)
                pivot = pivot.drop(columns=["_total"])
                if pivot_var == "Decile":
                    col_order = sorted(
                        pivot.columns,
                        key=lambda c: int(c[1:]) if c.startswith("D") and c[1:].isdigit() else 99
                    )
                    pivot = pivot[col_order]
                pivot_dict = {feat: {col: round(float(v), 6) for col, v in row.items()}
                              for feat, row in pivot.iterrows()}
                chunks.append({
                    "type":      "importance_pivot",
                    "model":     model,
                    "dataset":   dataset,
                    "framework": fw_name,
                    "pivot_var": pivot_var,
                    "features":  list(pivot.index),
                    "pivot":     pivot_dict,
                    "columns":   list(pivot.columns),
                })
    return chunks


def _build_drift_chunks(model_psi_df: pd.DataFrame,
                         feat_psi_df: pd.DataFrame) -> list:
    chunks = []
    if model_psi_df.empty:
        return chunks
    for _, mrow in model_psi_df.iterrows():
        model   = mrow["Model"]
        eval_ds = mrow["Evaluating_Dataset"]
        bench   = mrow.get("Benchmark", "DEV")
        feat_sub = pd.DataFrame()
        if not feat_psi_df.empty:
            feat_sub = feat_psi_df[
                (feat_psi_df["Evaluating_Dataset"] == eval_ds) &
                (feat_psi_df["Assoc_Models"].isin([model, "All"]))
            ].copy()
        n_stable  = int((feat_sub["Status"] == "Stable").sum())          if not feat_sub.empty else 0
        n_monitor = int((feat_sub["Status"] == "Monitor").sum())         if not feat_sub.empty else 0
        n_drifted = int((feat_sub["Status"] == "Significant Drift").sum()) if not feat_sub.empty else 0
        top_drift = []
        if not feat_sub.empty:
            top_drift = (feat_sub[feat_sub["Status"] == "Significant Drift"]
                         .sort_values("Feature_PSI", ascending=False)
                         .head(10)[["Feature", "Feature_PSI", "Status"]]
                         .to_dict("records"))
        all_feat = []
        if not feat_sub.empty:
            all_feat = feat_sub[["Feature", "Feature_PSI", "Status"]].to_dict("records")
        chunks.append({
            "type":        "drift",
            "model":       model,
            "benchmark":   bench,
            "eval_dataset": eval_ds,
            "model_PSI":   round(float(mrow["Model_Stability_PSI"]), 6),
            "model_status": mrow["Status"],
            "n_stable":    n_stable,
            "n_monitor":   n_monitor,
            "n_drifted":   n_drifted,
            "top_drifted": top_drift,
            "all_feature_psi": all_feat,
        })
    return chunks


def _build_descriptive_chunks(cont_df: pd.DataFrame, cat_df: pd.DataFrame) -> list:
    chunks = []
    datasets = cont_df["Dataset"].unique() if not cont_df.empty else []
    for dataset in datasets:
        c_sub = cont_df[cont_df["Dataset"] == dataset]
        k_sub = cat_df[cat_df["Dataset"] == dataset] if not cat_df.empty else pd.DataFrame()
        n_records = int(c_sub["Total_Count"].iloc[0]) if len(c_sub) else 0
        null_pct_max = 0.0
        high_null_features = []
        if len(c_sub) and "Null_Count" in c_sub.columns:
            null_rates = (c_sub["Null_Count"] / (c_sub["Total_Count"] + 1) * 100).fillna(0)
            null_pct_max = float(null_rates.max())
            high_null = c_sub[null_rates > 1.0][["Variable_Name"]].copy()
            high_null["null_pct"] = null_rates[null_rates > 1.0].values
            high_null_features = high_null.to_dict("records")
        # Build mean / std summary for key stats
        cont_summary = {}
        if len(c_sub):
            for _, row in c_sub.iterrows():
                cont_summary[row["Variable_Name"]] = {
                    "mean": round(float(row.get("Mean", 0)), 4),
                    "std":  round(float(row.get("Std_Dev", 0)), 4),
                    "p50":  round(float(row.get("p50", 0)), 4),
                    "null_count": int(row.get("Null_Count", 0)),
                }
        chunks.append({
            "type":               "descriptive",
            "dataset":            dataset,
            "n_records":          n_records,
            "n_continuous":       len(c_sub),
            "n_categorical":      len(k_sub),
            "null_pct_max":       null_pct_max,
            "high_null_features": high_null_features,
            "cont_summary":       cont_summary,
        })
    return chunks


# ── Consensus chunk builder ────────────────────────────────────────────────────

def _build_consensus_chunks(shap_df: pd.DataFrame, lime_df: pd.DataFrame,
                             surr_df: pd.DataFrame, top_n: int = 10) -> list:
    """
    For each (model, dataset, group_var, group_value), compute rank agreement
    across SHAP, LIME, and Surrogate and classify each feature as:
      Strong Consensus  — appears in top-N of all available frameworks
      Partial Agreement — appears in top-N of exactly 2 frameworks
      Divergent         — appears in top-N of only 1 framework
    """
    chunks = []

    fw_map = [
        ("SHAP",      shap_df, "Mean_Abs_SHAP",    None),
        ("LIME",      lime_df, "Mean_Abs_LIME",     None),
        ("Surrogate", surr_df, "Feature_Importance","Selected"),
    ]

    # Pre-index each framework: {(model, dataset, gvar, gval): {feature: rank}}
    fw_rank_index: dict[str, dict] = {}
    available_fw = []
    for fw_name, df, imp_col, sel_col in fw_map:
        if df.empty or imp_col not in df.columns:
            continue
        sub = df[df[sel_col] == True].copy() if sel_col and sel_col in df.columns else df.copy()
        rank_lookup: dict[tuple, dict] = {}
        for (model, dataset, gvar, gval), grp in sub.groupby(
                ["Model", "Dataset", "Group_Variable", "Group_Value"]):
            top_feats = grp.nlargest(top_n, imp_col)["Feature"].tolist()
            rank_lookup[(model, dataset, gvar, gval)] = {
                feat: rank + 1 for rank, feat in enumerate(top_feats)
            }
        fw_rank_index[fw_name] = rank_lookup
        available_fw.append(fw_name)

    if len(available_fw) < 2:
        return chunks

    # Collect all unique group keys from the primary (SHAP) framework
    primary_keys = set(fw_rank_index[available_fw[0]].keys())

    for key in primary_keys:
        model, dataset, gvar, gval = key

        # Feature → {framework: rank} mapping (only for features in any framework's top-N)
        all_features: dict[str, dict] = {}
        for fw in available_fw:
            for feat, rank in fw_rank_index[fw].get(key, {}).items():
                all_features.setdefault(feat, {})[fw] = rank

        n_fw = len(available_fw)
        feature_rows = []
        for feat, fw_ranks in all_features.items():
            n_agree = len(fw_ranks)
            # Missing frameworks get rank = top_n + 1 for averaging
            avg_rank = (sum(fw_ranks.values()) + (n_fw - n_agree) * (top_n + 1)) / n_fw
            if n_agree == n_fw:
                status = "Strong Consensus"
            elif n_agree == n_fw - 1:
                status = "Partial Agreement"
            else:
                status = "Divergent"
            row = {
                "feature":            feat,
                "n_frameworks_agree": n_agree,
                "avg_rank":           round(avg_rank, 2),
                "consensus":          status,
            }
            for fw in available_fw:
                row[f"{fw}_rank"] = fw_ranks.get(fw)   # None = not in top-N
            feature_rows.append(row)

        feature_rows.sort(key=lambda r: r["avg_rank"])

        chunks.append({
            "type":               "consensus",
            "model":              model,
            "dataset":            dataset,
            "group_var":          gvar,
            "group_value":        gval,
            "frameworks":         available_fw,
            "top_n":              top_n,
            "features":           feature_rows,
            "n_strong_consensus": sum(1 for r in feature_rows if r["consensus"] == "Strong Consensus"),
            "n_partial":          sum(1 for r in feature_rows if r["consensus"] == "Partial Agreement"),
            "n_divergent":        sum(1 for r in feature_rows if r["consensus"] == "Divergent"),
        })

    return chunks


# ── Main KB builder ────────────────────────────────────────────────────────────

def build_knowledge_base(outputs_dir: str, manifest_df: pd.DataFrame,
                          top_n: int = 10, client=None) -> KnowledgeBase:
    """
    Load all V2 output CSVs and build an in-memory knowledge base.
    If client is None, embeddings are zeroed (filter-based retrieval still works).
    """
    import os

    def _load(path: str) -> pd.DataFrame:
        full = os.path.join(outputs_dir, path)
        return pd.read_csv(full) if os.path.exists(full) else pd.DataFrame()

    print("  [KB] Loading CSVs...", flush=True)
    perf_df      = _load("model_performance.csv")
    shap_df      = _load("shap_importance.csv")
    lime_df      = _load("lime_importance.csv")
    surr_df      = _load("surrogate_importance.csv")
    model_psi_df = _load("drift_analysis/model_stability/model_stability_psi.csv")
    feat_psi_df  = _load("drift_analysis/population_stability/feature_population_psi.csv")
    cont_df      = _load("drift_analysis/descriptive_statistics/continuous_features_stats.csv")
    cat_df       = _load("drift_analysis/descriptive_statistics/categorical_features_stats.csv")

    print("  [KB] Building chunks...", flush=True)
    chunks = (
        _build_manifest_chunks(manifest_df) +
        _build_performance_chunks(perf_df) +
        _build_importance_chunks(shap_df, lime_df, surr_df, top_n) +
        _build_importance_pivot_chunks(shap_df, lime_df, surr_df, top_n) +
        _build_consensus_chunks(shap_df, lime_df, surr_df, top_n) +
        _build_drift_chunks(model_psi_df, feat_psi_df) +
        _build_descriptive_chunks(cont_df, cat_df)
    )
    texts = [_chunk_to_text(c) for c in chunks]
    print(f"  [KB] {len(chunks)} chunks built. Embedding...", flush=True)

    if client is not None:
        embs = _embed_texts(texts, client)
    else:
        embs = np.zeros((len(chunks), 768), dtype=np.float32)

    print(f"  [KB] Knowledge base ready.", flush=True)
    return KnowledgeBase(chunks=chunks, texts=texts, embs=embs)


# ── KB cache persistence ───────────────────────────────────────────────────────

_SOURCE_PATHS = [
    "model_performance.csv",
    "shap_importance.csv",
    "lime_importance.csv",
    "surrogate_importance.csv",
    "drift_analysis/model_stability/model_stability_psi.csv",
    "drift_analysis/population_stability/feature_population_psi.csv",
    "drift_analysis/descriptive_statistics/continuous_features_stats.csv",
    "drift_analysis/descriptive_statistics/categorical_features_stats.csv",
]


def is_cache_valid(cache_path: str, outputs_dir: str, manifest_path: str) -> bool:
    """
    Return True if the cache file is newer than all source CSVs and the manifest.
    Returns False if the cache does not exist or any source is newer.
    """
    import os
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    sources = [os.path.join(outputs_dir, p) for p in _SOURCE_PATHS] + [manifest_path]
    for path in sources:
        if os.path.exists(path) and os.path.getmtime(path) > cache_mtime:
            return False
    return True


def save_knowledge_base(kb: KnowledgeBase, cache_path: str):
    """Pickle the KnowledgeBase to disk with version tag."""
    import os, pickle
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {"version": KB_VERSION, "kb": kb}
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(f"  [KB] Cache saved → {cache_path}  ({size_mb:.1f} MB, v{KB_VERSION})", flush=True)


def load_knowledge_base(cache_path: str) -> "KnowledgeBase | None":
    """
    Load a previously pickled KnowledgeBase from disk.
    Returns None if the cache version is outdated (triggers a rebuild).
    """
    import pickle
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    # Support old-format caches (bare KnowledgeBase, no version wrapper)
    if isinstance(payload, KnowledgeBase):
        print(f"  [KB] Cache version mismatch (old format) — will rebuild.", flush=True)
        return None
    if payload.get("version") != KB_VERSION:
        print(f"  [KB] Cache version {payload.get('version')} ≠ current {KB_VERSION} — will rebuild.", flush=True)
        return None
    kb = payload["kb"]
    print(f"  [KB] Cache loaded ← {cache_path}  ({len(kb.chunks)} chunks, v{KB_VERSION})", flush=True)
    return kb

# ── Prompt task builders ───────────────────────────────────────────────────────

def _fmt_context(chunks: list) -> str:
    safe = [{k: v for k, v in c.items()} for c in chunks]
    return json.dumps(safe, indent=2, default=str)


def _build_task_prompt(narrative_type: str, params: dict, context_json: str) -> str:
    if narrative_type == "overview":
        model   = params["model"]
        champ   = params.get("champion", "N")
        champ_str = (
            "\n\nIMPORTANT: This is the CHAMPION MODEL — the organisation's recommended "
            "production model. Highlight this clearly in the Recommended Use section."
            if champ == "Y" else ""
        )
        return (
            f"You are analysing the Sales Forecasting model: **{model}**.\n"
            f"DEV dataset is the primary benchmark for all performance claims.{champ_str}\n\n"
            f"CONTEXT DATA (JSON):\n{context_json}\n\n"
            f"Write a Model Overview with exactly these sections:\n"
            f"## What This Model Does\n"
            f"## How It Works\n"
            f"## Key Strengths\n(bullet points, at least 4)\n"
            f"## Business Value\n"
            f"## Recommended Use\n\n"
            f"Base all performance claims exclusively on DEV metrics from the context. "
            f"Be specific — include actual numbers."
        )

    if narrative_type == "explanation":
        model  = params["model"]
        level  = params["level"]
        champ  = params.get("champion", "N")
        champ_note = " [CHAMPION MODEL]" if champ == "Y" else ""
        if level == "Global":
            level_instr = (
                "Analyse global-level importance across DEV (benchmark), OOS, and OOT_01 datasets. "
                "DEV is the baseline. Compute and comment on the importance shift (%) from DEV to OOS "
                "and DEV to OOT_01 for the top features."
            )
        elif level == "Decile":
            level_instr = (
                "Analyse feature importance pivoted across demand deciles (D1=lowest demand, D10=highest). "
                "Identify which features drive low-demand vs high-demand periods. "
                "Note any counterintuitive shifts as demand volume scales up."
            )
        else:
            level_instr = (
                "Analyse feature importance across customer segments (e.g. Gold, Silver, Bronze, Platinum, White). "
                "Identify segment-specific drivers. Note which segments are most/least sensitive to "
                "the top features and what this implies commercially."
            )
        return (
            f"You are analysing {level}-level feature explanations for model: **{model}**{champ_note}.\n"
            f"CONTEXT DATA (JSON):\n{context_json}\n\n"
            f"{level_instr}\n\n"
            f"Structure your response with these sections:\n"
            f"## Top Drivers\n"
            f"## Stability vs DEV Benchmark\n"
            f"## Notable Shifts\n"
            f"## Interpretation & Recommendations\n\n"
            f"Reference specific feature names and importance values from the context. "
            f"Always compare against DEV as the benchmark."
        )

    if narrative_type == "drift":
        model   = params["model"]
        bm      = params.get("benchmark", "DEV")
        eval_ds = params["eval_dataset"]
        champ   = params.get("champion", "N")
        champ_note = " [CHAMPION MODEL]" if champ == "Y" else ""
        return (
            f"You are writing a Model Drift Report for **{model}**{champ_note}: "
            f"benchmark={bm} → evaluation={eval_ds}.\n"
            f"CONTEXT DATA (JSON):\n{context_json}\n\n"
            f"Write a drift report following this exact structure:\n\n"
            f"Start with a summary line: 'The drift analysis for {model} comparing {bm} (benchmark) "
            f"to {eval_ds} shows X anomalies across Y features...'\n\n"
            f"Then use these sections:\n"
            f"**CRITICAL ISSUE DETECTED** (if model_status = Significant Drift or n_drifted > 30%): "
            f"describe the specific drift issue and its implications.\n"
            f"**WARNING** (for Monitor-status features): list and explain.\n"
            f"**Stable Features**: summarise.\n"
            f"**Data Quality Trend**: overall assessment from benchmark to evaluation.\n"
            f"**Recommended Actions**: numbered list.\n\n"
            f"Always name {bm} as the explicit benchmark. Use actual PSI values."
        )

    if narrative_type == "lineage":
        dataset = params["dataset"]
        return (
            f"You are writing a Data Lineage Report for the **{dataset}** dataset.\n"
            f"CONTEXT DATA (JSON):\n{context_json}\n\n"
            f"Write a pipeline data quality narrative following this structure:\n\n"
            f"Start with: 'The data lineage pipeline for {dataset} shows X anomalies across Y stages...'\n\n"
            f"Then cover:\n"
            f"**CRITICAL ISSUE DETECTED** (if any): temporal leakage, schema mismatches, "
            f"severe null rates, or major distribution breaks.\n"
            f"**WARNING** (moderate issues): distribution drift, encoding anomalies, "
            f"moderate null rates.\n"
            f"**Data Quality Trend**: trace quality from raw data → feature engineering → model input. "
            f"Compare against DEV (benchmark) dataset where applicable.\n"
            f"**Recommended Actions**: numbered list of concrete engineering actions.\n\n"
            f"Reference specific feature names, null counts, and PSI values. "
            f"Be precise about which pipeline stages are affected."
        )

    if narrative_type == "comparison":
        dataset  = params["dataset"]
        champion = params.get("champion_model", "XGBoost")
        return (
            f"You are writing a Model Comparison Report for the **{dataset}** dataset.\n"
            f"The champion model (organisational standard) is: **{champion}**.\n"
            f"CONTEXT DATA (JSON):\n{context_json}\n\n"
            f"Write a comparison report following this exact structure:\n\n"
            f"Start with: 'Comparing the available models for Sales Forecasting on {dataset} "
            f"reveals distinct trade-offs...'\n\n"
            f"Then use these sections:\n"
            f"**Performance Leader**: which model has the best Accuracy/R²? Quote actual numbers.\n"
            f"**Explainability Champion**: which model offers clearest interpretability?\n"
            f"**Champion Model Assessment** ({champion}): how does the champion compare? "
            f"Does it lead, match, or trail on key metrics? DEV is the baseline reference.\n"
            f"**Recommendation**: explicit recommendation on which model to use for which purpose.\n\n"
            f"Reference actual metric values. Always use DEV as the baseline for all comparisons."
        )

    if narrative_type == "consensus":
        model  = params["model"]
        level  = params["level"]
        champ  = params.get("champion", "N")
        champ_note = " [CHAMPION MODEL]" if champ == "Y" else ""
        if level == "Global":
            level_instr = (
                "The context contains Global-level consensus across DEV (benchmark), OOS, and OOT_01. "
                "Analyse how agreement between frameworks changes across datasets — do SHAP, LIME, "
                "and Surrogate agree more on DEV than on OOT_01?"
            )
        elif level == "Decile":
            level_instr = (
                "The context contains consensus data pivoted across demand deciles (D1=lowest, D10=highest). "
                "Identify whether framework agreement is stronger for extreme deciles (D1, D10) vs "
                "mid-range deciles. Note if certain frameworks uniquely capture drivers at high-demand periods."
            )
        else:
            level_instr = (
                "The context contains consensus data across customer segments (Gold, Silver, Bronze, etc.). "
                "Identify whether framework agreement is consistent across segments or if certain segments "
                "reveal disagreements. Note segment-specific divergences."
            )
        return (
            f"You are analysing explanation method consensus for model: **{model}**{champ_note} "
            f"at {level} level.\n"
            f"Three explainability frameworks were applied: SHAP, LIME, and Surrogate Regression.\n"
            f"DEV is the benchmark dataset. All comparisons must reference it explicitly.\n\n"
            f"CONTEXT DATA (JSON):\n{context_json}\n\n"
            f"{level_instr}\n\n"
            f"Write an Explanation Consensus & Triangulation report with these exact sections:\n\n"
            f"## Consensus Features\n"
            f"List features where all 3 methods agree (Strong Consensus). "
            f"Explain why these are the most trustworthy drivers. Quote their average rank.\n\n"
            f"## Method Disagreements\n"
            f"Identify Divergent features — those only one method flags as important. "
            f"Explain the methodological reason for the disagreement: "
            f"SHAP measures marginal contribution from the full model; "
            f"LIME fits a local linear approximation around each prediction; "
            f"Surrogate fits a global interpretable model (panel regression). "
            f"What does each method's unique finding reveal?\n\n"
            f"## Framework-Specific Insights\n"
            f"What does each framework uniquely surface that the others miss? "
            f"Flag any feature that appears only in SHAP, only in LIME, or only in Surrogate.\n\n"
            f"## Confidence Assessment\n"
            f"Rate overall explanation confidence as High / Medium / Low based on the level of consensus. "
            f"State how many features are in Strong Consensus vs Divergent, and what this means for "
            f"trust in the model's explainability.\n\n"
            f"## Recommendations\n"
            f"Which features can the business rely on with high confidence? "
            f"Which require caution due to method disagreement? "
            f"Always reference DEV as the benchmark for any cross-dataset comparisons."
        )

    return f"CONTEXT DATA:\n{context_json}\n\nGenerate a narrative of type: {narrative_type}."


# ── Generation ─────────────────────────────────────────────────────────────────

_RETRY_DELAYS = [10, 30, 60]   # seconds between attempts 1→2, 2→3, 3→fail


def generate_narrative(narrative_type: str, context_chunks: list, params: dict,
                        client, persona: str, sleep_s: float = 5.0) -> str:
    """
    Generate one narrative section for a given persona using Gemini.
    Sleeps sleep_s seconds before the first call, then retries up to 3 times
    with exponential backoff on any API error. Raises RuntimeError on final failure
    so callers can detect and propagate the failure rather than silently writing
    an error string into the output file.
    """
    time.sleep(sleep_s)
    from google.genai import types as gtypes

    system_prompt = PERSONAS[persona]
    context_json  = _fmt_context(context_chunks)
    task_prompt   = _build_task_prompt(narrative_type, params, context_json)

    last_exc = None
    for attempt, backoff in enumerate([0] + _RETRY_DELAYS, start=1):
        if backoff:
            print(f"      [retry {attempt}/{len(_RETRY_DELAYS)+1}] sleeping {backoff}s...",
                  end="", flush=True)
            time.sleep(backoff)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=task_prompt,
                config=gtypes.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                ),
            )
            return response.text or ""
        except Exception as e:
            last_exc = e
            print(f"      [attempt {attempt} failed: {e}]", flush=True)

    raise RuntimeError(
        f"generate_narrative failed after {len(_RETRY_DELAYS)+1} attempts "
        f"({narrative_type}, {persona}): {last_exc}"
    )


# ── RAGAS evaluation ───────────────────────────────────────────────────────────

def evaluate_narratives(records: list, client, api_key: str) -> pd.DataFrame:
    """
    Evaluate generated narratives using RAGAS faithfulness + answer_relevancy.
    Each record: {narrative_type, model, dataset, level, persona, query, answer, contexts}
    Returns a DataFrame with scores. Returns empty DataFrame if ragas is not installed.
    """
    try:
        from datasets import Dataset as HFDataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError as e:
        print(f"  [RAGAS] Package not available ({e}). Skipping evaluation.", flush=True)
        return pd.DataFrame()

    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0,
    )
    evaluator_embs = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )

    hf_data = HFDataset.from_dict({
        "question":          [r["query"]   for r in records],
        "answer":            [r["answer"]  for r in records],
        "contexts":          [r["contexts"] for r in records],
    })

    try:
        result = evaluate(
            dataset=hf_data,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_embs,
        )
        scores_df = result.to_pandas()
        for col in ["narrative_type", "model", "dataset", "level", "persona"]:
            if col in records[0]:
                scores_df[col] = [r.get(col, "") for r in records]
        return scores_df
    except Exception as e:
        print(f"  [RAGAS] Evaluation failed: {e}", flush=True)
        return pd.DataFrame()
