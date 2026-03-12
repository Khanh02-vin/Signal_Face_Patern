from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network

st.set_page_config(layout="wide")
st.title("Attraction Pattern Dashboard")

OUT = Path("output")
PAIRS_PATH = OUT / "person_to_person_top_pairs.csv"
SEGMENTS_PATH = OUT / "ensemble_taste_segments.csv"
RANKING_PATH = OUT / "ensemble_taste_ranking.csv"
SIGNAL_PATH = OUT / "attraction_signal_summary.json"
BENCHMARK_PATH = OUT / "preference_benchmark_report.json"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Cannot read CSV {path}: {exc}")
        st.stop()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _find_bridge_persons(graph: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["person", "bridge_score", "degree", "betweenness", "closeness"])

    deg = dict(graph.degree())
    bet = nx.betweenness_centrality(graph, normalized=True)
    clo = nx.closeness_centrality(graph)

    rows = []
    for node in graph.nodes():
        bridge_score = (0.55 * bet.get(node, 0.0)) + (0.30 * clo.get(node, 0.0)) + (0.15 * (deg.get(node, 0) / max(1, graph.number_of_nodes() - 1)))
        rows.append(
            {
                "person": node,
                "bridge_score": float(bridge_score),
                "degree": int(deg.get(node, 0)),
                "betweenness": float(bet.get(node, 0.0)),
                "closeness": float(clo.get(node, 0.0)),
            }
        )

    bridge_df = pd.DataFrame(rows).sort_values("bridge_score", ascending=False).head(top_n).reset_index(drop=True)
    return bridge_df


pairs_df = _load_csv(PAIRS_PATH)
ranking_df = _load_csv(RANKING_PATH) if RANKING_PATH.exists() else pd.DataFrame()
segments_df = _load_csv(SEGMENTS_PATH) if SEGMENTS_PATH.exists() else pd.DataFrame()
signal_json = _load_json(SIGNAL_PATH)
benchmark_json = _load_json(BENCHMARK_PATH)

best_method = benchmark_json.get("best_method", "weighted_mean")

required_cols = {"person_a", "person_b", "similarity"}
if not required_cols.issubset(pairs_df.columns):
    st.error("CSV must contain columns: person_a, person_b, similarity")
    st.stop()

st.sidebar.header("Filters")
threshold = st.sidebar.slider("Minimum similarity", 0.0, 1.0, 0.5, 0.05)

segment_options = ["All"]
tag_options = ["All"]
if not segments_df.empty:
    if "segment" in segments_df.columns:
        segment_options += sorted(segments_df["segment"].dropna().unique().tolist())
    if "tag" in segments_df.columns:
        tag_options += sorted(segments_df["tag"].dropna().unique().tolist())


allowed_people = None
if not segments_df.empty and "person" in segments_df.columns:
    filtered_seg = segments_df.copy()
    if selected_segment != "All" and "segment" in filtered_seg.columns:
        filtered_seg = filtered_seg[filtered_seg["segment"] == selected_segment]
    if selected_tag != "All" and "tag" in filtered_seg.columns:
        filtered_seg = filtered_seg[filtered_seg["tag"] == selected_tag]
    allowed_people = set(filtered_seg["person"].tolist())

G = nx.Graph()
for _, row in pairs_df.iterrows():
    try:
        person_a = str(row["person_a"])
        person_b = str(row["person_b"])
        sim = float(row["similarity"])
    except Exception:
        continue

    if sim < threshold:
        continue

    if allowed_people is not None and (person_a not in allowed_people or person_b not in allowed_people):
        continue

    G.add_edge(person_a, person_b, weight=sim)

if G.number_of_nodes() == 0:
    st.warning("No nodes found after current filters.")
    st.stop()

people = sorted(G.nodes())
selected_person = st.sidebar.selectbox("Focus person", people)

st.info(f"Benchmark best method: {best_method}")

manual_methods = ["auto", "weighted_mean", "mean_all", "core_percentile_40", "largest_cluster_center"]
method_mode = st.sidebar.selectbox("Preference method mode", manual_methods, index=0)
active_method = best_method if method_mode == "auto" else method_mode
st.caption(f"Active preference method: {active_method}")

tab_network, tab_bridge, tab_signals = st.tabs(["Network", "Bridge Persons", "Top Signals"])

with tab_network:
    st.subheader("Similarity Network")

    if G.degree(selected_person) > 0:
        ego = nx.ego_graph(G, selected_person)
    else:
        ego = nx.Graph()
        ego.add_node(selected_person)

    col1, col2 = st.columns([3, 2])

    with col1:
        net = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="black")

        for node in ego.nodes():
            degree = ego.degree(node)
            size = 12 + degree * 5
            color = "#ffe599" if node == selected_person else "#4FC3F7"
            net.add_node(node, label=node, size=size, color=color)

        for u, v, d in ego.edges(data=True):
            w = float(d.get("weight", 0.0))
            net.add_edge(u, v, value=w, width=max(1.0, w * 5.0))

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        net.write_html(tmp_file.name)
        with open(tmp_file.name, "r", encoding="utf-8") as file:
            st.components.v1.html(file.read(), height=720)

    with col2:
        st.subheader("Node Stats")
        neighbors = list(G.neighbors(selected_person))
        st.metric("Degree", G.degree(selected_person))
        st.metric("Connections shown", max(0, len(ego.nodes()) - 1))

        if neighbors:
            top_edges = sorted(
                [(n, float(G[selected_person][n]["weight"])) for n in neighbors],
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            st.markdown("**Top similar connections**")
            for node, score in top_edges:
                st.write(f"- {node}: {score:.3f}")
        else:
            st.write("No neighbors for selected person.")

with tab_bridge:
    st.subheader("Bridge Persons (network connectors)")
    bridge_df = _find_bridge_persons(G, top_n=25)
    st.dataframe(bridge_df, use_container_width=True)

with tab_signals:
    st.subheader("Attraction Signals")
    corr_df = pd.DataFrame(signal_json.get("top_correlation_features", []))
    imp_df = pd.DataFrame(signal_json.get("top_importance_features", []))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top correlation features**")
        st.dataframe(corr_df, use_container_width=True)
    with c2:
        st.markdown("**Top RF importance features**")
        st.dataframe(imp_df, use_container_width=True)

    if not ranking_df.empty and "person" in ranking_df.columns:
        st.markdown("**Selected person ranking details**")
        selected_row = ranking_df[ranking_df["person"] == selected_person]
        if not selected_row.empty:
            cols = ["person"]
            if active_method in selected_row.columns:
                cols.append(active_method)
            fallback_cols = [c for c in ["ensemble_score", "core_percentile_40", "largest_cluster_center", "weighted_mean"] if c in selected_row.columns]
            for c in fallback_cols:
                if c not in cols:
                    cols.append(c)
            st.dataframe(selected_row[cols], use_container_width=True)
