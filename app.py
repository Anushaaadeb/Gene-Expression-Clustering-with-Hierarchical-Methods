import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

# ---------- Page setup ----------
st.set_page_config(page_title="DNA Clustering", page_icon="🧬", layout="wide")
st.title("🧬 DNA Sequence Clustering")
st.caption("Upload DNA sequences, pick a distance and linkage method, and explore the clusters.")


# ---------- Helper functions ----------
def example_data():
    """12 made-up genes in 3 groups with different GC content."""
    rng = np.random.default_rng(1)
    groups = [[0.35, 0.35, 0.15, 0.15], [0.25, 0.25, 0.25, 0.25], [0.15, 0.15, 0.35, 0.35]]
    names, seqs = [], []
    for g, probs in enumerate(groups):
        for i in range(4):
            length = int(rng.integers(80, 150))
            names.append(f"Gene_{g + 1}{chr(65 + i)}")
            seqs.append("".join(rng.choice(list("ATGC"), size=length, p=probs)))
    return names, seqs


def read_file(uploaded):
    """Read a FASTA, CSV or TSV file into (names, sequences)."""
    text = uploaded.getvalue().decode("utf-8-sig")
    filename = uploaded.name.lower()
    if filename.endswith((".csv", ".tsv")):
        sep = "," if filename.endswith(".csv") else "\t"
        df = pd.read_csv(io.StringIO(text), sep=sep)
        cols = {c.strip().lower(): c for c in df.columns}
        seq_col = next((cols[c] for c in ["sequence", "dna_sequence", "seq"] if c in cols), None)
        name_col = next((cols[c] for c in ["gene", "gene_id", "name", "id"] if c in cols), None)
        if seq_col is None:
            st.error("Could not find a sequence column (use: sequence, dna_sequence or seq).")
            st.stop()
        seqs = df[seq_col].astype(str).tolist()
        names = df[name_col].astype(str).tolist() if name_col else [f"Sequence_{i + 1}" for i in range(len(df))]
    else:
        records = list(SeqIO.parse(io.StringIO(text), "fasta"))
        names = [r.id for r in records]
        seqs = [str(r.seq) for r in records]
    return names, seqs


def clean(names, seqs):
    """Keep only sequences made of A, T, G, C."""
    good_names, good_seqs = [], []
    for n, s in zip(names, seqs):
        s = re.sub(r"\s+", "", s.upper())
        if s and set(s) <= set("ATGC"):
            good_names.append(n)
            good_seqs.append(s)
    return good_names, good_seqs


def base_frequencies(seq):
    return [seq.count(b) / len(seq) for b in "ATGC"]


# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")
uploaded = st.sidebar.file_uploader("Upload sequences", type=["fasta", "fa", "fna", "csv", "tsv"])
use_example = st.sidebar.checkbox("Use example data", value=uploaded is None)
metric_label = st.sidebar.selectbox("Distance metric", ["Euclidean", "Manhattan"])
metric = "euclidean" if metric_label == "Euclidean" else "cityblock"
methods = ["single", "complete", "average"] + (["ward"] if metric == "euclidean" else [])
method = st.sidebar.selectbox("Linkage method", methods, index=2)
k = st.sidebar.slider("Number of clusters", 2, 8, 3)
max_seqs = st.sidebar.slider("Max sequences to analyse", 10, 500, 200)

# ---------- Load and clean ----------
if uploaded is not None and not use_example:
    names, seqs =
