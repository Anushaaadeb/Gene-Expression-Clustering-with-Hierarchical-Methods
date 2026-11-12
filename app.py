import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

st.set_page_config(page_title="DNA Clustering", page_icon="🧬", layout="wide")
st.title("🧬 DNA Sequence Clustering")
st.write("Upload FASTA, GenBank, FASTQ, EMBL, CSV or TSV. CSV/TSV need a `sequence` column (and optionally `gene`).")

FORMATS = {".fasta": "fasta", ".fa": "fasta", ".fna": "fasta", ".ffn": "fasta", ".fas": "fasta",
           ".gb": "genbank", ".gbk": "genbank", ".gbff": "genbank", ".genbank": "genbank",
           ".fastq": "fastq", ".fq": "fastq", ".embl": "embl"}
EXT = [e.strip(".") for e in FORMATS] + ["csv", "tsv"]

file = st.file_uploader("Upload your file", type=EXT)
n_clusters = st.slider("Number of clusters", 2, 8, 3)
if file is None:
    st.info("Upload a file to see the results.")
    st.stop()

# ---- 1. Read the file ----
text = file.getvalue().decode("utf-8-sig")
ext = "." + file.name.lower().rsplit(".", 1)[-1]
if ext in (".csv", ".tsv"):
    data = pd.read_csv(io.StringIO(text), sep="," if ext == ".csv" else "\t")
    data.columns = [c.strip().lower() for c in data.columns]
    if "sequence" not in data.columns:
        st.error("The file needs a column called 'sequence'.")
        st.stop()
    seqs = data["sequence"].astype(str).tolist()
    names = data["gene"].astype(str).tolist() if "gene" in data.columns else [f"Sequence_{i + 1}" for i in range(len(seqs))]
else:
    records = list(SeqIO.parse(io.StringIO(text), FORMATS[ext]))
    names = [r.id for r in records]
    seqs = [str(r.seq) for r in records]

# ---- 2. Clean and validate (A, T, G, C only) ----
total = len(seqs)
pairs = [(n, re.sub(r"\s+", "", s.upper())) for n, s in zip(names, seqs)]
pairs = [(n, s) for n, s in pairs if s and set(s) <= set("ATGC")]
if len(pairs) < 3:
    st.warning("You need at least 3 valid DNA sequences.")
    st.stop()
if len(pairs) > 500:
    pick = np.random.default_rng(42).choice(len(pairs), 500, replace=False)
    pairs = [pairs[i] for i in pick]
    st.info("More than 500 sequences: a random sample of 500 is analysed.")
names = [n for n, s in pairs]
seqs = [s for n, s in pairs]

# ---- 3. Features, distances, all linkage methods ----
features = np.array([[s.count(b) / len(s) for b in "ATGC"] for s in seqs])
results, scores = {}, {}
for metric in ["euclidean", "cityblock"]:
    d = pdist(features, metric=metric)
    for method in ["single", "complete", "average", "ward"]:
        if method == "ward" and metric != "euclidean":
            continue
        Z = linkage(d, method=method)
        results[(metric, method)] = Z
        scores[(metric, method)] = cophenet(Z, d)[0]
valid = {k: v for k, v in scores.items() if np.isfinite(v)}
if not valid:
    st.error("No valid cophenetic scores were obtained.")
    st.stop()
best = max(valid, key=valid.get)
Zbest = results[best]
clusters = fcluster(Zbest, t=min(n_clusters, len(seqs)), criterion="maxclust")
df = pd.DataFrame({"Gene": names, "Length": [len(s) for s in seqs],
                   "GC_content": [(s.count("G") + s.count("C")) / len(s) for s in seqs],
                   "Cluster": clusters})

# ---- 4. Show everything ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("Sequences read", total)
m2.metric("Valid used", len(seqs))
m3.metric("Best combination", f"{best[0].replace('cityblock', 'manhattan')} + {best[1]}")
m4.metric("Cophenetic score", round(valid[best], 4))

tabs = st.tabs(["Sequences", "Dendrograms", "Cophenetic comparison", "Final clusters", "Summary"])

with tabs[0]:
    st.subheader("Base frequencies (A, T, G, C)")
    st.dataframe(pd.DataFrame(features, index=names, columns=list("ATGC")).round(3), use_container_width=True)

with tabs[1]:
    st.subheader("Hierarchical clustering: distance and linkage comparison")
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()
    for ax, ((metric, method), Z) in zip(axes, results.items()):
        if len(names) <= 40:
            dendrogram(Z, labels=names, ax=ax, leaf_rotation=90)
        else:
            dendrogram(Z, no_labels=True, ax=ax)
        score_text = f"{scores[(metric, method)]:.3f}" if np.isfinite(scores[(metric, method)]) else "Undefined"
        ax.set_title(f"{method.capitalize()} - {metric.capitalize()}\nCophenetic correlation = {score_text}")
        ax.set_ylabel("Distance")
    for ax in axes[len(results):]:
        ax.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Distance and linkage comparison")
    comp = pd.DataFrame([{"Distance": m, "Linkage": l, "Cophenetic Score": round(s, 4)}
                         for (m, l), s in valid.items()]).sort_values("Cophenetic Score", ascending=False)
    st.success(f"Best: {best[0]} + {best[1]} (score {valid[best]:.4f})")
    st.dataframe(comp, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f"{m.capitalize()}\n{l.capitalize()}" for (m, l) in valid], list(valid.values()))
    ax.set_ylabel("Cophenetic Correlation")
    ax.set_title("Cophenetic Correlation Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

with tabs[3]:
    st.subheader("Final clusters")
    st.dataframe(df.round(3), use_container_width=True, hide_index=True)
    st.subheader("Cluster summary")
    summary = df.groupby("Cluster").agg(Mean_Length=("Length", "mean"), Mean_GC=("GC_content", "mean"),
                                        Number_of_Genes=("Gene", "count")).round(3)
    st.dataframe(summary, use_container_width=True)
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in sorted(df["Cluster"].unique()):
            g = df[df["Cluster"] == c]
            ax.scatter(g["Length"], g["GC_content"], label=f"Cluster {c}", s=70)
        ax.set_xlabel("Sequence Length (bp)")
        ax.set_ylabel("GC Content")
        ax.set_title(f"Final Gene Clusters ({best[1].capitalize()} - {best[0].capitalize()})")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with right:
        counts = df["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Genes")
        ax.set_title("Number of Genes in Each Cluster")
        st.pyplot(fig)
    st.download_button("Download clusters (CSV)", df.to_csv(index=False), "clusters.csv", "text/csv")

with tabs[4]:
    st.subheader("Final summary")
    st.write(f"**Input file:** {file.name}")
    st.write(f"**Total valid sequences:** {len(seqs)} (of {total} read)")
    st.write("**Distance metrics compared:** Euclidean and Manhattan (cityblock)")
    st.write("**Linkage methods compared:** Single, Complete, Average, Ward")
    st.write(f"**Best distance metric:** {best[0]}")
    st.write(f"**Best linkage method:** {best[1]}")
    st.write(f"**Best cophenetic score:** {round(valid[best], 4)}")
    st.write(f"**Number of clusters:** {len(np.unique(clusters))}")
