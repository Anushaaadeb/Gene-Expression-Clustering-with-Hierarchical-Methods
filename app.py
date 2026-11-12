import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

st.set_page_config(page_title="DNA Clustering", page_icon="🧬")
st.title("🧬 DNA Sequence Clustering")
st.write("Upload a FASTA file, or a CSV with columns `gene` and `sequence`.")

file = st.file_uploader("Upload your file", type=["fasta", "fa", "csv"])
k = st.slider("Number of clusters", 2, 8, 3)
if file is None:
    st.info("Upload a file to see the results.")
    st.stop()

# 1. Read the file
text = file.getvalue().decode("utf-8-sig")
if file.name.lower().endswith(".csv"):
    data = pd.read_csv(io.StringIO(text))
    data.columns = [c.strip().lower() for c in data.columns]
    if "sequence" not in data.columns:
        st.error("The CSV needs a column called 'sequence'.")
        st.stop()
    seqs = data["sequence"].astype(str).tolist()
    names = data["gene"].astype(str).tolist() if "gene" in data.columns else [f"Seq{i + 1}" for i in range(len(seqs))]
else:
    records = list(SeqIO.parse(io.StringIO(text), "fasta"))
    names = [r.id for r in records]
    seqs = [str(r.seq) for r in records]

# 2. Keep only valid sequences (A, T, G, C)
pairs = [(n, re.sub(r"\s+", "", s.upper())) for n, s in zip(names, seqs)]
pairs = [(n, s) for n, s in pairs if s and set(s) <= set("ATGC")]
st.write(f"Read {len(seqs)} sequences, {len(pairs)} are valid.")
if len(pairs) < 3:
    st.warning("You need at least 3 valid sequences.")
    st.stop()
names = [n for n, s in pairs]
seqs = [s for n, s in pairs]

# 3. Base frequencies, then cluster
features = np.array([[s.count(b) / len(s) for b in "ATGC"] for s in seqs])
Z = linkage(pdist(features), method="average")
clusters = fcluster(Z, t=min(k, len(seqs)), criterion="maxclust")

# 4. Show results
results = pd.DataFrame({
    "Gene": names,
    "Length": [len(s) for s in seqs],
    "GC_content": [round((s.count("G") + s.count("C")) / len(s), 3) for s in seqs],
    "Cluster": clusters,
})
st.subheader("Results")
st.dataframe(results, use_container_width=True)

st.subheader("Dendrogram")
fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(Z, labels=names, leaf_rotation=90, ax=ax)
st.pyplot(fig)

st.subheader("Genes per cluster")
st.bar_chart(results["Cluster"].value_counts().sort_index())
st.download_button("Download results (CSV)", results.to_csv(index=False), "clusters.csv", "text/csv")
