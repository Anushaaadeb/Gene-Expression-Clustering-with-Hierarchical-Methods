import streamlit as st

st.title("🧬 DNA Sequence Tool")
st.write("Paste a DNA sequence to count its bases.")

seq = st.text_area("DNA sequence", "ATGCATGCAATTGGCC")
seq = seq.strip().upper()

if st.button("Analyse"):
    if seq and set(seq) <= set("ATGC"):
        counts = {base: seq.count(base) for base in "ATGC"}
        st.bar_chart(counts)
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        st.write(f"Length: {len(seq)}")
        st.write(f"GC content: {gc:.1%}")
    else:
        st.error("Use only the letters A, T, G and C.")
