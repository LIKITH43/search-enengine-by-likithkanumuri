import json
import streamlit as st
from search_engine import SimplifiedSearchEngine

st.set_page_config(
    page_title="Section 23.6.4 Search Engine",
    page_icon="🔍",
    layout="wide",
)


@st.cache_data
def load_default_pages() -> dict:
    with open("input_pages.json", "r", encoding="utf-8") as f:
        return json.load(f)


if "pages" not in st.session_state:
    st.session_state.pages = load_default_pages()

if "seed_url" not in st.session_state:
    st.session_state.seed_url = "index.html"

# Build / Rebuild engine from current corpus
engine = SimplifiedSearchEngine()
crawled_urls = engine.crawl_and_index(
    st.session_state.pages,
    seed_urls=[st.session_state.seed_url],
)

st.title("🔍 Simplified Web Search Engine (Section 23.6.4)")
st.caption(
    "Implements a Compressed Trie (Radix Tree) index, external Occurrence Lists Array, "
    "Sorted-Sequence Intersection, BeautifulSoup Web Crawler, and TF + Link Popularity Ranking."
)

# Sidebar: Corpus & Web Crawler Controls
with st.sidebar:
    st.header("🕸️ Web Crawler & Corpus")
    seed_choice = st.selectbox(
        "Crawler Seed Page",
        options=list(st.session_state.pages.keys()),
        index=0,
    )
    if seed_choice != st.session_state.seed_url:
        st.session_state.seed_url = seed_choice
        st.rerun()

    st.metric("Crawled Pages", f"{len(crawled_urls)} / {len(st.session_state.pages)}")
    st.metric("Compressed Trie Terms", engine.trie.term_count)
    st.metric("Occurrence Lists Array Length", len(engine.occurrence_lists))

    st.divider()
    st.subheader("➕ Add a New Web Page")
    with st.form("add_page_form", clear_on_submit=True):
        new_url = st.text_input("Page Filename / URL", placeholder="graph_algorithms.html")
        new_title = st.text_input("Page Title", placeholder="Graph Traversal Algorithms")
        new_body = st.text_area(
            "Page Content (HTML or plain text)",
            placeholder="<p>Breadth first search and depth first search explore graph vertices.</p>",
        )
        link_targets = st.multiselect(
            "Outgoing Hyperlinks to Existing Pages",
            options=list(st.session_state.pages.keys()),
        )
        submitted = st.form_submit_button("Add & Re-Crawl")
        if submitted and new_url.strip():
            links_html = " ".join(f'<a href="{u}">{u}</a>' for u in link_targets)
            full_html = (
                f"<html><head><title>{new_title or new_url}</title></head>"
                f"<body><h1>{new_title or new_url}</h1>{new_body}<nav>{links_html}</nav></body></html>"
            )
            st.session_state.pages[new_url.strip()] = full_html
            if "index.html" in st.session_state.pages and new_url.strip() != "index.html":
                st.session_state.pages["index.html"] += f' <a href="{new_url.strip()}">{new_title or new_url}</a>'
            st.rerun()

# Main Tabs
tab_search, tab_index, tab_boundary, tab_corpus = st.tabs([
    "🔎 Search Interface",
    "🌲 Compressed Trie & Inverted Index",
    "🧪 Boundary Condition Suite",
    "📄 Crawled HTML Pages",
])

with tab_search:
    query = st.text_input(
        "Enter search keywords (multi-word queries compute sorted occurrence list intersection):",
        value="search engine trie",
        placeholder="e.g., compressed trie, web crawler, algorithm...",
    )

    response = engine.search_query(query)

    if response["status"] == "success":
        st.success(f"{response['message']} | Active Index Terms: `{response['terms']}`")
        for rank, item in enumerate(response["results"], start=1):
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### #{rank} — {item['title']}")
                    st.caption(f"URL: `{item['url']}` (Doc ID: `{item['doc_id']}`)")
                    st.write(item["snippet"])
                with col2:
                    st.metric("Relevance Score", item["score"])
                    st.caption(f"Term Freq: **{item['term_frequency']}** | In-Links: **{item['in_links']}**")
    elif response["status"] in ("empty_query", "only_stop_words"):
        st.warning(response["message"])
    else:
        st.error(response["message"])

with tab_index:
    st.subheader("Compressed Trie -> External Occurrence Lists Array")
    st.markdown(
        "Each external/terminal node in the **Compressed Trie** stores an integer index (`occ_idx`) "
        "into the separate **Occurrence Lists Array**, where every list is sorted ascending by `doc_id`."
    )

    filter_prefix = st.text_input("Filter index terms by prefix:", "").lower().strip()
    rows = []
    for term, occ_idx in engine.trie.collect_terms():
        if filter_prefix and not term.startswith(filter_prefix):
            continue
        occ_list = engine.occurrence_lists[occ_idx]
        formatted_occ = ", ".join(
            f"(doc_id={d} [{engine.documents[d]['url']}], tf={f})" for d, f in occ_list
        )
        rows.append({
            "Index Term (w)": term,
            "Trie Pointer (occ_idx)": occ_idx,
            "Sorted Occurrence List L (doc_id, tf)": formatted_occ,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

with tab_boundary:
    st.subheader("Automated Boundary Condition & Edge-Case Verification")
    st.markdown("Run built-in boundary tests required for project verification:")

    boundary_cases = [
        ("Single Keyword Match", "crawler"),
        ("Multi-Keyword Intersection", "compressed trie"),
        ("Case Insensitivity & Stop Words", "THE Trie AND the Index"),
        ("Boundary: Empty Query", ""),
        ("Boundary: Whitespace Only", "    "),
        ("Boundary: Only Stop Words", "the a an of in to for with on at"),
        ("Boundary: Punctuation & Digits Only", "2026 !@#$%^&*()"),
        ("Boundary: Out-of-Vocabulary Term", "supercalifragilistic"),
        ("Boundary: Disjoint Terms (Empty Intersection)", "crawler binary"),
    ]

    test_rows = []
    for desc, q in boundary_cases:
        out = engine.search_query(q)
        test_rows.append({
            "Test Case": desc,
            "Input Query": repr(q),
            "Parsed Index Terms": str(out["terms"]),
            "Status": out["status"],
            "Matches": len(out["results"]),
            "System Message": out["message"],
        })

    st.dataframe(test_rows, use_container_width=True, hide_index=True)

with tab_corpus:
    st.subheader("Crawled Site Pages & Hyperlink Graph")
    for doc in engine.documents:
        with st.expander(f"Doc ID {doc['doc_id']}: {doc['title']} ({doc['url']})"):
            st.write(f"**Outgoing Links:** {doc['out_links']}")
            st.write(f"**Incoming Link Count:** {engine.in_degrees.get(doc['doc_id'], 0)}")
            st.code(st.session_state.pages[doc["url"]], language="html")
