import re
from collections import deque
from bs4 import BeautifulSoup
import streamlit as st

# ==============================================================================
# 1. STOP WORDS LIST (Articles, Prepositions, Pronouns, Conjunctions)
# ==============================================================================
STOP_WORDS = {
    # Articles
    "a", "an", "the",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those",
    # Prepositions & Common Conjunctions / Auxiliaries
    "in", "on", "at", "to", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "of", "off", "over", "under", "again", "further", "then",
    "once", "by", "as", "until", "while", "and", "but", "if", "or", "because",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "not", "no", "nor",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "just"
}

# ==============================================================================
# 2. SAMPLE MINI-WEBSITE (7 Interlinked HTML Pages for Crawling & Indexing)
# ==============================================================================
DEFAULT_SITE_PAGES = {
    "index.html": """
    <html>
      <head><title>CS Algorithms Portal - Home</title></head>
      <body>
        <h1>Welcome to the Data Structures and Algorithms Portal</h1>
        <p>This website covers fundamental data structures, tries, inverted files,
           graph traversal, and web search engines.</p>
        <nav>
          <a href="tries.html">Compressed Tries</a> |
          <a href="inverted_index.html">Inverted Index</a> |
          <a href="crawler.html">Web Crawler</a> |
          <a href="sorting.html">Sorted Sequence Merging</a> |
          <a href="ranking.html">Page Ranking</a>
        </nav>
      </body>
    </html>
    """,
    "tries.html": """
    <html>
      <head><title>Compressed Tries (Radix Trees)</title></head>
      <body>
        <h1>Compressed Tries</h1>
        <p>A compressed trie is a space-efficient tree data structure used for
           storing a set of strings or index terms. Unlike a standard trie, every
           internal node in a compressed trie has at least two children by merging
           chains of single-child nodes into compressed edge labels.</p>
        <p>Search engines use a compressed trie in main memory to map each index
           term to an external occurrence list index.</p>
        <a href="index.html">Back to Home</a>
        <a href="inverted_index.html">Learn about Inverted Files</a>
      </body>
    </html>
    """,
    "inverted_index.html": """
    <html>
      <head><title>Inverted Index & Occurrence Lists</title></head>
      <body>
        <h1>Inverted Files and Occurrence Lists</h1>
        <p>An inverted index stores key-value pairs where each key is an index
           term and the value is an occurrence list of document references.</p>
        <p>Storing occurrence lists in an external array outside the compressed
           trie keeps the trie compact enough to fit inside internal memory.</p>
        <a href="index.html">Home</a>
        <a href="tries.html">Compressed Tries</a>
        <a href="sorting.html">Intersection via Merging</a>
      </body>
    </html>
    """,
    "crawler.html": """
    <html>
      <head><title>Web Crawler & HTML Parsing</title></head>
      <body>
        <h1>Web Crawler Architecture</h1>
        <p>A web crawler traverses hyperlinks across web pages using breadth-first
           search. It parses HTML documents, extracts clean text tokens, removes
           stop words, and feeds index terms into the inverted index.</p>
        <a href="index.html">Home</a>
        <a href="inverted_index.html">Inverted Index</a>
        <a href="ranking.html">Relevance Ranking</a>
      </body>
    </html>
    """,
    "sorting.html": """
    <html>
      <head><title>Sorted Sequence Merging & Intersection</title></head>
      <body>
        <h1>Multi-Keyword Query Intersection</h1>
        <p>When a user searches multiple keywords, the search engine retrieves
           each keyword occurrence list from the compressed trie and computes
           their intersection using sorted sequence merging in linear time.</p>
        <a href="index.html">Home</a>
        <a href="inverted_index.html">Inverted Index</a>
      </body>
    </html>
    """,
    "ranking.html": """
    <html>
      <head><title>Search Result Relevance Ranking</title></head>
      <body>
        <h1>Ranking Search Results</h1>
        <p>Search engines rank matching pages by relevance. Our ranking algorithm
           combines term frequency across query keywords with incoming hyperlink
           popularity discovered by the web crawler.</p>
        <a href="index.html">Home</a>
        <a href="crawler.html">Web Crawler</a>
        <a href="tries.html">Compressed Tries</a>
      </body>
    </html>
    """,
    "orphan.html": """
    <html>
      <head><title>Unlinked Archive Page</title></head>
      <body>
        <h1>Orphan Document</h1>
        <p>This page has no incoming hyperlinks from the home page, testing how
           the web crawler handles unreachable pages unless seeded directly.</p>
      </body>
    </html>
    """
}


# ==============================================================================
# 3. TEXTBOOK DATA STRUCTURES (SECTION 23.6 & SECTION 8.2)
# ==============================================================================

class OccurrenceItem:
    """
    Represents a single entry in an occurrence list:
    Stores document ID (sorted key) and term frequency within that document.
    """
    def __init__(self, doc_id: int, freq: int = 1):
        self.doc_id = doc_id
        self.freq = freq

    def __repr__(self):
        return f"(doc_id={self.doc_id}, freq={self.freq})"


class CompressedTrieNode:
    """
    Node in a Compressed Trie (Radix Tree).
    - children: dict mapping first character -> (edge_label: str, child_node: CompressedTrieNode)
    - occurrence_index: int >= 0 if this node marks the end of an index term
                        (points to index in the external occurrence_lists array),
                        or None if purely an internal branching node.
    """
    def __init__(self, occurrence_index=None):
        self.children = {}
        self.occurrence_index = occurrence_index


class CompressedTrie:
    """
    Compressed Trie storing index terms in main memory.
    Each external/terminal node stores ONLY the integer index into the
    external array of occurrence lists (per Section 23.6.4).
    """
    def __init__(self):
        self.root = CompressedTrieNode()
        self.term_count = 0

    @staticmethod
    def _common_prefix_len(s1: str, s2: str) -> int:
        limit = min(len(s1), len(s2))
        i = 0
        while i < limit and s1[i] == s2[i]:
            i += 1
        return i

    def insert(self, word: str, occurrence_index: int) -> None:
        """Inserts a word and maps it to its external occurrence list index."""
        node = self.root
        remaining = word

        while remaining:
            first_char = remaining[0]
            if first_char not in node.children:
                # Case 1: No outgoing edge starts with first_char -> create leaf
                node.children[first_char] = (remaining, CompressedTrieNode(occurrence_index))
                self.term_count += 1
                return

            edge_label, child = node.children[first_char]
            prefix_len = self._common_prefix_len(remaining, edge_label)

            if prefix_len == len(edge_label):
                # Case 2: Full edge matched -> descend into child
                remaining = remaining[prefix_len:]
                node = child
            else:
                # Case 3: Partial edge match -> split existing edge
                common_prefix = edge_label[:prefix_len]
                edge_suffix = edge_label[prefix_len:]
                word_suffix = remaining[prefix_len:]

                split_node = CompressedTrieNode()
                split_node.children[edge_suffix[0]] = (edge_suffix, child)
                node.children[first_char] = (common_prefix, split_node)

                if word_suffix:
                    split_node.children[word_suffix[0]] = (
                        word_suffix,
                        CompressedTrieNode(occurrence_index),
                    )
                else:
                    split_node.occurrence_index = occurrence_index

                self.term_count += 1
                return

        # Exact match on an existing internal node
        if node.occurrence_index is None:
            self.term_count += 1
        node.occurrence_index = occurrence_index

    def search(self, word: str):
        """
        Searches for an exact index term in the Compressed Trie.
        Returns the integer index into the occurrence_lists array, or None if not found.
        """
        node = self.root
        remaining = word

        while remaining:
            first_char = remaining[0]
            if first_char not in node.children:
                return None

            edge_label, child = node.children[first_char]
            if not remaining.startswith(edge_label):
                return None

            remaining = remaining[len(edge_label):]
            node = child

        return node.occurrence_index

    def get_all_terms(self) -> list:
        """Traverses the compressed trie to list (term, occurrence_index) pairs."""
        results = []

        def _dfs(current_node: CompressedTrieNode, prefix: str):
            if current_node.occurrence_index is not None:
                results.append((prefix, current_node.occurrence_index))
            for ch in sorted(current_node.children.keys()):
                label, child = current_node.children[ch]
                _dfs(child, prefix + label)

        _dfs(self.root, "")
        return results

    def get_tree_visualization(self) -> str:
        """Builds an ASCII tree representation of the Compressed Trie."""
        lines = ["(ROOT)"]

        def _render(node: CompressedTrieNode, indent: str):
            items = sorted(node.children.items())
            for idx, (_, (label, child)) in enumerate(items):
                is_last = idx == len(items) - 1
                branch = "└── " if is_last else "├── "
                ext_tag = (
                    f"  --> [OccurrenceList #{child.occurrence_index}]"
                    if child.occurrence_index is not None
                    else ""
                )
                lines.append(f"{indent}{branch}'{label}'{ext_tag}")
                next_indent = indent + ("    " if is_last else "│   ")
                _render(child, next_indent)

        _render(self.root, "")
        return "\n".join(lines)


def sorted_sequence_intersect(list1: list, list2: list) -> list:
    """
    Sorted Sequence Merging Intersection (Textbook Section 8.2 / 23.6.4).
    Both list1 and list2 are sorted in ascending order of `doc_id`.
    Runs in O(|list1| + |list2|) time.
    Returns a list of OccurrenceItem with summed term frequencies for matching doc_ids.
    """
    i, j = 0, 0
    merged = []
    while i < len(list1) and j < len(list2):
        if list1[i].doc_id == list2[j].doc_id:
            merged.append(
                OccurrenceItem(
                    doc_id=list1[i].doc_id,
                    freq=list1[i].freq + list2[j].freq,
                )
            )
            i += 1
            j += 1
        elif list1[i].doc_id < list2[j].doc_id:
            i += 1
        else:
            j += 1
    return merged


# ==============================================================================
# 4. WEB CRAWLER & INVERTED INDEX SEARCH ENGINE
# ==============================================================================

class SimplifiedSearchEngine:
    """
    Implements the simplified Search Engine from Section 23.6:
    - Web Crawler using BFS & BeautifulSoup HTML parsing
    - Inverted Index = Compressed Trie + External Array of Occurrence Lists
    - Query Intersection via Sorted Sequence Merging
    - Relevance Ranking using Term Frequency + In-Link Popularity
    """
    def __init__(self):
        self.trie = CompressedTrie()
        self.occurrence_lists = []     # External array of sorted lists of OccurrenceItem
        self.documents = []            # List of document metadata dicts
        self.in_links = {}             # doc_id -> count of incoming links
        self.crawl_log = []            # Step-by-step crawler trace

    @staticmethod
    def tokenize_and_filter(text: str) -> list:
        """Lowercases text, extracts alphabetic words, and removes stop words."""
        raw_words = re.findall(r"[a-zA-Z]+", text.lower())
        return [w for w in raw_words if w not in STOP_WORDS and len(w) > 1]

    def crawl_and_index(self, pages_dict: dict, start_page: str = "index.html", include_unlinked: bool = False):
        """
        Crawls the mini-website starting from `start_page` using Breadth-First Search (BFS),
        parses HTML with BeautifulSoup, and builds the Compressed Trie + Occurrence Lists.
        """
        self.trie = CompressedTrie()
        self.occurrence_lists = []
        self.documents = []
        self.in_links = {}
        self.crawl_log = []

        if not pages_dict:
            return

        visited = set()
        queue = deque()

        if start_page in pages_dict:
            queue.append(start_page)
            visited.add(start_page)
        else:
            first_key = next(iter(pages_dict))
            queue.append(first_key)
            visited.add(first_key)

        crawled_order = []
        raw_links = []

        # Phase 1: BFS Web Crawl using BeautifulSoup
        while queue:
            page_name = queue.popleft()
            html_content = pages_dict[page_name]
            soup = BeautifulSoup(html_content, "html.parser")

            title = soup.title.string.strip() if soup.title and soup.title.string else page_name
            body_text = soup.get_text(separator=" ", strip=True)
            links = [a.get("href") for a in soup.find_all("a", href=True)]

            crawled_order.append((page_name, title, body_text, links))
            self.crawl_log.append(f"Crawled '{page_name}' ('{title}') -> found links: {links}")

            for link in links:
                raw_links.append((page_name, link))
                if link in pages_dict and link not in visited:
                    visited.add(link)
                    queue.append(link)

        # Optionally include orphan/unlinked pages if toggled by user
        if include_unlinked:
            for page_name, html_content in pages_dict.items():
                if page_name not in visited:
                    soup = BeautifulSoup(html_content, "html.parser")
                    title = soup.title.string.strip() if soup.title and soup.title.string else page_name
                    body_text = soup.get_text(separator=" ", strip=True)
                    links = [a.get("href") for a in soup.find_all("a", href=True)]
                    crawled_order.append((page_name, title, body_text, links))
                    self.crawl_log.append(f"Seeded unlinked page '{page_name}' ('{title}')")

        # Phase 2: Assign ascending Document IDs (0, 1, 2, ...) so occurrence lists stay sorted
        name_to_id = {}
        for doc_id, (page_name, title, body_text, links) in enumerate(crawled_order):
            name_to_id[page_name] = doc_id
            self.in_links[doc_id] = 0
            self.documents.append({
                "doc_id": doc_id,
                "filename": page_name,
                "title": title,
                "text": body_text,
                "links": links,
            })

        # Compute incoming link counts for ranking
        for src_name, dst_name in raw_links:
            if dst_name in name_to_id:
                self.in_links[name_to_id[dst_name]] += 1

        # Phase 3: Build Inverted Index (Compressed Trie + External Occurrence Lists)
        for doc in self.documents:
            doc_id = doc["doc_id"]
            tokens = self.tokenize_and_filter(doc["text"])

            # Count term frequencies within this document
            term_freqs = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1

            # Insert each unique term of this document into the Compressed Trie
            # Because doc_id increases monotonically (0, 1, 2...), appending guarantees
            # each occurrence list is naturally sorted by doc_id!
            for term, freq in term_freqs.items():
                occ_idx = self.trie.search(term)
                if occ_idx is None:
                    new_idx = len(self.occurrence_lists)
                    self.occurrence_lists.append([OccurrenceItem(doc_id, freq)])
                    self.trie.insert(term, new_idx)
                else:
                    self.occurrence_lists[occ_idx].append(OccurrenceItem(doc_id, freq))

    def search(self, query: str) -> dict:
        """
        Executes a single-keyword or multi-keyword AND query.
        Returns a structured dictionary with status, tokens, intersection steps, and ranked results.
        """
        if not query or not query.strip():
            return {
                "status": "empty_query",
                "message": "Boundary Condition: Empty or whitespace-only query entered.",
                "valid_terms": [],
                "ignored_terms": [],
                "results": [],
            }

        raw_words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        if not raw_words:
            return {
                "status": "special_chars_only",
                "message": "Boundary Condition: Query contained only punctuation/special characters.",
                "valid_terms": [],
                "ignored_terms": [],
                "results": [],
            }

        valid_terms = []
        ignored_terms = []
        for w in raw_words:
            if not w.isalpha() or w in STOP_WORDS or len(w) <= 1:
                ignored_terms.append(w)
            elif w not in valid_terms:
                valid_terms.append(w)

        if not valid_terms:
            return {
                "status": "only_stop_words",
                "message": f"Boundary Condition: All query words were stop words or invalid tokens: {ignored_terms}.",
                "valid_terms": [],
                "ignored_terms": ignored_terms,
                "results": [],
            }

        # Look up each valid index term in the Compressed Trie
        term_occ_lists = []
        for term in valid_terms:
            occ_idx = self.trie.search(term)
            if occ_idx is None:
                return {
                    "status": "term_not_found",
                    "message": f"Index term '{term}' was not found in the Compressed Trie (0 matching pages).",
                    "valid_terms": valid_terms,
                    "ignored_terms": ignored_terms,
                    "results": [],
                }
            term_occ_lists.append((term, occ_idx, self.occurrence_lists[occ_idx]))

        # Intersect occurrence lists sequentially using Sorted Sequence Merging
        intersected = term_occ_lists[0][2]
        for i in range(1, len(term_occ_lists)):
            intersected = sorted_sequence_intersect(intersected, term_occ_lists[i][2])
            if not intersected:
                return {
                    "status": "empty_intersection",
                    "message": f"All terms exist individually, but their intersection is empty (no single page contains all of {valid_terms}).",
                    "valid_terms": valid_terms,
                    "ignored_terms": ignored_terms,
                    "results": [],
                }

        # Rank matching documents:
        # Score = (10 * total_query_term_frequency) + (2 * incoming_hyperlink_count)
        ranked_results = []
        for item in intersected:
            doc = self.documents[item.doc_id]
            in_link_count = self.in_links.get(item.doc_id, 0)
            score = (10 * item.freq) + (2 * in_link_count)
            ranked_results.append({
                "rank": 0,
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "title": doc["title"],
                "term_freq": item.freq,
                "in_links": in_link_count,
                "score": score,
                "snippet": doc["text"][:180] + ("..." if len(doc["text"]) > 180 else ""),
            })

        ranked_results.sort(key=lambda r: (-r["score"], -r["term_freq"], r["doc_id"]))
        for idx, res in enumerate(ranked_results, start=1):
            res["rank"] = idx

        return {
            "status": "success",
            "message": f"Found {len(ranked_results)} matching page(s) for {valid_terms}.",
            "valid_terms": valid_terms,
            "ignored_terms": ignored_terms,
            "term_lookups": [(t, idx, len(lst)) for t, idx, lst in term_occ_lists],
            "results": ranked_results,
        }


# ==============================================================================
# 5. STREAMLIT USER INTERFACE
# ==============================================================================

st.set_page_config(page_title="Section 23.6 Search Engine", layout="wide")

st.title("Simplified Web Search Engine (Section 23.6)")
st.caption(
    "Implements a Web Crawler (BeautifulSoup), Compressed Trie, External Occurrence Lists Array, "
    "Sorted Sequence Merging Intersection, and Relevance Ranking."
)

# Initialize session state
if "pages" not in st.session_state:
    st.session_state.pages = dict(DEFAULT_SITE_PAGES)
if "engine" not in st.session_state:
    st.session_state.engine = SimplifiedSearchEngine()
    st.session_state.engine.crawl_and_index(st.session_state.pages, start_page="index.html", include_unlinked=False)

# Sidebar: Crawler Configuration & Page Upload
with st.sidebar:
    st.header("1. Web Crawler Settings")
    start_page = st.selectbox("Seed / Start Page", options=list(st.session_state.pages.keys()), index=0)
    include_unlinked = st.checkbox("Also index unlinked (orphan) pages", value=False)

    if st.button("Re-Crawl & Rebuild Index", type="primary", use_container_width=True):
        st.session_state.engine.crawl_and_index(
            st.session_state.pages,
            start_page=start_page,
            include_unlinked=include_unlinked,
        )
        st.success(f"Crawled {len(st.session_state.engine.documents)} pages & indexed {st.session_state.engine.trie.term_count} terms!")

    st.divider()
    st.header("2. Add / Upload HTML Pages")
    uploaded_files = st.file_uploader("Upload .html files", type=["html", "htm"], accept_multiple_files=True)
    if uploaded_files:
        for uf in uploaded_files:
            st.session_state.pages[uf.name] = uf.read().decode("utf-8", errors="ignore")
        st.session_state.engine.crawl_and_index(
            st.session_state.pages,
            start_page=start_page,
            include_unlinked=include_unlinked,
        )
        st.success(f"Added {len(uploaded_files)} file(s) and rebuilt index.")

    if st.button("Reset to Default 7 Pages", use_container_width=True):
        st.session_state.pages = dict(DEFAULT_SITE_PAGES)
        st.session_state.engine.crawl_and_index(st.session_state.pages, start_page="index.html", include_unlinked=False)
        st.rerun()

engine: SimplifiedSearchEngine = st.session_state.engine

# Top Metrics Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Crawled Pages", len(engine.documents))
col2.metric("Unique Index Terms", engine.trie.term_count)
col3.metric("Occurrence Lists", len(engine.occurrence_lists))
col4.metric("Stop Words Filtered", len(STOP_WORDS))

# Main Tabs
tab_search, tab_ds, tab_pages, tab_tests = st.tabs([
    "Search & Ranking",
    "Compressed Trie & Inverted Index",
    "Crawled Web Pages",
    "Boundary Tests & Project Deliverables"
])

# ------------------------------------------------------------------------------
# TAB 1: SEARCH & RANKING
# ------------------------------------------------------------------------------
with tab_search:
    st.subheader("Search Indexed Web Pages")
    query_input = st.text_input(
        "Enter single or multiple keywords (stop words are automatically removed):",
        value="compressed trie memory",
        placeholder="e.g., inverted index, compressed trie, crawler ranking...",
    )

    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    if sample_col1.button("Query: 'compressed trie'"):
        query_input = "compressed trie"
    if sample_col2.button("Query: 'inverted index'"):
        query_input = "inverted index"
    if sample_col3.button("Query: 'crawler ranking'"):
        query_input = "crawler ranking"
    if sample_col4.button("Query: 'the and of' (Stop words)"):
        query_input = "the and of"

    response = engine.search(query_input)

    if response["ignored_terms"]:
        st.info(f"Filtered out stop words / non-index tokens: `{response['ignored_terms']}`")

    if response["status"] != "success":
        st.warning(response["message"])
    else:
        st.success(response["message"])

        with st.expander("View Trie Lookups & Sorted Sequence Merging Trace", expanded=False):
            for term, idx, length in response["term_lookups"]:
                occ_str = ", ".join(str(x) for x in engine.occurrence_lists[idx])
                st.write(f"- Term **`{term}`** -> Trie external node points to `OccurrenceLists[{idx}]`: `[{occ_str}]`")

        for res in response["results"]:
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"### #{res['rank']} — {res['title']} (`{res['filename']}`)")
                c1.write(res["snippet"])
                c2.metric("Relevance Score", res["score"])
                c2.caption(f"Term Freq: `{res['term_freq']}` | In-Links: `{res['in_links']}`")

# ------------------------------------------------------------------------------
# TAB 2: COMPRESSED TRIE & EXTERNAL OCCURRENCE LISTS
# ------------------------------------------------------------------------------
with tab_ds:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Compressed Trie Structure (In-Memory)")
        st.caption("Internal nodes compress single-child chains; external nodes store ONLY the array index.")
        st.code(engine.trie.get_tree_visualization(), language="text")

    with right:
        st.subheader("External Occurrence Lists Array")
        st.caption("Each list is sorted in ascending order of `doc_id` for O(n + m) sequence merging.")
        terms_table = []
        for term, idx in engine.trie.get_all_terms():
            occ_items = engine.occurrence_lists[idx]
            terms_table.append({
                "Index Term": term,
                "Array Index": idx,
                "Sorted Occurrence List (doc_id, freq)": ", ".join(
                    f"[doc {x.doc_id} ({engine.documents[x.doc_id]['filename']}): {x.freq}]"
                    for x in occ_items
                ),
            })
        st.dataframe(terms_table, use_container_width=True, height=520)

# ------------------------------------------------------------------------------
# TAB 3: CRAWLED PAGES & LINK GRAPH
# ------------------------------------------------------------------------------
with tab_pages:
    st.subheader("BFS Web Crawler Log")
    st.code("\n".join(engine.crawl_log), language="text")

    st.subheader("Indexed Documents & Incoming Link Popularity")
    for doc in engine.documents:
        with st.expander(f"Doc ID {doc['doc_id']}: {doc['filename']} — {doc['title']} (In-Links: {engine.in_links[doc['doc_id']]})"):
            st.write("**Extracted Text:**", doc["text"])
            st.write("**Outgoing Hyperlinks:**", doc["links"])
            st.code(st.session_state.pages[doc["filename"]], language="html")

# ------------------------------------------------------------------------------
# TAB 4: BOUNDARY TESTS & DELIVERABLE EXPORTS
# ------------------------------------------------------------------------------
with tab_tests:
    st.subheader("Automated Boundary Condition Test Suite")
    boundary_queries = [
        ("", "Empty string query"),
        ("     ", "Whitespace-only query"),
        ("!!! @@@ ###", "Special characters / punctuation only"),
        ("the and of in to with", "Stop words only"),
        ("quantumblockchain", "Single keyword not present in corpus"),
        ("COMPRESSED TRIE", "Case-insensitivity check (uppercase)"),
        ("compressed trie", "Multi-word AND intersection (matches multiple docs)"),
        ("crawler sorting", "Two existing terms whose intersection is empty"),
        ("orphan", "Unlinked page test (depends on crawler toggle)"),
    ]

    test_rows = []
    output_log_lines = ["=== SEARCH ENGINE BOUNDARY CONDITION & SAMPLE RUN OUTPUT ===\n"]
    for q, desc in boundary_queries:
        res = engine.search(q)
        matched_files = [r["filename"] for r in res["results"]]
        test_rows.append({
            "Test Case": desc,
            "Query": repr(q),
            "Status": res["status"],
            "Matched Pages (Ranked)": ", ".join(matched_files) if matched_files else "None",
            "Details": res["message"],
        })
        output_log_lines.append(f"Test: {desc}\nQuery: {repr(q)}\nStatus: {res['status']}\nMessage: {res['message']}\nResults: {matched_files}\n" + "-" * 60)

    st.dataframe(test_rows, use_container_width=True)

    st.divider()
    st.subheader("Download Project Submission Files (Files #1, #3, #4)")

    readme_text = """# Simplified Web Search Engine (Section 23.6)

## 1. Overview & Approach
This project implements the simplified web search engine model from Section 23.6 using Python, BeautifulSoup4, and Streamlit.

## 2. Algorithms & Data Structures Used
1. **Web Crawler (Breadth-First Search + BeautifulSoup)**:
   - Starts from `index.html`, extracts `<a href="...">` hyperlinks and clean text via `BeautifulSoup`, and visits reachable pages in BFS order, assigning sequential `doc_id` integers (`0, 1, 2, ...`).
2. **Stop Word Filtering**:
   - Tokenizes words via regex (`[a-zA-Z]+`), lowercases them, and filters out common articles, prepositions, pronouns, and conjunctions.
3. **Compressed Trie (Radix Tree)**:
   - Stores all unique index terms in main memory by compressing chains of single-child nodes into multi-character edge labels (`O(k)` lookup where `k` is query length).
   - External/terminal nodes store **only** an integer index pointing into an external array of occurrence lists.
4. **External Occurrence Lists Array**:
   - Each entry is a list of `(doc_id, frequency)` objects sorted in ascending order of `doc_id`.
5. **Multi-Keyword Query Intersection (Sorted Sequence Merging)**:
   - Uses a two-pointer linear merge (`O(n1 + n2)`) across sorted occurrence lists to find pages containing **all** query keywords.
6. **Relevance Ranking Algorithm**:
   - Ranks matching documents using `Score = (10 * Total_Query_Term_Frequency) + (2 * Incoming_Hyperlink_Count)`.
"""

    combined_html_input = "\n\n".join(
        f"<!-- ==================== FILE: {fname} ==================== -->\n{content.strip()}"
        for fname, content in st.session_state.pages.items()
    )

    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button("Download README.md (File #1)", data=readme_text, file_name="README.md", mime="text/markdown")
    dl2.download_button("Download input_pages.html (File #3)", data=combined_html_input, file_name="input_pages.html", mime="text/html")
    dl3.download_button("Download sample_output.txt (File #4)", data="\n".join(output_log_lines), file_name="sample_output.txt", mime="text/plain")
