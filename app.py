import os
import re
from collections import deque
from bs4 import BeautifulSoup
import streamlit as st

# Standard stop words: articles, prepositions, pronouns, conjunctions, auxiliary verbs
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "i", "me", "my", "myself", "we", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
    "its", "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "as", "of", "until", "while"
}


class CompressedTrieNode:
    """
    A node in a Compressed Trie (Radix Tree).
    - children: dict mapping first character -> (edge_label: str, child_node: CompressedTrieNode)
    - occurrence_index: int or None. External/terminal nodes store the integer index
      into the external Occurrence Lists array (keeping the trie small in memory).
    """
    def __init__(self):
        self.children = {}
        self.occurrence_index = None


class CompressedTrie:
    """
    Compressed Trie storing index terms.
    Edges are labeled with substrings rather than single characters so chains
    of single-child nodes are compressed into a single edge.
    """
    def __init__(self):
        self.root = CompressedTrieNode()
        self.node_count = 1

    @staticmethod
    def _common_prefix_length(s1: str, s2: str) -> int:
        limit = min(len(s1), len(s2))
        idx = 0
        while idx < limit and s1[idx] == s2[idx]:
            idx += 1
        return idx

    def insert(self, word: str, occurrence_index: int) -> None:
        """Inserts `word` into the compressed trie and associates `occurrence_index`."""
        node = self.root
        remaining = word

        while remaining:
            first_char = remaining[0]
            if first_char not in node.children:
                # Create a new external leaf node for the remaining substring
                new_leaf = CompressedTrieNode()
                new_leaf.occurrence_index = occurrence_index
                node.children[first_char] = (remaining, new_leaf)
                self.node_count += 1
                return

            edge_label, child = node.children[first_char]
            common_len = self._common_prefix_length(remaining, edge_label)

            if common_len == len(edge_label):
                # Full edge matched; continue down the trie
                node = child
                remaining = remaining[common_len:]
            else:
                # Split the existing edge at `common_len`
                split_node = CompressedTrieNode()
                self.node_count += 1

                # Existing child becomes a child of split_node with remaining edge suffix
                existing_suffix = edge_label[common_len:]
                split_node.children[existing_suffix[0]] = (existing_suffix, child)

                # Update current node's edge to point to split_node
                common_prefix = edge_label[:common_len]
                node.children[first_char] = (common_prefix, split_node)

                new_suffix = remaining[common_len:]
                if new_suffix:
                    new_leaf = CompressedTrieNode()
                    new_leaf.occurrence_index = occurrence_index
                    split_node.children[new_suffix[0]] = (new_suffix, new_leaf)
                    self.node_count += 1
                else:
                    split_node.occurrence_index = occurrence_index
                return

        # Exact match ended on an existing internal node
        node.occurrence_index = occurrence_index

    def search(self, word: str):
        """
        Searches for `word` in the compressed trie.
        Returns the integer `occurrence_index` if found, otherwise None.
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

    def get_structure_preview(self, max_depth: int = 4) -> list[str]:
        """Generates a human-readable tree view of the compressed trie edges."""
        lines = []

        def _dfs(curr: CompressedTrieNode, prefix: str, depth: int):
            if depth > max_depth:
                return
            items = sorted(curr.children.items(), key=lambda x: x[0])
            for i, (_, (label, child)) in enumerate(items):
                connector = "└── " if i == len(items) - 1 else "├── "
                leaf_tag = f"  [OccList #{child.occurrence_index}]" if child.occurrence_index is not None else ""
                lines.append(f"{prefix}{connector}'{label}'{leaf_tag}")
                extension = "    " if i == len(items) - 1 else "│   "
                _dfs(child, prefix + extension, depth + 1)

        _dfs(self.root, "", 1)
        return lines



class SimplifiedSearchEngine:
    def __init__(self):
        self.trie = CompressedTrie()
        # Array storing occurrence lists of index terms (Section 23.6)
        # Each entry is a list of (doc_id, frequency) sorted by doc_id
        self.occurrence_lists: list[list[tuple[int, int]]] = []
        self.terms_by_index: list[str] = []

        # Document metadata storage
        self.documents: list[dict] = []  # indexed by doc_id
        self.in_degrees: list[int] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Extracts lowercase alphabetic tokens and filters out stop words."""
        raw_words = re.findall(r"[a-zA-Z]+", text.lower())
        return [w for w in raw_words if w not in STOP_WORDS and len(w) > 1]

    def crawl_and_index(self, folder_path: str, start_page: str = "index.html") -> None:
        """
        Crawls interlinked HTML files using Breadth-First Search (BFS),
        parses content with BeautifulSoup, and builds the inverted index.
        """
        self.trie = CompressedTrie()
        self.occurrence_lists = []
        self.terms_by_index = []
        self.documents = []

        start_full = os.path.join(folder_path, start_page)
        if not os.path.exists(start_full):
            raise FileNotFoundError(f"Seed page '{start_page}' not found in '{folder_path}'.")

        visited = set()
        queue = deque([start_page])
        visited.add(start_page)

        raw_pages = []
        filename_to_id = {}

        # Step 1: BFS Web Crawl following <a href="..."> links
        while queue:
            current_file = queue.popleft()
            full_path = os.path.join(folder_path, current_file)
            if not os.path.exists(full_path):
                continue

            with open(full_path, "r", encoding="utf-8") as f:
                html = f.read()

            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else current_file

            # Extract outgoing hyperlinks
            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                if href.endswith(".html") and not href.startswith(("http://", "https://")):
                    links.append(href)
                    if href not in visited and os.path.exists(os.path.join(folder_path, href)):
                        visited.add(href)
                        queue.append(href)

            # Extract visible body text
            body_text = soup.get_text(separator=" ", strip=True)
            tokens = self.tokenize(body_text)

            doc_id = len(raw_pages)
            filename_to_id[current_file] = doc_id
            raw_pages.append({
                "doc_id": doc_id,
                "filename": current_file,
                "title": title,
                "text": body_text,
                "tokens": tokens,
                "out_links": links
            })

        self.documents = raw_pages
        self.in_degrees = [0] * len(self.documents)

        # Compute link in-degree for ranking
        for doc in self.documents:
            for target in set(doc["out_links"]):
                if target in filename_to_id:
                    self.in_degrees[filename_to_id[target]] += 1

        # Step 2: Build Inverted Index (Compressed Trie + Array of Occurrence Lists)
        # Documents are processed in ascending doc_id order (0, 1, 2, ...),
        # so each occurrence list is naturally sorted by doc_id!
        for doc in self.documents:
            doc_id = doc["doc_id"]
            term_freqs: dict[str, int] = {}
            for token in doc["tokens"]:
                term_freqs[token] = term_freqs.get(token, 0) + 1

            for term, freq in sorted(term_freqs.items()):
                occ_idx = self.trie.search(term)
                if occ_idx is None:
                    occ_idx = len(self.occurrence_lists)
                    self.occurrence_lists.append([(doc_id, freq)])
                    self.terms_by_index.append(term)
                    self.trie.insert(term, occ_idx)
                else:
                    self.occurrence_lists[occ_idx].append((doc_id, freq))

    @staticmethod
    def intersect_sorted_lists(
        list_a: list[tuple[int, int]], list_b: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """
        Sorted sequence intersection algorithm (Section 8.2 / Section 23.6).
        Merges two occurrence lists sorted by doc_id in O(|A| + |B|) time,
        summing term frequencies for matching documents.
        """
        i, j = 0, 0
        merged = []
        while i < len(list_a) and j < len(list_b):
            doc_a, freq_a = list_a[i]
            doc_b, freq_b = list_b[j]
            if doc_a == doc_b:
                merged.append((doc_a, freq_a + freq_b))
                i += 1
                j += 1
            elif doc_a < doc_b:
                i += 1
            else:
                j += 1
        return merged

    def search(self, query: str) -> dict:
        """
        Executes a single- or multi-keyword query:
        1. Tokenizes and removes stop words from query.
        2. Looks up each keyword in the Compressed Trie.
        3. Intersects occurrence lists for ALL keywords.
        4. Ranks results by: Score = Total_Term_Frequency + 2 * Page_In_Degree.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return {"status": "empty_query", "terms": [], "results": []}

        terms = self.tokenize(cleaned_query)
        if not terms:
            return {"status": "only_stop_words", "terms": [], "results": []}

        # Deduplicate query terms while preserving order
        unique_terms = list(dict.fromkeys(terms))

        # Retrieve occurrence lists from trie + array
        retrieved_lists = []
        for term in unique_terms:
            occ_idx = self.trie.search(term)
            if occ_idx is None:
                # Since query requires ALL keywords, one missing term -> empty result
                return {
                    "status": "not_found",
                    "missing_term": term,
                    "terms": unique_terms,
                    "results": []
                }
            retrieved_lists.append(self.occurrence_lists[occ_idx])

        # Intersect all sorted occurrence lists
        intersected = retrieved_lists[0]
        for next_list in retrieved_lists[1:]:
            intersected = self.intersect_sorted_lists(intersected, next_list)
            if not intersected:
                break

        if not intersected:
            return {"status": "no_intersection", "terms": unique_terms, "results": []}

        # Rank matching pages
        ranked_results = []
        for doc_id, combined_tf in intersected:
            doc = self.documents[doc_id]
            in_deg = self.in_degrees[doc_id]
            score = combined_tf + (2 * in_deg)

            # Build snippet highlighting matched terms
            snippet = self._build_snippet(doc["text"], unique_terms)
            ranked_results.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "filename": doc["filename"],
                "term_frequency": combined_tf,
                "in_degree": in_deg,
                "score": score,
                "snippet": snippet,
                "out_links": doc["out_links"]
            })

        # Sort by descending relevance score, tie-break by doc_id
        ranked_results.sort(key=lambda r: (-r["score"], r["doc_id"]))
        return {"status": "ok", "terms": unique_terms, "results": ranked_results}

    @staticmethod
    def _build_snippet(text: str, terms: list[str], window: int = 180) -> str:
        lower_text = text.lower()
        pos = -1
        for t in terms:
            idx = lower_text.find(t)
            if idx != -1 and (pos == -1 or idx < pos):
                pos = idx
        if pos == -1:
            return text[:window] + ("..." if len(text) > window else "")
        start = max(0, pos - 40)
        end = min(len(text), start + window)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"



def main():
    st.set_page_config(page_title="Compressed Trie Search Engine", page_icon="🔍", layout="wide")
    st.title("🔍 Simplified Web Search Engine ")
    st.caption("Powered by a BFS Web Crawler, Compressed Trie, Sorted Occurrence Lists, and Link-Weighted TF Ranking.")

    pages_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")

    if "engine" not in st.session_state:
        engine = SimplifiedSearchEngine()
        engine.crawl_and_index(pages_dir, "index.html")
        st.session_state.engine = engine

    engine: SimplifiedSearchEngine = st.session_state.engine

    # Sidebar: Crawl Controls & Data Structure Metrics
    with st.sidebar:
        st.header("Crawler & Index Stats")
        if st.button("🔄 Re-Crawl & Rebuild Index"):
            engine.crawl_and_index(pages_dir, "index.html")
            st.success("Crawled and indexed all reachable pages!")

        col1, col2 = st.columns(2)
        col1.metric("Pages Crawled", len(engine.documents))
        col2.metric("Index Terms", len(engine.occurrence_lists))
        st.metric("Compressed Trie Nodes", engine.trie.node_count)

        st.divider()
        st.subheader("Crawled Pages (BFS Order)")
        for doc in engine.documents:
            st.markdown(
                f"**[{doc['doc_id']}] {doc['title']}** (`{doc['filename']}`)  \n"
                f"In-links: `{engine.in_degrees[doc['doc_id']]}` | Tokens: `{len(doc['tokens'])}`"
            )

    # Main Search Tabs
    tab_search, tab_index, tab_trie, tab_tests = st.tabs([
        "🔎 Search Engine",
        "📑 Inverted Index & Occurrence Lists",
        "🌳 Compressed Trie Structure",
        "🧪 Boundary Condition Tests"
    ])

    with tab_search:
        query = st.text_input(
            "Enter search keywords (multi-word queries return pages containing ALL keywords):",
            placeholder="e.g., compressed trie search"
        )

        # Quick example pills
        st.markdown("**Sample Queries:** `trie` • `compressed trie` • `web crawler graph` • `the and of` • `quantum`")

        if query:
            response = engine.search(query)
            status = response["status"]

            if status == "empty_query":
                st.warning("Please enter at least one keyword.")
            elif status == "only_stop_words":
                st.warning("Your query contained only stop words (e.g., 'the', 'and', 'of'). Try meaningful keywords.")
            elif status == "not_found":
                st.error(f"No matches found: index term **`{response['missing_term']}`** does not appear in any crawled page.")
            elif status == "no_intersection":
                st.info(f"Each keyword exists individually, but **no single page** contains all of: `{', '.join(response['terms'])}`.")
            else:
                st.success(f"Found **{len(response['results'])}** matching page(s) for terms: `{', '.join(response['terms'])}`")
                for rank, res in enumerate(response["results"], start=1):
                    with st.container(border=True):
                        st.markdown(f"### #{rank} — {res['title']} (`{res['filename']}`)")
                        st.write(res["snippet"])
                        st.caption(
                            f"**Relevance Score:** `{res['score']}` "
                            f"(Term Frequency: `{res['term_frequency']}` + 2 × In-Degree `{res['in_degree']}`) | "
                            f"**Doc ID:** `{res['doc_id']}`"
                        )

    with tab_index:
        st.subheader("Inverted File: Compressed Trie Pointer → Array of Sorted Occurrence Lists")
        st.write(
            "Each external node in the Compressed Trie stores an integer index pointing into the "
            "external Occurrence Lists array. Each list stores `(doc_id, frequency)` pairs sorted by `doc_id`."
        )
        rows = []
        for idx, term in enumerate(engine.terms_by_index):
            occ = engine.occurrence_lists[idx]
            rows.append({
                "Array Index": idx,
                "Index Term": term,
                "Doc Count": len(occ),
                "Sorted Occurrence List (doc_id, freq)": str(occ)
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab_trie:
        st.subheader("Compressed Trie (Radix Tree) Prefix Visualization")
        st.write("Showing top levels of compressed edge labels and their target Occurrence List array indices:")
        tree_lines = engine.trie.get_structure_preview(max_depth=4)
        st.code("\n".join(tree_lines), language="text")

    with tab_tests:
        st.subheader("Automated Boundary Condition Verification")
        test_cases = [
            ("Single keyword match", "trie"),
            ("Multi-keyword AND intersection", "compressed trie"),
            ("Case-insensitivity & punctuation", "CRAWLER, Graph!!"),
            ("Stop-words mixed with valid term", "the search of a engine"),
            ("Only stop words (Boundary)", "the and is of to"),
            ("Empty / whitespace query (Boundary)", "   "),
            ("Term absent from corpus (Boundary)", "blockchain"),
            ("Disjoint terms with empty intersection (Boundary)", "pagerank suffix")
        ]
        for label, sample_q in test_cases:
            out = engine.search(sample_q)
            matched_files = [r["filename"] for r in out["results"]]
            st.markdown(
                f"- **{label}** — Query: `{sample_q}` → Status: `{out['status']}` | "
                f"Matches: `{matched_files}`"
            )


if __name__ == "__main__":
    main()
