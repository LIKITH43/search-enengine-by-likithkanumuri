"""
Simplified Search Engine Implementation (Section 23.6.4)
- Compressed Trie (Radix Tree) storing external references to an Occurrence Lists Array.
- Sorted Occurrence Lists enabling linear sorted-sequence intersection (Section 8.2).
- BeautifulSoup HTML parser & BFS Web Crawler.
- Term-Frequency + In-Link Relevance Ranking.
"""

from collections import deque
import json
import re
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

# Standard English stop words: articles, prepositions, pronouns, and common auxiliaries
STOP_WORDS = [
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "as", "until", "while", "of", "can", "will", "just", "should", "now"
]


class CompressedTrieNode:
    """
    A node in the Compressed Trie (Radix Tree).
    - children: maps first character -> (edge_label, CompressedTrieNode)
    - occurrence_index: integer index into the external occurrence_lists array,
      or None if this is purely an internal routing node.
    """

    def __init__(self) -> None:
        self.children: Dict[str, Tuple[str, "CompressedTrieNode"]] = {}
        self.occurrence_index: Optional[int] = None


class CompressedTrie:
    """
    Compressed Trie storing index terms.
    External/terminal nodes store only the integer index of the occurrence list
    in the separate occurrence_lists array (Section 23.6.4).
    """

    def __init__(self) -> None:
        self.root = CompressedTrieNode()
        self.term_count = 0

    @staticmethod
    def _common_prefix_len(s1: str, s2: str) -> int:
        limit = min(len(s1), len(s2))
        idx = 0
        while idx < limit and s1[idx] == s2[idx]:
            idx += 1
        return idx

    def insert(self, word: str, occurrence_index: int) -> None:
        """Inserts a word into the compressed trie, splitting edges when needed."""
        node = self.root
        remaining = word

        while remaining:
            first_char = remaining[0]
            if first_char not in node.children:
                new_leaf = CompressedTrieNode()
                new_leaf.occurrence_index = occurrence_index
                node.children[first_char] = (remaining, new_leaf)
                self.term_count += 1
                return

            edge_label, child = node.children[first_char]
            prefix_len = self._common_prefix_len(remaining, edge_label)

            if prefix_len == len(edge_label):
                remaining = remaining[prefix_len:]
                node = child
            else:
                common_prefix = edge_label[:prefix_len]
                edge_suffix = edge_label[prefix_len:]
                word_suffix = remaining[prefix_len:]

                split_node = CompressedTrieNode()
                split_node.children[edge_suffix[0]] = (edge_suffix, child)
                node.children[first_char] = (common_prefix, split_node)

                if not word_suffix:
                    split_node.occurrence_index = occurrence_index
                    self.term_count += 1
                else:
                    new_leaf = CompressedTrieNode()
                    new_leaf.occurrence_index = occurrence_index
                    split_node.children[word_suffix[0]] = (word_suffix, new_leaf)
                    self.term_count += 1
                return

        if node.occurrence_index is None:
            self.term_count += 1
        node.occurrence_index = occurrence_index

    def search(self, word: str) -> Optional[int]:
        """
        Searches for an exact index term in O(|word|) time.
        Returns the integer index in the occurrence_lists array, or None if absent.
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

    def collect_terms(self) -> List[Tuple[str, int]]:
        """Traverses the compressed trie and returns sorted (term, occurrence_index) pairs."""
        entries: List[Tuple[str, int]] = []

        def _dfs(curr: CompressedTrieNode, prefix: str) -> None:
            if curr.occurrence_index is not None:
                entries.append((prefix, curr.occurrence_index))
            for _, (label, nxt) in sorted(curr.children.items()):
                _dfs(nxt, prefix + label)

        _dfs(self.root, "")
        entries.sort(key=lambda item: item[0])
        return entries


class SimplifiedSearchEngine:
    """
    Search engine implementing:
    1. Web crawler over hyperlinked HTML pages using BeautifulSoup.
    2. Inverted index backed by a Compressed Trie + external Occurrence Lists Array.
    3. Sorted-sequence intersection for multi-keyword queries (Section 23.6.4 / 8.2).
    4. Relevance ranking using term frequency and incoming link count.
    """

    def __init__(self) -> None:
        self.trie = CompressedTrie()
        # Array storing occurrence lists in no particular order (Section 23.6.4).
        # Each entry is a list of (doc_id, frequency) sorted ascending by doc_id.
        self.occurrence_lists: List[List[Tuple[int, int]]] = []
        self.documents: List[Dict] = []
        self.url_to_doc_id: Dict[str, int] = {}
        self.in_degrees: Dict[int, int] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Extracts lowercase alphabetic words and removes stop words."""
        raw_tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return [tok for tok in raw_tokens if tok not in STOP_WORDS and len(tok) > 1]

    @staticmethod
    def _unique_ordered_terms(tokens: List[str]) -> List[str]:
        unique_terms: List[str] = []
        for tok in tokens:
            if tok not in unique_terms:
                unique_terms.append(tok)
        return unique_terms

    def crawl_and_index(self, pages: Dict[str, str], seed_urls: Optional[List[str]] = None) -> List[str]:
        """
        Crawls pages starting from seed_urls using BFS over <a href> links,
        parses HTML with BeautifulSoup, and builds the inverted index.
        """
        self.trie = CompressedTrie()
        self.occurrence_lists = []
        self.documents = []
        self.url_to_doc_id = {}
        self.in_degrees = {}

        if not pages:
            return []

        if not seed_urls:
            seed_urls = [next(iter(pages.keys()))]

        visited_urls: List[str] = []
        queue = deque(url for url in seed_urls if url in pages)
        crawled_order: List[str] = []
        discovered_links: List[Tuple[str, str]] = []

        # Step 1: BFS Web Crawl across hyperlinked pages
        while queue:
            current_url = queue.popleft()
            if current_url in visited_urls or current_url not in pages:
                continue

            visited_urls.append(current_url)
            doc_id = len(self.documents)
            self.url_to_doc_id[current_url] = doc_id
            self.in_degrees[doc_id] = 0

            soup = BeautifulSoup(pages[current_url], "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else current_url
            visible_text = soup.get_text(separator=" ", strip=True)

            out_links: List[str] = []
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"].strip()
                if href in pages:
                    out_links.append(href)
                    discovered_links.append((current_url, href))
                    if href not in visited_urls and href not in queue:
                        queue.append(href)

            self.documents.append({
                "doc_id": doc_id,
                "url": current_url,
                "title": title,
                "text": visible_text,
                "out_links": out_links,
            })
            crawled_order.append(current_url)

        # Compute incoming link counts (in-degree) for ranking boost
        for src_url, dst_url in discovered_links:
            if dst_url in self.url_to_doc_id and src_url != dst_url:
                dst_id = self.url_to_doc_id[dst_url]
                self.in_degrees[dst_id] = self.in_degrees.get(dst_id, 0) + 1

        # Step 2: Build Inverted Index (ascending doc_id keeps occurrence lists sorted by address)
        for doc in self.documents:
            doc_id = doc["doc_id"]
            tokens = self.tokenize(f"{doc['title']} {doc['text']}")
            unique_doc_terms = self._unique_ordered_terms(tokens)

            for term in unique_doc_terms:
                freq = sum(1 for t in tokens if t == term)
                occ_idx = self.trie.search(term)
                if occ_idx is None:
                    occ_idx = len(self.occurrence_lists)
                    self.occurrence_lists.append([(doc_id, freq)])
                    self.trie.insert(term, occ_idx)
                else:
                    self.occurrence_lists[occ_idx].append((doc_id, freq))

        return crawled_order

    @staticmethod
    def intersect_sorted_lists(
        list_a: List[Tuple[int, int]], list_b: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Sorted-sequence intersection algorithm (Section 8.2 / 23.6.4).
        Both input lists are sorted by doc_id ascending.
        Merges matching doc_ids and sums their term frequencies in O(|A| + |B|) time.
        """
        i, j = 0, 0
        merged: List[Tuple[int, int]] = []

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

    def search_query(self, raw_query: str) -> Dict:
        """Executes a single- or multi-keyword search query."""
        if not raw_query or not raw_query.strip():
            return {
                "status": "empty_query",
                "message": "Query is empty. Please enter one or more keywords.",
                "terms": [],
                "results": [],
            }

        terms = self._unique_ordered_terms(self.tokenize(raw_query))
        if not terms:
            return {
                "status": "only_stop_words",
                "message": "All words in your query were filtered out as stop words.",
                "terms": [],
                "results": [],
            }

        retrieved_lists: List[List[Tuple[int, int]]] = []
        for term in terms:
            occ_idx = self.trie.search(term)
            if occ_idx is None:
                return {
                    "status": "no_match",
                    "message": f"No pages matched all query terms (missing index term: '{term}').",
                    "terms": terms,
                    "results": [],
                }
            retrieved_lists.append(self.occurrence_lists[occ_idx])

        retrieved_lists.sort(key=len)
        intersected = retrieved_lists[0]
        for next_list in retrieved_lists[1:]:
            intersected = self.intersect_sorted_lists(intersected, next_list)
            if not intersected:
                return {
                    "status": "no_match",
                    "message": "No single page contains all of the searched keywords.",
                    "terms": terms,
                    "results": [],
                }

        ranked_results = []
        for doc_id, combined_tf in intersected:
            doc = self.documents[doc_id]
            in_links = self.in_degrees.get(doc_id, 0)
            score = round(combined_tf + 0.5 * in_links, 2)
            ranked_results.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "url": doc["url"],
                "term_frequency": combined_tf,
                "in_links": in_links,
                "score": score,
                "snippet": doc["text"][:180] + ("..." if len(doc["text"]) > 180 else ""),
            })

        ranked_results.sort(key=lambda r: (-r["score"], r["doc_id"]))

        return {
            "status": "success",
            "message": f"Found {len(ranked_results)} matching page(s).",
            "terms": terms,
            "results": ranked_results,
        }


def run_cli_tests() -> str:
    """Runs standard and boundary condition tests and formats output for Deliverable 4."""
    with open("input_pages.json", "r", encoding="utf-8") as f:
        pages = json.load(f)

    engine = SimplifiedSearchEngine()
    crawled = engine.crawl_and_index(pages, seed_urls=["index.html"])

    lines = [
        "======================================================================",
        "SIMPLIFIED SEARCH ENGINE (SECTION 23.6.4) - SAMPLE & BOUNDARY TEST RUN",
        "======================================================================",
        f"Crawled Pages ({len(crawled)}): {', '.join(crawled)}",
        f"Unique Indexed Terms in Compressed Trie: {engine.trie.term_count}",
        f"External Occurrence Lists Array Size: {len(engine.occurrence_lists)}",
        "----------------------------------------------------------------------",
    ]

    test_queries = [
        ("Single Keyword Query", "trie"),
        ("Multi-Keyword Intersection Query", "search engine algorithm"),
        ("Case-Insensitive & Stop-Word Mixed Query", "The COMPRESSED trie AND occurrence"),
        ("Boundary Test 1: Empty String", ""),
        ("Boundary Test 2: Whitespace Only", "     "),
        ("Boundary Test 3: Stop Words Only", "the and of in to with"),
        ("Boundary Test 4: Punctuation & Numbers Only", "12345 !!! ??? ---"),
        ("Boundary Test 5: Non-Existent Keyword", "quantumblockchain"),
        ("Boundary Test 6: Partial Intersection Failure (terms exist separately, not together)", "crawler binary"),
    ]

    for label, query in test_queries:
        res = engine.search_query(query)
        lines.append(f"\n[TEST] {label}")
        lines.append(f"  Raw Query     : {repr(query)}")
        lines.append(f"  Indexed Terms : {res['terms']}")
        lines.append(f"  Status        : {res['status']} -> {res['message']}")
        for rank, item in enumerate(res["results"], start=1):
            lines.append(
                f"    #{rank} [{item['url']}] '{item['title']}' | "
                f"Score={item['score']} (TF={item['term_frequency']}, InLinks={item['in_links']})"
            )

    report = "\n".join(lines)
    with open("sample_output.txt", "w", encoding="utf-8") as out:
        out.write(report + "\n")
    return report


if __name__ == "__main__":
    print(run_cli_tests())
