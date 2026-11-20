# Simplified Search Engine (Compressed Trie & Inverted Index)

An implementation of the simplified search engine model described in **Section 23.6** (*Data Structures and Algorithms*, Goodrich & Tamassia)[cite: 1], featuring a **BFS Web Crawler**, a **Compressed Trie (Radix Tree)** for index terms[cite: 1], an **Array of Occurrence Lists**[cite: 1], **Sorted-Sequence Intersection** for multi-keyword queries[cite: 1], and a **Term-Frequency + In-Degree Page Ranking** algorithm.

---

## 1. Data Structures Used

Strictly following the textbook specification[cite: 1]:
1. **Compressed Trie (Radix Tree)**:
   - Stores all unique index terms (vocabulary words extracted from crawled pages, excluding stop words)[cite: 1].
   - Unlike a standard trie, chains of single-child internal nodes are compressed into a single edge labeled with a substring, reducing space complexity to $O(s)$ where $s$ is the number of index terms.
   - Each **external (leaf) node** does **not** store the document list directly; instead, it stores an integer index pointing into a separate Occurrence Lists array[cite: 1]. This keeps the trie compact in memory[cite: 1].
2. **Array of Occurrence Lists**:
   - A dynamic array where each slot $i$ corresponds to a distinct index term[cite: 1].
   - Each occurrence list is stored as a **sorted sequence of `(doc_id, term_frequency)` pairs**, ordered by `doc_id` (document address/ID)[cite: 1].
3. **Adjacency Graph (Arrays / Lists)**:
   - Tracks directed hyperlinks between crawled pages so we can compute each page's **in-degree** (number of incoming links) for relevance ranking.

---

## 2. Algorithms Used

1. **Web Crawler (Breadth-First Search — BFS)**:
   - Starts at `index.html`, parses HTML using `BeautifulSoup`, extracts `<a href="...">` links, and traverses unvisited local pages in FIFO queue order.
2. **Text Tokenization & Stop-Word Filtering**:
   - Extracts visible text via BeautifulSoup, converts to lowercase, tokenizes alphabetic words via regex, and removes standard English stop words (articles, prepositions, pronouns, conjunctions).
3. **Compressed Trie Construction & Search ($O(|P|)$)**:
   - **Insertion**: Splits existing edges at the longest common prefix when a new word diverges from an existing edge label.
   - **Lookup**: Traverses edge labels matching prefixes of the query keyword $P$ in time proportional to the keyword length $|P|$, returning the `occurrence_list_index` stored at the external node[cite: 1].
4. **Multi-Keyword Query Intersection (Sorted Sequence Merging — Section 8.2)**:
   - Given $k$ keywords, retrieves the sorted occurrence list for each keyword from the array[cite: 1].
   - Uses a two-pointer linear merge (`intersect_sorted_lists`) in $O(n_1 + n_2)$ time per pair to find documents containing **all** query terms (`AND` semantics)[cite: 1].
5. **Relevance Ranking Algorithm**:
   - Matching documents are scored and sorted in descending order using a composite score:
     $$\text{Score}(d, Q) = \sum_{w \in Q} \text{TF}(w, d) + 2 \times \text{InDegree}(d)$$
     - **Term Frequency ($\text{TF}$)**: How many times the query words appear in document $d$[cite: 1].
     - **Link Authority ($\text{InDegree}$)**: Bonus weight for pages frequently referenced by other pages on the site.

---
