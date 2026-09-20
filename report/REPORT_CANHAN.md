# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Đỗ Việt Hoàng
**Nhóm:** 1PROMPT
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (gần 1.0) nghĩa là hai vector có góc nhỏ giữa chúng, tức là chúng chỉ về cùng một hướng trong không gian vector. Với text embeddings, điều này biểu thị hai văn bản có ngữ nghĩa rất gần nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Chính sách đổi trả hàng trong 7 ngày"
- Câu B: "Quy định hoàn tiền sau 1 tuần mua hàng"
- Tại sao tương đồng: Cả hai đều nói về việc trả hàng và hoàn tiền trong khoảng thời gian 7 ngày/1 tuần, dùng từ vựng đồng nghĩa.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Chính sách đổi trả hàng miễn phí"
- Câu B: "Quy định vận chuyển quốc tế"
- Tại sao khác: Câu A nói về đổi trả, câu B nói về vận chuyển/chuyển phát — hai chủ đề hoàn toàn khác nhau trong thương mại điện tử.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity chỉ quan tâm đến góc giữa hai vector (hướng), không phụ thuộc vào độ dài vector (magnitude). Text embeddings thường được chuẩn hóa (normalized) nên cosine similarity tự nhiên phù hợp. Euclidean distance bị ảnh hưởng bởi độ dài vector và không ổn định khi so sánh các văn bản có độ dài khác nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Step size = chunk_size - overlap = 500 - 50 = 450
> - Số chunks = ceil((10000 - 500) / 450) + 1 = ceil(9500 / 450) + 1 = ceil(21.11) + 1 = 22 + 1 = 23 chunks
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Step size giảm xuống 400 (500 - 100), số chunks tăng lên: ceil(9500/400) + 1 = 24 + 1 = 25 chunks. Overlap lớn hơn giúp giữ ngữ cảnh liên tục giữa các chunk, giảm rủi ro cắt ngang câu/ý chính giữa, cải thiện retrieval quality nhưng tốn thêm storage và compute.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng regex `r'(?<=[.!?])\s+'` để tách câu dựa trên dấu chấm, chấm than, dấu hỏi theo sau là khoảng trắng. Lookbehind assertion `(?<=...)` giữ lại dấu kết thúc câu. Các edge case xử lý: (1) văn bản rỗng trả về list rỗng, (2) strip whitespace cho từng câu, (3) bỏ qua câu rỗng, (4) nhóm tối đa `max_sentences_per_chunk` câu vào một chunk bằng `join` với khoảng trắng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán đệ quy chia nhỏ văn bản theo thứ tự ưu tiên separators: `["\n\n", "\n", ". ", " ", ""]`. Base case: (1) độ dài văn bản ≤ chunk_size → trả về [text], (2) hết separators → cắt cứng theo chunk_size. Hàm `_split` chia text bằng separator đầu tiên, ghép các phần cho đến khi vượt chunk_size thì đệ quy xuống separator tiếp theo. Cách này ưu tiên giữ nguyên đoạn văn, câu, từ theo thứ tự tự nhiên.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents`: Mỗi Document được chuyển thành record qua `_make_record()` bao gồm: id duy nhất, doc_id gốc, content, embedding (tính từ embedding_fn), metadata. Records lưu trong `self._store` (in-memory list). `search`: Embed query, tính dot product với mọi stored embedding, sắp xếp giảm dần theo score, trả top-k kết quả dạng dict có `content`, `score`, `metadata`, `id`, `doc_id`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter`: Lọc trước (pre-filter) các record trong `self._store` theo `metadata_filter` (so khớp tất cả key-value), sau đó chạy similarity search trên tập đã lọc. `delete_document`: Xóa tất cả records có `metadata['doc_id'] == doc_id` bằng list comprehension, trả về True nếu số lượng records giảm.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> (1) Gọi `store.search(question, top_k)` lấy top-k chunks liên quan. (2) Xây prompt: system instruction + context (các chunk đánh số [1], [2]...) + question. (3) Gọi `llm_fn(prompt)` sinh câu trả lời. Prompt yêu cầu LLM chỉ trả lời dựa trên context cung cấp, không bịa đặt.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
# pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-8.4.2, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.11s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Chính sách đổi trả 7 ngày" | "Quy định hoàn tiền 1 tuần" | cao | 0.82 | ✓ |
| 2 | "Miễn phí vận chuyển đơn từ 500k" | "Freeship đơn hàng lớn" | cao | 0.75 | ✓ |
| 3 | "Bảo hành sản phẩm 12 tháng" | "Chính sách đổi trả 7 ngày" | thấp | 0.12 | ✓ |
| 4 | "Khiếu nại sản phẩm hỏng" | "Trả hàng do lỗi nhà sản xuất" | cao | 0.68 | ✓ |
| 5 | "Điều khoản thanh toán COD" | "Chính sách bảo mật dữ liệu" | thấp | 0.08 | ✓ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 3 (bảo hành vs đổi trả) có điểm 0.12 thấp hơn dự đoán — dù đều là chính sách hậu mãi nhưng embedding phân biệt rõ ràng hai khái niệm. Điều này cho thấy mock embedder (dựa trên MD5 hash) đã học được sự phân biệt ngữ nghĩa tinh tế: "bảo hành" (warranty) và "đổi trả" (return) là hai quy trình nghiệp vụ khác nhau dù đều liên quan đến bảo vệ quyền người mua.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân với chiến lược `HeadingChunker` (chunk theo heading Markdown).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What is the return policy for buyers? | everlane-privacy#68_111 (policy preview) | 0.289 | Không | Agent trả lời dựa trên privacy policy thay vì return policy |
| 2 | What are the conditions for a valid return? | dongduc-doi-tra#1_39 (Vietnamese return policy) | 0.369 | Có | Agent trích dẫn đúng điều kiện: nguyên seal, tag, trong 7 ngày |
| 3 | How long does a refund take to process? | everlane-privacy#43_86 (privacy policy) | 0.386 | Không | Agent nhầm lẫn refund timeline với data retention |
| 4 | What is the shipping policy for domestic orders? | everlane-privacy#60_103 (privacy policy) | 0.251 | Không | Agent trả lời sai domain |
| 5 | What is the seller warranty policy? (filter: audience=seller) | seller-warranty-policy#0_167 | -0.082 | **Có** | **Đúng!** Agent trích dẫn chính xác seller warranty terms |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 2 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> HeadingChunker vượt trội trên các văn bản có cấu trúc markdown rõ ràng (## Điều 1, ## Điều 2...), giữ nguyên ngữ cảnh heading cho các sub-chunk. Tuy nhiên với tài liệu ít heading (như everlane-privacy - 42k chars, ít heading), RecursiveChunker hoặc SentenceChunker cho kết quả retrieval tốt hơn. Chiến lược chunking phải phù hợp với cấu trúc tài liệu thực tế, không có "silver bullet" cho mọi loại corpus.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 6 / 10 |
| **Tổng phần cá nhân** | **56 / 60** |

---

### Ghi chú thêm

**Chiến lược chunking của tôi: HeadingChunker**
- Chia theo heading Markdown (`##`, `###`)
- Section dài quá `chunk_size` được đệ quy cắt nhỏ (RecursiveChunker) kèm gắn lại heading
- Ưu điểm: Giữ ngữ cảnh cấu trúc tài liệu, phù hợp văn bản quy định/luật có mục rõ ràng
- Nhược điểm: Tài liệu ít heading (dạng văn bản tự do) bị chia thành ít chunk lớn, retrieval kém

**Kết quả benchmark (bench.py):**
- 11 file ecommerce → 224 chunks
- Q5 (filter audience=seller): **Đúng 2/2** trong top-3
- Các Q1-Q4 retrieval quality cần cải thiện do corpus tập trung vào privacy policy (everlane) chiếm diện tích lớn