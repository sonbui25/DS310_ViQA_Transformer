# BÁO CÁO THỰC HÀNH LAB 3: QUESTION ANSWERING

**Môn học:** DS310 - Xử lý ngôn ngữ tự nhiên cho Khoa học dữ liệu  
**Bộ dữ liệu:** UIT-ViQuAD 2.0  
**Sinh viên thực hiện:** Bùi Trương Thái Sơn

## 1. Giới thiệu

Bài thực hành tập trung giải quyết bài toán Trả lời câu hỏi (Question Answering) trên văn bản tiếng Việt. Mục tiêu là xây dựng hệ thống có khả năng trích xuất chính xác đoạn văn bản (span) chứa câu trả lời từ ngữ cảnh (context) cho trước.

Thực nghiệm được tiến hành theo hai phương pháp tiếp cận chính:

1. **Fine-tuning (Tinh chỉnh):** Sử dụng các mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Models) dạng Encoder.
2. **Prompting:** Sử dụng các mô hình ngôn ngữ lớn (Large Language Models - LLM) dạng Decoder với kỹ thuật Zero-shot và Few-shot. 



## 2. Kết quả thực nghiệm

Kết quả được đánh giá trên tập Test của bộ dữ liệu UIT-ViQuAD 2.0 dựa trên 2 độ đo tiêu chuẩn:

- **Exact Match (EM):** Tỉ lệ câu trả lời khớp chính xác tuyệt đối với đáp án mẫu.
- **F1-Score:** Độ đo trung bình điều hòa giữa Precision và Recall ở cấp độ từ ngữ (token overlap).

### 2.1. Phương pháp Fine-tuning (Transformer Encoder)

Nhóm này bao gồm các mô hình BERT-based được huấn luyện lại toàn bộ trọng số trên tập Train.

| Mô hình (Model) | Exact Match (EM) | F1-Score | Ghi chú |
|---|---|---|---|
| mBERT (base) | 46.42 | 57.21 | Mức cơ sở (Baseline) ổn định cho đa ngôn ngữ. |
| XLM-RoBERTa (base) | 48.65 | 59.02 | Tăng ~1.8% F1 so với mBERT, hiệu quả nhất nhóm. |
| PhoBERT (base) | 27.64 | 47.77 | Thấp hơn mBERT (~9.4% F1), chưa tối ưu tốt. |

**Nhận xét:**
- Trong nhóm này, XLM-RoBERTa đạt hiệu năng cao nhất với 48.65% EM và 59.02% F1.
- Kết quả của PhoBERT thấp hơn đáng kể so với mBERT và XLM-R trong lần chạy thử nghiệm này (EM chỉ đạt 27.64). Điều này có thể do bộ tham số huấn luyện (learning rate, số epochs) chưa được tối ưu hoàn toàn cho mô hình đơn ngữ này.

### 2.2. Phương pháp Prompting (LLM Decoder)

Nhóm này thực hiện suy luận (inference) trực tiếp mà không cập nhật trọng số, sử dụng kỹ thuật In-context Learning.

**Note:** Ở mô hình GPT-4o-mini do vấn đề về kinh phí cho API nên chúng tôi chỉ thực hiện kịch bản Zero-shot.

| Mô hình | Kịch bản (Scenario) | Exact Match (EM) | F1-Score | Ghi chú |
|---|---|---|---|---|
| Vinallama-2.7B-chat | Zero-shot | 12.42 | 33.48 | Kết quả khởi điểm thấp. |
| Vinallama-2.7B-chat | Few-shot (3 examples) | 16.65 | 35.02 | Tăng nhẹ +1.54% F1 nhờ có ví dụ mẫu. |
| Qwen1.5-4B-Chat | Zero-shot | 16.03 | 35.08 | Cao hơn Vinallama ở mức Zero-shot. |
| Qwen1.5-4B-Chat | Few-shot (3 examples) | 16.24 | 38.01 | Tăng +2.93% F1, học từ ngữ cảnh tốt hơn Vinallama. |
| GPT-4o-mini | Zero-shot | 41.95 | 62.61 | Vượt trội hoàn toàn, F1 cao hơn cả Fine-tuning (+3.6%). |

**Nhận xét:**
- **Hiệu quả của Few-shot:** Cả hai mô hình mã nguồn mở (Vinallama và Qwen) đều cho thấy sự cải thiện khi chuyển từ Zero-shot sang Few-shot.
  - Vinallama F1 từ 33.48 lên 35.02.
  - Qwen tăng F1 từ 35.08 lên 38.01.
- **So sánh Open-source vs. Commercial:** GPT-4o-mini vượt trội hoàn toàn so với các mô hình nhỏ (2.7B, 4B) với F1-Score đạt 62.61, thậm chí cao hơn F1 của phương pháp Fine-tuning tốt nhất (XLM-R đạt 59.02).

## 3. Phân tích và Thảo luận

### 3.1. So sánh chiến lược huấn luyện

- **Fine-tuning (XLM-R):** Đạt độ chính xác tuyệt đối (EM) cao nhất trong tất cả các thí nghiệm (48.65). Điều này cho thấy việc huấn luyện chuyên sâu giúp mô hình học được chính xác ranh giới của câu trả lời (start/end tokens).
- **Prompting (GPT-4o-mini):** Mặc dù EM thấp hơn XLM-R (41.95 so với 48.65), nhưng F1-Score lại cao hơn (62.61 so với 59.02). Điều này ngụ ý rằng GPT-4o-mini tìm được nội dung câu trả lời đúng nhưng thường có xu hướng thừa hoặc thiếu một vài từ ngữ xung quanh so với đáp án chuẩn, dẫn đến EM thấp hơn nhưng độ chồng lắp (overlap) cao.

### 3.2. Về các mô hình nhỏ (Vinallama, Qwen)

Các mô hình kích thước nhỏ (dưới 4B tham số) gặp khó khăn trong việc tuân thủ chính xác định dạng trích xuất span. Điểm số thấp (F1 ~35-38%) cho thấy các mô hình này thường trả lời dài dòng hoặc sai lệch ngữ cảnh mặc dù đã được cung cấp ví dụ Few-shot.

## 4. Kết luận

Qua quá trình thực nghiệm, có thể rút ra các kết luận sau:

1. **Đối với bài toán yêu cầu độ chính xác tuyệt đối về vị trí trích xuất:** Fine-tuning XLM-RoBERTa là lựa chọn tốt nhất (EM 48.65).
2. **Nếu xét về khả năng hiểu ngữ nghĩa tổng quát và linh hoạt mà không cần huấn luyện lại:** GPT-4o-mini cho kết quả F1 ấn tượng nhất (62.61).
3. **Kỹ thuật Few-shot prompting** chứng minh được hiệu quả trong việc cải thiện độ chính xác cho các mô hình mã nguồn mở cỡ nhỏ, giúp định hướng mô hình trả về kết quả đúng định dạng hơn.
