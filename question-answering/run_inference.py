import argparse
import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

# ==============================================================================
# PHẦN 1: CẤU HÌNH PROMPT VÀ VÍ DỤ MẪU
# ==============================================================================

# Lời dẫn (Instruction) chung cho cả 2 chế độ.
# Đây là phần quan trọng nhất để định hướng model tuân thủ luật chơi.
SYSTEM_INSTRUCTION = """Bạn là một hệ thống Đọc hiểu máy (Machine Reading Comprehension). Nhiệm vụ của bạn là trích xuất câu trả lời từ đoạn văn bản cho trước.

Quy tắc bắt buộc:
1. Câu trả lời phải là một đoạn văn bản (span) được lấy NGUYÊN VĂN từ "Đoạn văn". Không được viết lại hay thay đổi từ ngữ. Đặc biệt là các tên riêng hay thuật ngữ, số liệu, thời gian phải giữ nguyên.
2. Nếu thông tin không có trong "Đoạn văn", hãy trả về chuỗi rỗng "" (không viết gì cả).
3. Câu trả lời phải ngắn gọn và chính xác nhất.
4. Câu hỏi có thể hỏi ngoài lề, tuy có một số thông tin trong câu hỏi có thể liên quan đến đoạn văn nhưng KHÔNG PHẢI LÀ CÂU TRẢ LỜI. Hãy cẩn thận để không bị nhầm lẫn.
"""

# Template cho từng ví dụ trong Few-shot
# (Model sẽ học cách ánh xạ từ Context + Question -> Answer dựa trên mẫu này)
EXAMPLE_TEMPLATE = "Đoạn văn: {context}\nCâu hỏi: {question}\nTrả lời: {answer_text}\n\n"

# Template cho câu hỏi thực tế cần dự đoán (Target)
TARGET_TEMPLATE = "Đoạn văn: {context}\nCâu hỏi: {question}\nTrả lời:"

# Có cả ví dụ trả lời được VÀ ví dụ không trả lời được (để dạy model trả về rỗng)
FEW_SHOT_EXAMPLES_DATA = [
    {
        "context": "Sau khi Mao Trạch Đông từ trần 1976 và vụ bắt giữ bè phái mang tên Tứ nhân bang, Đặng Tiểu Bình lên nắm quyền và lãnh đạo quốc gia đến cải cách kinh tế quan trọng. Đảng Cộng sản sau đó nới lỏng kiểm soát của chính phủ đối với đời sống cá nhân của công dân và các công xã nhân dân bị bãi bỏ nhằm tạo điều kiện cho kinh tế tư nhân. Sự kiện này đánh dấu Trung Quốc chuyển đổi từ kinh tế kế hoạch sang kinh tế hỗn hợp, kinh tế tư nhân phát triển nhưng Nhà nước vẫn nắm giữ quyền điều tiết thị trường và các lĩnh vực quan trọng, với sự gia tăng của môi trường thị trường mở. Trung Quốc thông qua hiến pháp hiện hành vào ngày 4 tháng 1 năm 1982. Năm 1989, hành động trấn áp bạo lực các cuộc biểu tình của sinh viên tại quảng trường Thiên An Môn khiến chính phủ Trung Quốc bị nhiều quốc gia chỉ trích và áp đặt chế tài.",
        "question": "Rất nhiều chính phủ các nước đã chỉ trích Trung Quốc về sự kiện gì vào năm 1989?",
        "answer_text": "hành động trấn áp bạo lực các cuộc biểu tình của sinh viên tại quảng trường Thiên An Môn"
    },
    {
        "context": "Năm 1762, Tây Ban Nha xâm chiếm lãnh thổ Bồ Đào Nha trong khuôn khổ Chiến tranh Bảy Năm, song đến năm 1763 Tây Ban Nha và Bồ Đào Nha khôi phục hiện trạng trước chiến tranh. Sau vụ án Távora, Sebastião de Melo không còn thế lực đối lập, ông cai trị Bồ Đào Nha trên thực tế cho đến khi José I mất vào năm 1779. Tuy nhiên, các sử gia cũng lập luận rằng 'khai sáng' của Pombal dù có ảnh hưởng sâu rộng song chủ yếu là một kỹ xảo giúp nâng cao chế độ chuyên quyền gây tổn hại cho tự do cá nhân và đặc biệt là một công cụ để nghiền nát phe đối lập, đàn áp chỉ trích, và đẩy mạnh khai thác kinh tế thuộc địa cũng như tăng cường kiểm duyệt sách và củng cố quyền kiểm soát và lợi ích cá nhân.",
        "question": "Đến năm bao nhiêu thì Pombal đã khôi phục hiện trạng trước chiến tranh?",
        "answer_text": "" 
    },
    {
        "context": "Mặc dù nằm ở phía Nam so với chí tuyến Bắc, Quảng Châu có khí hậu ôn đới ẩm ướt (Köppen Cfa) chịu ảnh hưởng của gió mùa Đông Á. Mùa hè ẩm ướt với nhiệt độ cao, độ ẩm cao và chỉ số nhiệt cao. Mùa đông khá ôn hòa và tương đối khô. Quảng Châu có mùa mưa kéo dài, kéo dài từ tháng Tư đến tháng Chín. Nhiệt độ mức trung bình hàng tháng dao động từ 13,6 °C (56,5 °F) vào tháng Giêng đến 28,6 °C (83,5 °F) vào tháng 7, trong khi trung bình năm là 22,6 °C (72,7 °F). Mùa thu, từ tháng 10 đến tháng 12, có khí hậu rất dễ chịu, mát mẻ và nhiều gió, và là thời gian đi du lịch thành phố tốt nhất. Độ ẩm tương đối khoảng 68 %, trong khi lượng mưa hàng năm trong khu vực đô thị là hơn 1.700 mm (67 inch). Với mức ánh sáng mặt trời hàng tháng có thể dao động từ 17% trong tháng 3 và tháng 4 đến 52% vào tháng 11, thành phố này nhận được 1,628 giờ ánh nắng mặt trời hàng năm, ít hơn đáng kể so với Thâm Quyến và Hồng Kông. Nhiệt độ cực đại dao động từ 0 °C (32 °F) đến 39,1 °C (102,4 °F). Lần tuyết rơi cuối cùng được ghi lại trong thành phố là vào ngày 24 tháng 1 năm 2016, đó là lần đầu tiên thành phố có tuyết sau 87 năm.",
        "question": "Quảng Châu có tuyết rơi vào tháng 1 năm 2016 là sự kết thúc cho chuỗi thời gian bao lâu mà tuyết đã không ghé thăm Quảng Châu?",
        "answer_text": "87 năm"
    }, 
    {
        "context": "Năm 1762, Tây Ban Nha xâm chiếm lãnh thổ Bồ Đào Nha trong khuôn khổ Chiến tranh Bảy Năm, song đến năm 1763 Tây Ban Nha và Bồ Đào Nha khôi phục hiện trạng trước chiến tranh. Sau vụ án Távora, Sebastião de Melo không còn thế lực đối lập, ông cai trị Bồ Đào Nha trên thực tế cho đến khi José I mất vào năm 1779. Tuy nhiên, các sử gia cũng lập luận rằng 'khai sáng' của Pombal dù có ảnh hưởng sâu rộng song chủ yếu là một kỹ xảo giúp nâng cao chế độ chuyên quyền gây tổn hại cho tự do cá nhân và đặc biệt là một công cụ để nghiền nát phe đối lập, đàn áp chỉ trích, và đẩy mạnh khai thác kinh tế thuộc địa cũng như tăng cường kiểm duyệt sách và củng cố quyền kiểm soát và lợi ích cá nhân.",
        "question": "Đến năm bao nhiêu thì Pombal đã khôi phục hiện trạng trước chiến tranh?",
        "answer_text": ""
    }
]

def build_prompt(mode, context, question, num_shots=3):
    """
    Hàm dựng prompt hoàn chỉnh dựa trên chế độ.
    Luôn bắt đầu bằng SYSTEM_INSTRUCTION.
    """
    # 1. Bắt đầu bằng lời dẫn quyền lực
    full_prompt = SYSTEM_INSTRUCTION

    # 2. Nếu là Few-shot, nhồi thêm ví dụ vào giữa
    if mode == "few-shot":
        # Lấy số lượng ví dụ tối đa có thể
        real_shots = min(num_shots, len(FEW_SHOT_EXAMPLES_DATA))
        
        for i in range(real_shots):
            ex = FEW_SHOT_EXAMPLES_DATA[i]
            # Ghép ví dụ theo format
            full_prompt += EXAMPLE_TEMPLATE.format(
                context=ex['context'], 
                question=ex['question'], 
                answer_text=ex['answer_text']
            )
    
    # 3. Kết thúc bằng câu hỏi thực tế cần trả lời
    full_prompt += TARGET_TEMPLATE.format(context=context, question=question)
    
    return full_prompt

# ==============================================================================
# PHẦN 2: LOGIC CHẠY INFERENCE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Script chạy Inference QA cho UIT-ViQuAD")
    
    # Các tham số file và model
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Tên model hoặc đường dẫn checkpoint")
    parser.add_argument("--test_file", type=str, required=True, help="File input JSON (format SQuAD/ViQuAD)")
    parser.add_argument("--output_file", type=str, default="predictions.json", help="File output JSON kết quả")
    
    # Các tham số cấu hình Prompting
    parser.add_argument("--mode", type=str, choices=["zero-shot", "few-shot"], default="zero-shot", help="Chế độ chạy")
    parser.add_argument("--num_shots", type=int, default=4, help="Số lượng ví dụ mẫu (chỉ dùng cho few-shot)")
    
    # Các tham số kỹ thuật
    parser.add_argument("--max_length", type=int, default=1024, help="Độ dài tối đa của Prompt đầu vào")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Độ dài tối đa câu trả lời sinh ra")
    parser.add_argument("--batch_size", type=int, default=8, help="Số lượng câu hỏi xử lý cùng lúc")
    parser.add_argument("--cache_dir", type=str, default=None, help="Thư mục cache model (tùy chọn)")
    parser.add_argument("--auth_token", type=str, default=None, help="HF token nếu model yêu cầu quyền truy cập")
    parser.add_argument("--trust_remote_code", action="store_true", help="Cho phép load code tùy chỉnh của model")
    parser.add_argument("--multi_gpu", action="store_true", help="Sử dụng DataParallel cho nhiều GPU")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"\n{'='*40}")
    print(f"CẤU HÌNH CHẠY: {args.mode.upper()}")
    print(f"Model: {args.model_name}")
    print(f"Input: {args.test_file}")
    print(f"Batch size: {args.batch_size}")
    if args.mode == "few-shot":
        print(f"Số lượng shots: {min(args.num_shots, len(FEW_SHOT_EXAMPLES_DATA))}")
    print(f"{'='*40}\n")

    # 1. Load Model
    print("Đang tải model...")
    
    # Tự động detect loại model (Seq2Seq hay CausalLM)
    config = AutoConfig.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        token=args.auth_token,
        trust_remote_code=args.trust_remote_code,
    )
    is_encoder_decoder = getattr(config, 'is_encoder_decoder', False)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        token=args.auth_token,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Đảm bảo tokenizer có pad token (cần cho batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Chuẩn bị device_map cho multi-GPU
    device_map = "auto" if args.multi_gpu and torch.cuda.device_count() > 1 else None
    
    if is_encoder_decoder:
        print(f"  → Loại model: Encoder-Decoder (Seq2Seq)")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            token=args.auth_token,
            trust_remote_code=args.trust_remote_code,
            device_map=device_map,
        )
        if device_map is None:
            model = model.to(args.device)
    else:
        print(f"  → Loại model: Decoder-only (CausalLM)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            token=args.auth_token,
            trust_remote_code=args.trust_remote_code,
            device_map=device_map,
        )
        if device_map is None:
            model = model.to(args.device)
    
    model.eval()
    
    # Hiển thị thông tin GPU
    if device_map == "auto":
        print(f"  → Sử dụng {torch.cuda.device_count()} GPU với device_map='auto'")
        print(f"  → Model đã được tự động chia ra các GPU")
    elif torch.cuda.is_available():
        print(f"  → Sử dụng 1 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  → Sử dụng CPU")

    # 2. Load Data
    print(f"Đang đọc dữ liệu từ {args.test_file}...")
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Không tìm thấy file: {args.test_file}")
    
    with open(args.test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = {}

    def normalize_paragraphs(raw_data):
        # Chuẩn hóa mọi format có thể: SQuAD (paragraphs), đơn lẻ context/qas,
        # hoặc mỗi item chỉ chứa context-question-id (private test)
        articles = raw_data.get('data', raw_data) if isinstance(raw_data, dict) else raw_data
        if not isinstance(articles, list):
            raise ValueError("Định dạng file không hợp lệ: data phải là list hoặc có key 'data'")

        paragraphs = []
        for art in articles:
            if {'context', 'question', 'id'}.issubset(art.keys()):
                qas_answers = art.get('answers', {'answer_start': [], 'text': []})
                paragraphs.append({
                    'context': art['context'],
                    'qas': [{
                        'question': art['question'],
                        'id': art['id'],
                        'answers': qas_answers
                    }]
                })
            else:
                raise KeyError(f"Article không hợp lệ, thiếu keys cần thiết. Keys có: {list(art.keys())}")
        return paragraphs

    paragraphs = normalize_paragraphs(data)
    
    # Tính tổng số câu hỏi để hiện thanh loading
    total_qas = sum(len(p['qas']) for p in paragraphs)
    
    # 3. Chuẩn bị tất cả samples
    all_samples = []
    for paragraph in paragraphs:
        context = paragraph['context']
        for qa in paragraph['qas']:
            all_samples.append({
                'id': qa['id'],
                'context': context,
                'question': qa['question']
            })
    
    # 4. Inference Loop với Batching
    print("Bắt đầu dự đoán...")
    with tqdm(total=len(all_samples), desc="Processing") as pbar:
        for i in range(0, len(all_samples), args.batch_size):
            batch_samples = all_samples[i:i + args.batch_size]
            
            # Tạo prompts cho cả batch
            batch_prompts = [
                build_prompt(
                    mode=args.mode,
                    context=sample['context'],
                    question=sample['question'],
                    num_shots=args.num_shots
                )
                for sample in batch_samples
            ]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
                padding=True
            ).to(args.device)
            
            # Generate batch (greedy decoding - nhanh hơn beam search)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("\n")]
                )
            
            # Decode batch
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # Lưu kết quả
            for sample, answer in zip(batch_samples, answers):
                predictions[sample['id']] = answer
            
            pbar.update(len(batch_samples))

    # 4. Save Results
    print(f"\nĐã xong! Lưu kết quả vào {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    main()