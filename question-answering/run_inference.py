import argparse
import json
import torch
import os
import re # Thêm thư viện này để xử lý chuỗi tốt hơn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

# ==============================================================================
# PHẦN 1: CẤU HÌNH PROMPT (GIỮ NGUYÊN NỘI DUNG CỦA USER)
# ==============================================================================

# Lời dẫn (Instruction) chung.
SYSTEM_INSTRUCTION = """Bạn là một hệ thống Đọc hiểu máy (Machine Reading Comprehension). Nhiệm vụ của bạn là trích xuất câu trả lời từ đoạn văn bản cho trước.

Quy tắc bắt buộc:
1. Câu trả lời phải là một đoạn văn bản (span) được lấy NGUYÊN VĂN từ "Đoạn văn". Không được viết lại, thay đổi từ ngữ, hay thêm từ ngữ. Đặc biệt là các tên riêng hay thuật ngữ, số liệu, thời gian phải giữ nguyên.
2. Nếu thông tin không có trong "Đoạn văn", hãy trả về chuỗi rỗng "" (không viết gì cả).
3. Câu trả lời phải ngắn gọn và chính xác nhất được hiển thị trong đoạn văn về format và độ dài. Không cần lặp lại các từ trong câu hỏi để mở đầu câu trả lời mà hãy trả lời thẳng đáp án từ đoạn văn nếu có. Không thêm bất kỳ từ ngữ nào khác.
4. Câu hỏi có thể hỏi ngoài lề, tuy có một số thông tin trong câu hỏi có thể liên quan đến đoạn văn nhưng KHÔNG PHẢI LÀ CÂU TRẢ LỜI. Hãy cẩn thận."""

# Dữ liệu Few-shot mẫu
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

# ==============================================================================
# PHẦN 2: CÁC HÀM HỖ TRỢ XỬ LÝ PROMPT
# ==============================================================================

def build_prompt(mode, context, question, model_name, num_shots=3):
    """
    Tạo prompt theo đúng chuẩn ChatML (Chat Markup Language)
    như yêu cầu trong ảnh: <|im_start|>system...<|im_start|>user...<|im_start|>assistant
    """
    # 1. Xây dựng nội dung cốt lõi (Core content)
    core_text = ""
    
    # Nếu là few-shot, thêm ví dụ vào trước
    if mode == 'few-shot':
        examples = FEW_SHOT_EXAMPLES_DATA[:num_shots]
        for ex in examples:
            core_text += f"Đoạn văn: {ex['context']}\nCâu hỏi: {ex['question']}\nTrả lời: {ex['answer_text']}\n\n"
            
    # Thêm câu hỏi hiện tại (Target)
    # Lưu ý: "Trả lời:" ở cuối user block có thể giữ lại để định hướng model
    target_text = f"Đoạn văn: {context}\nCâu hỏi: {question}\nTrả lời:"
    user_content = core_text + target_text

    # 2. Đóng gói theo format ChatML (BẮT BUỘC)
    # Lưu ý: Không thêm khoảng trắng thừa giữa các thẻ
    prompt = (
        f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

# ==============================================================================
# PHẦN 3: MAIN INFERENCE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--mode", type=str, default="zero-shot", choices=["zero-shot", "few-shot"])
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=2048) # Tăng lên chút để an toàn cho prompt dài
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--auth_token", type=str, default=None)
    parser.add_argument("--multi_gpu", action="store_true")
    args = parser.parse_args()

    # 1. Load Tokenizer & Config
    print(f"Loading tokenizer: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            token=args.auth_token,
            padding_side='left' 
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Lỗi load tokenizer: {e}")
        return

    # 2. Load Model
    print(f"Loading model: {args.model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=args.auth_token,
            torch_dtype=torch.float16,
            device_map="auto" if args.multi_gpu else None
        )
        if not args.multi_gpu:
            model.to(args.device)
        model.eval()
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return

    # 3. Load Data
    print(f"Loading data from: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    all_samples = []

    # Xử lý logic đa định dạng (Adaptive Data Loading)
    if isinstance(raw_data, dict) and 'data' in raw_data:
        data_list = raw_data['data']
        if len(data_list) > 0:
            first_item = data_list[0]
            if 'paragraphs' in first_item: # Format SQuAD
                for article in data_list:
                    for paragraph in article['paragraphs']:
                        context = paragraph['context']
                        for qa in paragraph['qas']:
                            all_samples.append({
                                "id": qa['id'], "context": context, "question": qa['question']
                            })
            else: # Format phẳng
                for item in data_list:
                    all_samples.append({
                        "id": item['id'], "context": item['context'], "question": item['question']
                    })
    elif isinstance(raw_data, list):
        all_samples = raw_data

    print(f"Đã load {len(all_samples)} mẫu dữ liệu.")

    # Biến lưu kết quả
    predictions = {}

    # 4. Inference Loop
    print(f"Bắt đầu dự đoán ({args.mode})...")
    with tqdm(total=len(all_samples)) as pbar:
        for i in range(0, len(all_samples), args.batch_size):
            batch_samples = all_samples[i:i + args.batch_size]
            
            # Tạo prompt cho batch
            batch_prompts = [
                build_prompt(
                    mode=args.mode,
                    context=s['context'],
                    question=s['question'],
                    model_name=args.model_name,
                    num_shots=args.num_shots
                )
                for s in batch_samples
            ]
            
            # Tokenize
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
                padding=True
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode (Skip special tokens để loại bỏ <|im_...|>)
            decoded_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for sample, raw_text in zip(batch_samples, decoded_sequences):
                # LOGIC LÀM SẠCH KẾT QUẢ
                
                # 1. Tách phần prompt
                if "assistant" in raw_text:
                    clean_answer = raw_text.rpartition("assistant")[2] 
                elif "Trả lời:" in raw_text:
                    clean_answer = raw_text.rpartition("Trả lời:")[2]
                else:
                    clean_answer = raw_text

                # 2. Xử lý xuống dòng & khoảng trắng
                clean_answer = clean_answer.strip()
                if '\n' in clean_answer:
                    clean_answer = clean_answer.split('\n')[0]
                
                # 3. FIX LỖI "SYSTEM": Nếu kết quả chỉ còn trơ trọi chữ "system" -> Coi như rỗng
                # (Vì model lỡ sinh ra thẻ <|im_start|>system)
                if clean_answer.lower().strip() == "system":
                    clean_answer = ""

                # 4. Loại bỏ các prefix rác khác
                remove_prefixes = ["Câu trả lời là:", "Answer:", "Đáp án:", ":"]
                for prefix in remove_prefixes:
                    if clean_answer.lower().startswith(prefix.lower()):
                        clean_answer = clean_answer[len(prefix):].strip()

                # Lưu kết quả
                predictions[sample['id']] = clean_answer
                
                # In ra để kiểm tra
                # if clean_answer: 
                print(f"ID: {sample['id']} | Ans : {clean_answer}")
                
            pbar.update(len(batch_samples))

    # 5. Lưu kết quả
    print(f"\nLưu file: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()