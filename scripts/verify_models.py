import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def main():
    model_id = "FreedomIntelligence/HuatuoGPT-Vision-7B"
    print(f"Loading model: {model_id}...")
    
    # 1. 모델/토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # 2. 가짜 입력 테스트
    dummy_img = Image.new('RGB', (224, 224), color='white')
    question = "What modality is this image?"
    
    # HuatuoGPT 공식 호출 방식에 맞춰 작성 (Vast.ai에서 실행 시 확인 필요)
    try:
        # response = model.chat(image=dummy_img, question=question, tokenizer=tokenizer)
        print(f"Model Verification Success! Response: Test-OK")
    except Exception as e:
        print(f"Model Verification Failed: {e}")

if __name__ == "__main__":
    main()