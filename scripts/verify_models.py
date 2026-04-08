import torch
from PIL import Image
import warnings
import os

# Transformer 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    model_id = "FreedomIntelligence/HuatuoGPT-Vision-7B"
    print(f"🚀 Loading model: {model_id}...")
    
    try:
        # HuatuoGPT 공식 커스텀 로더 사용
        from transformers import AutoProcessor, CLIPImageProcessor
        
        print("📥 Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        
        print("📥 Loading model (this may take a few minutes on first run)...")
        # 커스텀 모델 클래스를 직접 불러오기
        model = torch.hub.load(
            'FreedomIntelligence/HuatuoGPT-Vision-7B',
            'huatuo_vision_model',
            pretrained=True,
            trust_repo_owner=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        if model is None:
            # torch.hub 실패 시 fallback
            print("ℹ️  Using transformers fallback...")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        model.eval()
        print("✅ Model & Processor loaded successfully!")

        # 2. 실제 추론 테스트 (Sanity Check)
        print("🧪 Running a quick inference test...")
        
        # 가짜 입력 생성 (흰색 배경 이미지)
        dummy_img = Image.new('RGB', (224, 224), color='white')
        question = "What is shown in this image?"
        
        # HuatuoGPT 전용 프롬프트 템플릿 적용
        prompt = f"<|user|>\n<image>\n{question}</s>\n<|assistant|>\n"
        
        # 데이터 준비 (GPU 이동)
        inputs = processor(text=prompt, images=dummy_img, return_tensors="pt").to("cuda", torch.float16)

        # 답변 생성
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True
            )

        # 결과 디코딩
        input_token_len = inputs.input_ids.shape[1]
        response = processor.decode(output_ids[0][input_token_len:], skip_special_tokens=True).strip()
        
        print(f"✨ Model Verification Success!")
        print(f"📝 Model Response: {response}")

    except Exception as e:
        print(f"❌ Model Verification Failed: {e}")
        print("\n💡 Tip: 'pip install --upgrade transformers'를 했는지 다시 확인해보세요.")

if __name__ == "__main__":
    main()