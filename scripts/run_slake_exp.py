import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from src.perturbations import ImagePerturber
from src.dataset_slake import SlakeDataset
from src.evaluator import SlakeEvaluator

# HuatuoGPT 로더 (실제 서버에서 돌아갈 추론 함수 예시)
# 실제 모델 로드 로직은 모델 배포 문서에 따라 조금씩 다를 수 있습니다.
def run_experiment(config_path):
    # 1. 설정 로드
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. 부품 초기화
    perturber = ImagePerturber(
        lpf_sigma=cfg['perturbation']['lpf_sigma'],
        hpf_sigma=cfg['perturbation']['hpf_sigma'],
        patch_size=cfg['perturbation']['patch_size']
    )
    dataset = SlakeDataset(cfg['data']['json_path'], cfg['data']['img_dir'])
    evaluator = SlakeEvaluator()
    
    # 3. 모델 로드 (Vast.ai에서 실행 시 주석 해제)
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained(cfg['model']['name'], ...)
    
    results = []

    # 4. 루프 시작
    print(f"Starting experiment on {len(dataset)} samples...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        
        for cond in cfg['perturbation']['conditions']:
            # 이미지 변형 적용
            p_img = perturber.apply(sample['image'], cond)
            
            # 모델 추론 (로컬에선 "Test Answer"로 대체)
            # prediction = model.chat(p_img, sample['question'])
            prediction = "Test Answer" 
            
            # 채점
            is_correct = evaluator.evaluate(prediction, sample['answer'])
            
            # 결과 저장
            results.append({
                "img_id": sample['img_id'],
                "condition": cond,
                "q_type": sample['q_type'],
                "modality": sample['modality'],
                "question": sample['question'],
                "gt": sample['answer'],
                "pred": prediction,
                "correct": is_correct
            })

    # 5. 결과 저장
    os.makedirs(cfg['data']['output_dir'], exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg['data']['output_dir'], "final_results.csv"), index=False)
    print("Experiment Finished! Results saved to results/final_results.csv")

if __name__ == "__main__":
    run_experiment("configs/slake_config.yaml")