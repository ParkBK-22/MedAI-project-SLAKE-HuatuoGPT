# SLAKE - Medical VQA 이미지 Perturbation 실험

## 1. 프로젝트 개요

의료 VQA 모델인 **HuatuoGPT-Vision-7B**가 정제된 의료 데이터셋(SLAKE)에서 실제로 시각적 특징을 추론하는지, 아니면 모델 내부의 의학적 지식(Prior)에 의존해 답변을 생성하는지를 검증합니다.

**핵심 아이디어**: SLAKE는 PMC-VQA보다 정제되어 있어 시각 정보의 중요도가 높습니다. 이미지를 훼손했을 때 **Location(위치)**이나 **Shape(모양)** 관련 질문에서 정답률이 급락하는지를 통해 모델의 'Visual Grounding' 능력을 측정합니다.

### 사용 모델 (HuatuoGPT 중심)

| **모델** | **HF Repo** | **특징** |
| --- | --- | --- |
| **HuatuoGPT-Vision-7B** | `FreedomIntelligence/HuatuoGPT-Vision-7B` | 의학 지식 그래프와 정렬된 의료 특화 VLM |

### 답변 결정 및 평가 방식 (Open-ended)

- **Prompt Engineering**: 모델이 짧고 명확한 답을 내도록 `Short Answer` 프롬프트를 강제합니다.
- **Evaluation**:
    - **Closed (Yes/No)**: 모델 출력 내 'yes/no' 포함 여부로 Accuracy 측정.
    - **Open (Short Answer)**: Ground Truth 단어가 모델 출력에 포함되었는지 확인하는 **Exact Match (EM)** 방식 사용.

---

## 2. 실험 설계

### 기본 설정

| **항목** | **설정** |
| --- | --- |
| **데이터셋** | [SLAKE](https://www.google.com/search?q=https://github.com/vqdang/SLAKE_Dataset) (English subset, ~14,000 samples) |
| **모델** | HuatuoGPT-Vision-7B (기존 LLaVA, MedVInT 병행 가능) |
| **질문 유형 분석** | Vision-Essential (Location, Shape) vs Knowledge-Based (Modality, Organ) |
| **이미지 조건** | Original, Black, LPF, HPF, Patch Shuffle (16x16) |
| **디코딩** | Greedy (temperature=0, do_sample=False) |

### 이미지 조건 (Image Conditions)

*기존 PMC-VQA 실험과 동일한 규격을 유지하여 비교 가능성을 확보합니다.*

| **Condition** | **설명** | **의도** |
| --- | --- | --- |
| **Original** | 원본 이미지 (CT, MRI, X-ray) | Baseline |
| **Black** | All Pixels 0 | 시각 정보 차단 시 '의학적 지식만으로 찍기' 성능 측정 |
| **LPF** | Gaussian Blur (sigma=3) | 저주파 정보(전체 장기 윤곽)의 중요도 확인 |
| **HPF** | Edge preservation (sigma=25) | 고주파 정보(병변의 미세 텍스처) 의존도 확인 |
| **Patch Shuffle** | 16x16 Patch Shuffle | **해부학적 구조 파괴 시 성능 변화 측정** |

---

## 3. SLAKE 전용 진단 지표 (Diagnostic Metrics)

| **Metric** | **Formula** | **해석** |
| --- | --- | --- |
| **VRS** (Vision Reliance Score) | EM(Original) - EM(Black) | 모델이 이미지를 보고 답하는 정도 (0에 가까우면 텍스트 편향) |
| **L-Drop** (Location Drop) | Acc_loc(Original) - Acc_loc(Shuffle) | 위치 기반 질문에서 공간 정보를 얼마나 중요하게 여기는지 측정 |
| **K-Ratio** (Knowledge Ratio) | EM(Black) / EM(Original) | 모델이 시각 정보 없이 지식으로만 맞추는 비율 (높을수록 환각 위험) |

---

## 4. 셋업 및 실행 가이드

### HuatuoGPT-Vision 전용 설정

Bash

`# 1. SLAKE 데이터셋 다운로드 및 구조화
python scripts/setup_slake.py --lang en

# 2. HuatuoGPT 환경 확인 (transformers 4.40.0)
pip install -r requirements_huatuogpt.txt

# 3. 실험 실행 (Short Answer 프롬프트 적용)
python scripts/run_slake_exp.py --model huatuogpt --condition all --prompt_type short`

### 프롬프트 템플릿 (src/dataset.py)

Python

`# HuatuoGPT용 SLAKE 프롬프트
SYSTEM_PROMPT = "You are a professional radiologist. Answer the question in a single word or a short phrase based on the image."
USER_PROMPT = f"Question: {question}\nAnswer:"`

---

## 5. 결과 분석 포인트 (SLAKE 특화)

SLAKE 데이터셋의 `q_type` 필드를 활용하여 다음과 같이 분석합니다.

1. **Location vs Organ**:
    - `Location` 질문(예: "Where is the lesion?")은 Patch Shuffle에서 성능이 크게 떨어져야 정상입니다.
    - `Organ` 질문(예: "What is this organ?")은 Black 이미지에서도 성능이 어느 정도 유지될 수 있습니다 (지식 편향).
2. **Modality Recognition**:
    - 모델이 CT와 MRI를 구분할 때 어떤 필터(LPF vs HPF)에 더 민감한지 분석하여 모델의 시각 인코더 특성을 파악합니다.
3. **Knowledge Graph Correlation**:
    - 지식 그래프에 정의된 '흔한 질병' 관련 질문에서 Black 이미지 성능이 유독 높게 나오는지(암기 여부) 확인합니다.

---



### 💡 HuatuoGPT 담당자 주의사항

- **Max New Tokens**: 단답형이므로 `max_new_tokens=20` 정도로 제한하여 불필요한 설명을 방지하고 추론 속도를 높이세요.
- **Parsing**: 모델이 "Yes, it is."라고 답할 경우 "Yes"로 인식할 수 있도록 `strip()` 및 소문자 변환 처리를 철저히 해야 합니다.
