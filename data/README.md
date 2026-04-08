# SLAKE - Medical VQA 이미지 Perturbation 실험

## 1. 프로젝트 개요
의료 VQA 모델인 **HuatuoGPT-Vision-7B**가 실제 의료 영상의 시각적 특징을 활용하는지, 아니면 질문에 포함된 텍스트 지식(Prior)에 의존하는지를 **이미지 변형(Perturbation) 실험**으로 검증합니다.

**핵심 가설**: SLAKE는 PMC-VQA보다 정제된 데이터셋이므로, 이미지를 훼손(Black, Patch Shuffle)했을 때 **Location(위치)** 및 **Shape(모양)** 관련 질문에서 정답률이 유의미하게 하락해야 모델이 시각 정보를 올바르게 사용하고 있다고 볼 수 있습니다.

### 사용 모델
| 모델 | HF Repo | 특징 |
|------|---------|------|
| **HuatuoGPT-Vision-7B** | `FreedomIntelligence/HuatuoGPT-Vision-7B` | 의료 지식 특화 LLM 기반 VLM |

---

## 2. 실험 설계

### 기본 설정
| 항목 | 설정 |
|------|------|
| **데이터셋** | [SLAKE](https://github.com/vqdang/SLAKE_Dataset) (English subset) |
| **이미지 조건** | Original, Black, LPF, HPF, Patch Shuffle (16x16) |
| **평가 지표** | Exact Match (EM), Yes/No Accuracy |
| **디코딩** | Greedy (temperature=0) |

### 이미지 변형 조건 (Image Conditions)
| Condition | 설명 | 의도 |
|-----------|------|------|
| **Original** | 원본 의료 영상 | Baseline 성능 확인 |
| **Black** | 모든 픽셀 0 (검정) | **Textual Bias(이미지 없이 맞추는 비율) 측정** |
| **LPF** | Gaussian Blur (sigma=3) | 고주파 디테일(미세 병변) 제거 시 반응 |
| **HPF** | Edge preservation (sigma=25) | 저주파 구조(장기 형태) 제거 시 반응 |
| **Patch Shuffle** | 16x16 패치 위치 셔플 | **해부학적 공간 구조 파괴 시 성능 하락 확인** |

---

## 3. 진단 지표 (Diagnostic Metrics)

| Metric | Formula | 해석 |
|--------|---------|------|
| **VRS** (Vision Reliance Score) | EM(Original) - EM(Black) | 높을수록 시각 정보에 많이 의존함 |
| **L-Drop** (Location Drop) | Acc_loc(Original) - Acc_loc(Shuffle) | 위치 기반 질문에서 공간 정보 활용도 측정 |
| **K-Ratio** (Knowledge Ratio) | EM(Black) / EM(Original) | 모델이 지식만으로 답변을 때려 맞추는 경향성 |

---

## 4. 셋업 및 실행 가이드

### 1단계: 환경 구축 (Vast.ai)
```bash
# 레포 클론
git clone [https://github.com/GWB21/MedAI-project.git](https://github.com/GWB21/MedAI-project.git)
cd MedAI-project

# 패키지 설치
pip install -r requirements_base.txt