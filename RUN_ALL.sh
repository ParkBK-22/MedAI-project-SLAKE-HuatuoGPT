#!/bin/bash
# SLAKE HuatuoGPT 전체 실행 스크립트
# git clone부터 결과 분석까지 자동화

set -e  # 에러 발생 시 중단

echo "=========================================="
echo "🚀 SLAKE HuatuoGPT 완전 자동 실행"
echo "=========================================="
echo ""

# ==================== 1단계: 환경 준비 ====================
echo "[1/6] 환경 준비..."
echo ""
echo "# Git 클론 (처음 설정할 때만)"
echo "git clone https://github.com/YourUserName/MedAI-project-SLAKE-HuatuoGPT.git"
echo "cd MedAI-project-SLAKE-HuatuoGPT"
echo ""
echo "# Python 가상환경 생성"
echo "python -m venv venv"
echo "source venv/bin/activate  # Linux/Mac"
echo "venv\\Scripts\\activate  # Windows"
echo ""
echo "# 기본 패키지 업그레이드"
echo "pip install --upgrade pip setuptools wheel"
echo ""

# ==================== 2단계: 의존성 설치 ====================
echo "[2/6] 의존성 설치..."
echo ""
echo "# PyTorch + CUDA 12.1 설치"
echo "pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "# 최신 transformers (llava_qwen2 지원)"
echo "pip install git+https://github.com/huggingface/transformers.git --upgrade"
echo ""
echo "# 나머지 패키지"
echo "pip install -r requirements_huatuogpt.txt"
echo ""

# ==================== 3단계: 환경 검증 ====================
echo "[3/6] 환경 검증..."
echo ""
echo "python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')\""
echo ""

# ==================== 4단계: 프로젝트 설정 ====================
echo "[4/6] 프로젝트 설정..."
echo ""
echo "python scripts/setup_slake.py"
echo ""

# ==================== 5단계: 모델 검증 ====================
echo "[5/6] 모델 검증..."
echo ""
echo "python scripts/verify_models.py"
echo ""

# ==================== 6단계: 실험 실행 ====================
echo "[6/6] 실험 실행 (약 2-4시간 소요)..."
echo ""
echo "python scripts/run_slake_exp.py"
echo ""

# ==================== 7단계: 결과 분석 ====================
echo "[7/7] 결과 분석..."
echo ""
echo "python scripts/analyze_slake.py"
echo ""

echo "=========================================="
echo "✅ 완료!"
echo "=========================================="
echo ""
echo "결과 파일:"
echo "  - results/slake_results.csv"
echo "  - results/diagnostics.yaml"
echo "  - results/question_type_analysis.yaml"
