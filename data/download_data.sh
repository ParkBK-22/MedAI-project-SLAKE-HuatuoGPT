#!/bin/bash

mkdir -p data/slake
cd data/slake

echo "Downloading SLAKE files including Knowledge Graph..."

BASE_URL="https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main"

# 1. 질문지 3종
wget ${BASE_URL}/test.json
wget ${BASE_URL}/train.json
wget ${BASE_URL}/validation.json

# 2. 이미지 데이터 (필수)
wget ${BASE_URL}/imgs.zip

# 3. 지식 그래프 (연구의 깊이를 위해 추가!)
wget ${BASE_URL}/KG.zip

# 4. 압축 해제
echo "Extracting files..."
unzip -q imgs.zip
unzip -q KG.zip -d ./kg_info  # KG는 따로 폴더를 만들어 주는 게 관리하기 편합니다.
rm imgs.zip KG.zip

echo "Setup complete with Knowledge Graph!"
ls -R