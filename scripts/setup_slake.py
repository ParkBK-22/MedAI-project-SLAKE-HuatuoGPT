import os
import subprocess
import sys

def run_command(command):
    """터미널 명령어를 실행하고 결과를 출력합니다."""
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing: {command}")
        print(e)
        sys.exit(1)

def main():
    print("=== Starting SLAKE Project Setup ===")

    # 1. 필수 폴더 생성
    folders = ['data', 'results', 'checkpoints', 'results/viz']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    # 2. 데이터 다운로드 스크립트 실행 권한 부여 및 실행
    # (data/download_data.sh가 이미 존재한다고 가정)
    if os.path.exists('data/download_data.sh'):
        print("\n--- Downloading Dataset ---")
        run_command("chmod +x data/download_data.sh")
        run_command("./data/download_data.sh")
    else:
        print("Warning: data/download_data.sh not found. Skipping data download.")

    # 3. 환경 검증 (PyTorch 및 CUDA)
    print("\n--- Verifying Environment ---")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch is not installed. Please run 'pip install -r requirements_base.txt' first.")

    # 4. .gitkeep 생성 (빈 폴더 유지를 위해)
    with open('results/.gitkeep', 'w') as f:
        pass

    print("\n=== Setup Complete! You are ready to run experiments. ===")

if __name__ == "__main__":
    main()