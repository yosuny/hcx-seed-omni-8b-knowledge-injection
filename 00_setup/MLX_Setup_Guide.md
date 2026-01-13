# MLX Environment Setup Guide

이 가이드는 Apple Silicon (M1/M2/M3 등)이 탑재된 Mac에서 이 프로젝트를 실행하기 위한 환경 설정 방법을 안내합니다.

## 1. 사전 요구 사항 (Prerequisites)

- **Hardware**: Apple Silicon (M1/M2/M3 칩셋)이 탑재된 Mac
- **OS**: macOS 13.3 (Ventura) 이상 권장
- **Python**: Python 3.9 이상

## 2. 가상 환경 생성 및 활성화

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 격리된 Python 환경을 생성합니다.

```bash
# 가상 환경(.venv) 생성
python3 -m venv .venv

# 가상 환경 활성화
source .venv/bin/activate
```

> **Note**: 가상 환경을 활성화하면 터미널 프롬프트 앞에 `(.venv)`가 표시됩니다.

## 3. 필수 패키지 설치

`mlx-lm`, `transformers` 등 프로젝트 실행에 필요한 라이브러리를 설치합니다.

```bash
pip install -U pip
pip install mlx-lm transformers huggingface_hub pyyaml
```

- **mlx-lm**: Apple Silicon용 LLM 학습/추론 라이브러리
- **transformers**: 모델 및 토크나이저 로딩
- **huggingface_hub**: Hugging Face 모델 다운로드 및 업로드
- **pyyaml**: 설정 파일(`config.yaml` 등) 파싱

## 4. Hugging Face 인증 설정 (권장)

Gemma-2-9b-it 등 게이트(Gated) 모델이나 비공개 모델에 접근하려면 Hugging Face 토큰이 필요합니다.

1. [Hugging Face Settings](https://huggingface.co/settings/tokens)에서 Access Token 발급 (Read 권한).
2. 터미널에서 환경 변수로 설정:

```bash
export HUGGINGFACE_HUB_TOKEN="your_hf_token_here"
```

또는 CLI 로그인을 통해 영구 저장할 수도 있습니다:

```bash
huggingface-cli login
```

## 5. 설치 검증

설치가 올바르게 되었는지 확인하기 위해 다음 Python 코드를 실행해 봅니다.

```bash
python -c "import mlx.core as mx; print(f'MLX Version: {mx.__version__}'); print(f'GPU Available: {mx.metal.is_available()}')"
```

**정상 출력 예시**:
```text
MLX Version: 0.21.1 (버전은 다를 수 있음)
GPU Available: True
```

## 6. 문제 해결 (Troubleshooting)

- **Memory Warning**: 32B 모델 학습 시 48GB 이상의 통합 메모리(RAM)가 권장됩니다. 메모리가 부족한 경우 `batch-size`를 줄이거나 다른 애플리케이션을 종료하세요.
- **Op Error**: 오래된 macOS 버전을 사용하는 경우 특정 MLX 연산이 실패할 수 있습니다. macOS를 최신 버전으로 업데이트하세요.

---

이제 프로젝트의 스크립트(예: `train_solverx_cpt_hcx.sh`)를 실행할 준비가 되었습니다.
