# HyperCLOVA X Seed (Omni 8B) 지식 주입 프로젝트

이 프로젝트는 Apple Silicon (M 시리즈 칩) 환경에서 LLM(대규모 언어 모델)을 특정 도메인에 맞게 조정(Adaptation)하는 전체 수명 주기를 시뮬레이션합니다. **HyperCLOVA X Seed Omni 8B**를 베이스 모델로 사용하며, MLX-LM을 활용하여 **CLaRa** 논문(Continuous Latent Reasoning)의 지식을 주입했습니다.

## 프로젝트 개요

이 프로젝트의 목표는 모델이 **"CLaRa" 프레임워크**와 **"SCP" (Salient Compressor Pretraining)** 등 전문적인 질문에 대해, 일반적인 약어(예: SCP를 보안 프로토콜로 오인)로 환각(Hallucination)하지 않고 정확하게 답변하도록 만드는 것입니다.

**핵심 방법론:**
1.  **4-bit 양자화 (Quantization)**: 모델 크기를 약 5.1GB로 압축하여 로컬 하드웨어(Mac)에서 효율적으로 실행.
2.  **CPT (연속 사전 학습)**: 논문 원문(`text chunks`)을 학습하여 모델에 지식 베이스 구축.
3.  **SFT (지도 미세 조정) + Golden Data**: 합성된 Q&A 데이터와 수동으로 정제된 "Golden Data"를 사용하여, 모델이 지식을 올바르게 인출하고 환각을 교정하도록 훈련.

## 디렉토리 구조

| 디렉토리 | 설명 |
| :--- | :--- |
| `models/` | 4-bit 양자화된 베이스 모델 (`HyperCLOVAX-SEED-Omni-8B-Text-4bit`) 포함. |
| `adapters_omni_8b_paper_cpt/` | 1단계(CPT)에서 학습된 LoRA 어댑터. 지식 가중치 포함. |
| `adapters_omni_8b_paper_sft/` | **최종 2단계(SFT) LoRA 어댑터**. 추론 시 이것을 사용하세요. |
| `03_quantization/` | 원본 모델 변환 및 양자화 스크립트. |
| `04_inference/` | 대화형 테스트를 위한 **Streamlit UI** (`infer_ui.py`). |
| `10_stage1_cpt/` | 연속 사전 학습(CPT)을 위한 데이터 생성 및 학습 스크립트. |
| `20_stage2_sft/` | SFT 학습 스크립트(`train_omni_8b_paper_sft.sh`) 및 Golden Data 주입. |
| `30_evaluation/` | 벤치마크(`compare_hcx_stages.py`) 및 검증 스크립트. |
| `reference/` | 기존 프로젝트 파일(32B 모델, SolverX 실험) 아카이브. |

## 워크플로우 및 결과

### 1단계: CPT (지식 주입)
- **목표**: 모델에게 CLaRa 논문의 원문 텍스트 내용을 학습시킵니다.
- **과정**: PDF에서 추출한 36개의 텍스트 청크로 학습 진행.
- **결과**: 모델이 텍스트 패턴은 익혔으나, Q&A 형식으로 질문했을 때 정의를 제대로 인출하지 못함 (예: "SCP"를 "Self-optimizing Presenter"로 엉뚱하게 정의).

### 2단계: SFT (지시 튜닝 및 교정)
- **목표**: 환각을 수정하고 자연스러운 질의응답이 가능하게 합니다.
- **과정**:
    1.  CPT 데이터로부터 합성 Q&A 쌍 생성.
    2.  **"Golden Data" 주입**: *CLaRa*, *SCP*, *Joint Training*에 대한 정확한 정의를 수동으로 추가하여 학습 데이터 강화.
    3.  CPT 어댑터를 시작점으로 하여 재학습 (지속 학습, Continual Learning).
- **결과**: 모델이 이제 "SCP"를 **"Salient Compressor Pretraining"**으로, "CLaRa"를 **"Continuous Latent Reasoning"**으로 정확히 답변합니다.

### 벤치마크 결과
세 가지 단계의 모델 성능을 비교했습니다:

| 모델 단계 | CLaRa 정의 | SCP 정의 | 추론 속도 |
| :--- | :--- | :--- | :--- |
| **Base (4-bit)** | 오답 (환각 발생) | 오답 (Secure Copy Protocol로 인식) | ~15.9 t/s |
| **CPT** | 불안정 | 오답 (환각 발생) | ~15.8 t/s |
| **SFT (Final)** | **정답** | **정답** | **~15.7 t/s** |

## 실행 방법

### 1. 대화형 추론 (UI)
Streamlit UI를 통해 모델을 가장 쉽게 테스트할 수 있습니다.

```bash
# 가상 환경 활성화
source .venv/bin/activate

# UI 실행
streamlit run 04_inference/infer_ui.py
```
- **모델 선택**: `HyperCLOVAX-SEED-Omni-8B-Text-4bit`
- **어댑터 선택**: `adapters_omni_8b_paper_sft` (최종 모델)

### 2. CLI 검증
검증 스크립트를 실행하여 모델이 핵심 질문에 어떻게 답하는지 확인합니다.

```bash
python 30_evaluation/verify_paper_sft.py
```

### 3. 벤치마크 비교 실행
위의 비교 표를 재현하려면 다음을 실행하세요:

```bash
python 30_evaluation/compare_hcx_stages.py
```

## 레거시 프로젝트
HyperCLOVA X 32B 및 SolverX에 대한 이전 작업 내용은 `reference/README_legacy.md`에서 확인할 수 있습니다.

## 감사의 글 (Acknowledgements)
이 프로젝트는 [hcx-seed-think-32b-knowledge-injection](https://github.com/shezgone/hcx-seed-think-32b-knowledge-injection) 리포지토리에서 분기(Spin-off)되었습니다. 기반 코드를 제공해 준 원작자에게 감사를 표합니다.
