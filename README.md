# traffic-lawyer-llm

## 교통 AI 변호사 (Traffic Law AI Assistant)
공공데이터 기반 Instruction LLM Fine-Tuning 프로젝트
## 프로젝트 개요

이 프로젝트는 교통 관련 법률 상황을 입력하면, 해당 상황에서 어떤 법률 조항이 위반되는지를 자동으로 판단하는 AI 법률 보조 모델을 만드는 것을 목표로 합니다.
경찰청·법률민원센터 등에서 제공하는 공공데이터를 직접 수집·정제하여 instruction–input–output 구조의 데이터셋을 구축하고, Polyglot-ko 언어 모델을 QLoRA 방식으로 파인튜닝하여 실제 법률 판단이 가능한 모델을 만들었습니다.

## 프로젝트 특징

공공 법률 데이터셋을 직접 정제하여 Instruction Dataset 제작

Polyglot-ko 모델을 활용한 경량 파인튜닝

4bit quantization + LoRA 조합으로 Colab 환경에서 학습 가능하도록 최적화

샘플별 Loss 분석을 통해 성능 문제와 데이터 편향을 점검

실사용 형태의 “교통 AI 변호사” 시나리오 구현

### 프로젝트 구조
/traffic-lawyer-llm
 ┣ data/
 ┃ ┣ final_train_data.csv
 ┃ ┣ final_test_data.csv
 ┃ ┗ final_val_data.csv
 ┣ notebooks/
 ┃ ┗ capstone_finetuning.ipynb  ← 전체 실험 코드
 ┣ saved_model/
 ┃ ┗ polyglot-lora (LoRA weight 저장)
 ┣ README.md  ← 프로젝트 설명
 ┗ inference_example.py (옵션)

## 데이터셋 제작 과정
### 공공 데이터 수집

경찰청 교통사고 통계

법률정보 서비스(도로교통법 전문, 판례 예시)

온라인 질의응답 및 민원 사례

### 데이터 정제

문장 단위 정리

사건 → 위반 법률 매핑

의미 없는 텍스트(광고, 기타 설명) 제거

중복 사례 제거

### Instruction Dataset 구성

모델이 학습하기 쉬운 형태로 세 열 구조로 정리:

instruction	input	output
다음 상황에서 위반된 법률 조항을 식별하시오.	음주운전을 하다 적발됨	도로교통법 제44조 제1항 …
### HuggingFace Dataset 변환

Notebook 내부 코드:

dataset = Dataset.from_pandas(df.reset_index(drop=True))

## 모델 및 학습 방식 (QLoRA Fine-Tuning)
### Base Model
 - EleutherAI/polyglot-ko-1.3b

### Quantization

4bit NF4

BitsAndBytesConfig로 메모리 최적화

Colab T4에서도 학습 가능

### LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

### 학습 하이퍼파라미터

Epoch 10

Batch size 32

Learning rate 2e-4

fp16 + gradient checkpointing

### Trainer 기반 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)
trainer.train()

## 모델 평가 (Loss 기반 분석)

전체 테스트셋에 대해 샘플별 Loss를 계산:

loss = model(input_ids=input_ids, labels=labels).loss

분석 결과:

평균 Loss: (Notebook 실행 결과 값 입력)

표준편차: (Notebook 실행 결과 값 입력)

High-Loss 샘플 추출:

법률적 표현이 복잡하거나 사례가 불명확한 입력에서 오류 증가

데이터 정제나 instruction 다양화 필요성을 확인

### 추론 예시 (Inference Example)

입력:

술마시고 운전하다 적발됨


모델 출력:

도로교통법 제44조 제1항 음주운전 금지 위반에 해당합니다.


또는 Notebook 코드 내에서:

decoded[len(prompt):].strip()

## 성과 및 배운 점

도메인 데이터(법률)를 AI가 이해하도록 재구성하는 과정의 중요성을 체감

LoRA 기반 경량 파인튜닝 전체 파이프라인을 구현

법률 데이터는 문장 구조 차이로 인해 loss 편차가 크다는 특징 확인

Instruction tuning과 RAG의 차이·적용 위치를 명확하게 이해
