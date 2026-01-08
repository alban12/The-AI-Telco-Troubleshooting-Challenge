# The AI Telco Troubleshooting Challenge - Track 3

Fine-tuning a specialized edge-cloud LLM (Qwen2.5-1.5B-Instruct) for detecting and explaining unseen network failures in telecom environments.

**Challenge Link:** [Zindi Challenge](https://zindi.africa/competitions/the-ai-telco-troubleshooting-challenge)

## 📋 Challenge Overview

### Track 3 Objective
Can your fine-tuned LLM detect and explain unseen network failures?

Enhance the accuracy of **Qwen2.5-1.5B-Instruct** when answering telco troubleshooting questions in telelogs data. This track focuses on building a specialized edge-cloud LLM capable of:
- Diagnosing network faults from automatically generated fault and event logs (telelogs)
- Performing root-cause analysis with limited computational resources
- Generalizing across unseen faults and new network environments
- Running efficiently on constrained edge servers

### Evaluation Metric
- **Pass @ 1**: Measures the ability of the model to produce a correct answer in a single attempt
- Evaluated on troubleshooting capability and knowledge retention
- Private dataset includes network faults with different data structures and general knowledge questions

## 🏆 Prize
- 🥇 1st Place: $8,500 + leader pass (worth $2,750) + travel & accommodation to MCW Barcelona (March 2026)

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── train/                    # Training data
│   ├── test/                     # Test data (Phase 1 & 2)
│   └── processed/                # Processed and prepared data
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_fine_tuning.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py          # Data preprocessing
│   ├── model_trainer.py          # Fine-tuning logic
│   ├── inference.py              # Inference pipeline
│   └── evaluation.py             # Evaluation metrics
├── models/
│   └── qwen2.5-1.5b-finetuned/   # Fine-tuned model checkpoint
├── outputs/
│   ├── submission.csv            # Final submission file
│   └── metrics/                  # Evaluation results
└── report/
    └── technical_report.md       # Technical report (~2 pages)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA 12.0+ (for GPU acceleration)
- 8GB+ GPU memory (for edge deployment considerations)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd The-AI-Telco-Troubleshooting-Challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
# See requirements.txt for complete list
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
bitsandbytes>=0.41.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## 📊 Data Overview

### Dataset Structure
- **Training set**: Labeled telelogs with root cause annotations
- **Test set**: Unlabeled telelogs for generating predictions
- **Question format**: Network troubleshooting questions based on telelogs
- **Target**: 4 generations per question required for Pass @ 1 evaluation

### Submission Format
```csv
ID,Qwen2.5-1.5B-Instruct
question_1_1,root_cause_explanation
question_1_2,root_cause_explanation
question_1_3,root_cause_explanation
question_1_4,root_cause_explanation
...
```

## 🔧 Methodology

### 1. Data Preprocessing
- [ ] Exploratory analysis of telelogs
- [ ] Text cleaning and normalization
- [ ] Data augmentation strategies
- [ ] Train/validation split strategy

### 2. Model Fine-tuning
- **Base Model**: Qwen2.5-1.5B-Instruct
- **Approach**: Low-rank adaptation (LoRA) / QLoRA for efficient tuning
- **Training Strategy**: 
  - Multi-epoch training with early stopping
  - Learning rate scheduling
  - Gradient accumulation for larger effective batch sizes
  
### 3. Inference & Generation
- Batch inference on test set
- Multiple generation strategies (temperature, top-k, top-p)
- Post-processing and format validation

### 4. Evaluation
- Knowledge retention assessment
- Cross-domain generalization testing
- Pass @ 1 metric calculation

## 📈 Results & Performance

### Baseline
- Model: Qwen2.5-1.5B-Instruct (base, no fine-tuning)
- Score: TBD

### Fine-tuned Model
- Method: [Describe your approach]
- Public Leaderboard Score: TBD
- Private Leaderboard Score: TBD
- Key Improvements: TBD

### Metrics
```
Pass @ 1 Score: X.XX%
Knowledge Retention: X.XX%
Inference Time (edge): X.XXs
Model Size: X.XGB
```

## 🛡️ Responsible AI & Ethical Considerations

### Data Privacy and Compliance
- Complies with CC-BY SA 4.0 license
- [Details on data handling and privacy measures]

### Model Security Risks
- [Assessment of potential security vulnerabilities]
- [Mitigation strategies implemented]

### Data and Model Access Control
- [Transparency measures]
- [Access control mechanisms]

### Edge Computing Considerations
- Optimized for resource-constrained environments
- Model quantization strategies (int8, int4)
- Inference optimization for latency-sensitive deployments
- [Security measures for edge deployment]

### Data Governance
- [Data lineage tracking]
- [Quality assurance processes]
- [Bias and fairness assessments]

## 🔄 Reproducibility

### To Reproduce Results
```bash
# Set seeds for reproducibility
export SEED=42

# Run preprocessing
python src/preprocessing.py --config config/preprocessing.yaml

# Fine-tune model
python src/model_trainer.py --config config/training.yaml

# Generate predictions
python src/inference.py --model models/qwen2.5-1.5b-finetuned --test-data data/test

# Evaluate
python src/evaluation.py --predictions outputs/submission.csv
```

### Key Hyperparameters
```yaml
seed: 42
learning_rate: 2e-4
batch_size: 4
max_epochs: 3
warmup_steps: 500
weight_decay: 0.01
```

## 📚 References

- [Qwen Model Documentation](https://huggingface.co/Qwen)
- [ScalarLM Framework](https://github.com/scalar-labs/scalar-lm)
- [LoRA Fine-tuning](https://arxiv.org/abs/2106.09685)
- [Telco AI Research](https://example.com)

## 📝 Technical Report

A comprehensive technical report (2 pages) is included in the `report/` directory covering:
- Methodology and approach
- Data privacy and compliance considerations
- Model security measures
- Edge computing optimizations
- Results and analysis

See [report/technical_report.md](report/technical_report.md) for details.

## 📋 Submission Checklist

- [ ] Model fine-tuned and evaluated
- [ ] Predictions generated for all test samples
- [ ] Submission file formatted correctly (CSV with proper columns)
- [ ] Code is reproducible and documented
- [ ] Technical report completed (~2 pages)
- [ ] Model and code made publicly available
- [ ] All seeds set for reproducibility
- [ ] Top 10 criteria: 48-hour submission window checked

## 📖 License

This project uses the CC-BY SA 4.0 license. See LICENSE file for details.

## 🤝 Contributing

This is a competition submission. Contributions following the Zindi challenge rules are welcome.

---

**Challenge Timeline:**
- **Phase 1**: 28 Nov - 17 Jan (Initial questions released)
- **Phase 2**: 18 Jan - 02 Feb (Remaining questions released)
- **Deadline**: 02 Feb 2026
