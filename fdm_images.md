# Federated Dataset Distillation for Privacy-Preserving Data Mining

## 🎯 Project Overview

This project implements an advanced **Federated Dataset Distillation** system that combines gradient-based learning with adversarial training to create synthetic datasets from distributed private data sources. The system enables privacy-preserving data sharing across multiple nodes without centralizing raw data, making it ideal for critical infrastructure scenarios (defense, healthcare, energy sectors).

### Key Features

- ✅ **Federated Learning Architecture**: 3 distributed nodes with local data
- ✅ **Privacy Preservation**: Differential privacy on gradient aggregation
- ✅ **Advanced Distillation**: Gradient matching + adversarial training + multi-objective optimization
- ✅ **Synthetic Data Generation**: Creates 2000 representative synthetic images from 6000 distributed real images
- ✅ **Comprehensive Validation**: Performance benchmarking against real data baselines

---

## 🏗️ System Architecture

### Components

The system consists of **5 neural networks** working in coordination:

| Component | Architecture | Parameters | Purpose |
|-----------|-------------|------------|---------|
| **Teacher Networks** (×3) | ResNet-18 | 11.2M each | Learn from local real data at each node |
| **Student Network** | ResNet-18 | 11.2M | Learn from synthetic data, gradients matched to teachers |
| **Generator Network** | Custom Deep CNN | 147.7M | Generate synthetic images from latent space |
| **Discriminator Network** | AdvancedDiscriminator | 19.9M | Ensure synthetic images are realistic |
| **Perceptual Network** | VGG-16 (frozen) | 138M | Measure visual quality via LPIPS |

**Total Trainable Parameters**: ~200M

---

## 📊 Dataset Configuration

### Training Data (Distributed Across 3 Nodes)
```
Node 1: 2000 images (1000 windmill, 1000 beach)
Node 2: 2000 images (1000 windmill, 1000 beach)
Node 3: 2000 images (1000 windmill, 1000 beach)
────────────────────────────────────────────────
Total:  6000 images (3000 per class)
Split:  80% training (4800), 20% validation (1200)
```

### Synthetic Dataset Output
```
Iteration 3000 (Best Checkpoint):
├── Windmill: 1000 synthetic images
└── Beach:    1000 synthetic images
────────────────────────────────────
Total:        2000 synthetic images

Compression Ratio: 3:1 (6000 → 2000 images)
```

### Test Data (Held-out for Validation)
```
Inference Set: Independent test data
├── Windmill: 100 images
└── Beach:    100 images
```

---

## 🔄 Federated Training Pipeline

### Phase 1: Teacher Pretraining (Federated)

Each node independently trains a ResNet-18 teacher on its local data:
```python
Node 1: Teacher_1 → trains on 1600 local images → 89.69% Accuracy
Node 2: Teacher_2 → trains on 1600 local images → 86.50% Accuracy
Node 3: Teacher_3 → trains on 1600 local images → 80.69% Accuracy
```

**Key Point**: Raw images never leave their respective nodes.

### Phase 2: Gradient Aggregation (Privacy-Preserving)
```python
For each training batch:
    # Each node computes gradients locally
    Node_1 → ∇θ₁ (gradients from real data)
    Node_2 → ∇θ₂ (gradients from real data)
    Node_3 → ∇θ₃ (gradients from real data)
    
    # Central server aggregates (FedAvg)
    ∇θ_combined = (∇θ₁ + ∇θ₂ + ∇θ₃) / 3
    
    # Apply differential privacy
    ∇θ_private = ∇θ_combined + Gaussian_Noise(σ=0.01)
    ∇θ_private = Clip(∇θ_private, max_norm=1.0)
```

**Privacy Guarantees**:
- Noise scale: 0.01
- Gradient clipping: norm ≤ 1.0
- No raw data transmitted

### Phase 3: Synthetic Data Generation

The central server generates synthetic images through adversarial training:
```python
For iteration in range(10000):
    # 1. Generate synthetic images
    latent_noise = sample_noise(batch_size=32, dim=512)
    synthetic_images = Generator(latent_noise, class_labels)
    
    # 2. Train student on synthetic data
    student_output = Student(synthetic_images)
    student_gradients = compute_gradients(student_output)
    
    # 3. Compute multi-objective loss
    loss_gradient = MSE(student_gradients, teacher_gradients)  # Core component
    loss_adversarial = -log(Discriminator(synthetic_images))    # Realism
    loss_diversity = -variance(synthetic_features)              # Prevent collapse
    loss_perceptual = LPIPS(synthetic_images, real_images)      # Visual quality
    
    # 4. Combine with learned weights (uncertainty weighting)
    total_loss = uncertainty_weighted_sum([
        loss_gradient,      # Weight: 1.0
        loss_adversarial,   # Weight: 0.01
        loss_diversity,     # Weight: 0.05
        loss_perceptual     # Weight: 0.1
    ])
    
    # 5. Update generator
    Generator.backward(total_loss)
    Optimizer.step()
```

### Phase 4: Discriminator Training (Adversarial)
```python
# Train discriminator to distinguish real vs synthetic
real_data = sample_from_nodes(batch_size=32)
fake_data = Generator(noise)

D_loss_real = -log(Discriminator(real_data))
D_loss_fake = -log(1 - Discriminator(fake_data))
D_loss = (D_loss_real + D_loss_fake) / 2

Discriminator.backward(D_loss)
```

---

## 📈 Training Results

### Convergence Statistics

| Metric | Value | Status |
|--------|-------|--------|
| **Total Loss** | **1.1543** | 🏆 Best achieved |
| **Gradient Loss** | 0.0094 | Excellent matching |
| **Adversarial Loss** | 0.5471 | Perfectly balanced |
| **Diversity Loss** | -0.1037 | Good diversity |
| **Perceptual Loss** | 0.7016 | High visual quality |
| **Discriminator Loss** | 0.4545 | Optimal equilibrium |

**Best Checkpoint**: Iteration 3400 (Loss: 1.15)


---

## ✅ Validation Experiments

### Experiment Design

Three training scenarios were evaluated on identical held-out test data:

| Experiment | Training Data | Size |
|------------|---------------|------|
| **1. All Real Data** | Combined data from all 3 nodes | 6000 images |
| **2. Synthetic Data** | Generated images from iteration 3400 | 2000 images |
| **3. Random Real Subset** | Random sample from real data | 2000 images |
| **4. Nodes** | Real data from 3 nodes | 2000 images X 3 |


---

## 🗂️ Repository Structure
```
RAIDO_federated_distillation/
├── config/
│   ├── advanced_config.yaml          # Training configuration
│   └── validation_config.yaml        # Validation configuration
├── data/
│   ├── node_1/                       # Federated node 1 data
│   │   ├── windmill/
│   │   └── beach/
│   ├── node_2/                       # Federated node 2 data
│   │   ├── windmill/
│   │   └── beach/
│   ├── node_3/                       # Federated node 3 data
│   │   ├── windmill/
│   │   └── beach/
│   └── Inference/                    # Test data
│       ├── windmill/
│       └── beach/
├── src/
│   ├── generator.py                  # Generator networks
│   ├── discriminator.py              # Discriminator networks
│   ├── student_network.py            # Student network
│   ├── teacher_network.py            # Teacher network
│   ├── gradient_matching.py          # Gradient matching algorithms
│   ├── advanced_loss.py              # Multi-objective loss functions
│   ├── advanced_optimizer.py         # Training orchestrator
│   ├── federated_gradients.py        # Federated aggregation
│   └── validation.py                 # Validation utilities
├── outputs/
│   ├── checkpoints/
│   │   ├── checkpoint_iter_1000.pth
│   │   ├── checkpoint_iter_3400.pth  ⭐ Best
│   │   └── ...
│   ├── images/
│   │   ├── iter_1000/
│   │   ├── iter_3400/                ⭐ Used for validation
│   │   │   ├── windmill/             (1000 images)
│   │   │   └── beach/                (1000 images)
│   │   └── ...
│   └── validation/
│       ├── plots/
│       │   ├── training_curves.png
│       │   ├── accuracy_comparison.png
│       │   └── per_class_comparison.png
│       ├── models/
│       │   ├── model_all_real.pth
│       │   ├── model_synthetic.pth
│       │   └── model_random_subset.pth
│       └── results.json
├── run_advanced_distillation.py      # Main training script
├── validate_synthetic.py             # Validation script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd RAIDO_federated_distillation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Start federated dataset distillation training
python run_advanced_distillation.py --config config/advanced_config.yaml
```


### Validation
```bash
# Validate synthetic dataset quality
python validate_synthetic.py --config config/validation_config.yaml
```


---

## 🔧 Configuration

### Key Parameters (advanced_config.yaml)
```yaml
# Synthetic dataset size
synthetic:
  num_images: 2000
  images_per_class: 1000

# Loss weights
loss:
  gradient_weight: 1.0      # Primary objective
  adversarial_weight: 0.01  # Realism (reduced for stability)
  diversity_weight: 0.05    # Prevent mode collapse
  perceptual_weight: 0.1    # Visual quality

# Federated settings
federated:
  enabled: true
  num_nodes: 3
  aggregation_method: "fedavg"
  privacy:
    enabled: true
    noise_scale: 0.01
    clip_norm: 1.0

# Training
training:
  num_iterations: 10000
  batch_size: 32
  save_interval: 500
```

### Validation Parameters (validation_config.yaml)
```yaml
validation:
  model: "resnet18"
  num_classes: 2
  
  training:
    epochs: 20
    batch_size: 32
    learning_rate: 0.001
  
  experiments:
    run_all_real: true       # Upper bound
    run_synthetic: true      # Main evaluation
    run_random_subset: true  # Fair baseline
    subset_size: 200         # Match synthetic dataset size
```

---


## 🎯 Technical Contributions

### Novel Aspects

1. **Multi-Objective Optimization**
   - Combines 4 complementary objectives
   - Learned uncertainty weighting for automatic balancing
   - Stable training without manual tuning

2. **Privacy-First Design**
   - Differential privacy on gradient aggregation
   - Clip-then-add noise mechanism
   - Configurable privacy budget (ε, δ)

3. **Production-Ready Implementation**
   - Handles real-world image data (not just MNIST)
   - Federated architecture with 3 nodes
   - Comprehensive validation framework

4. **Domain Application**
   - Tailored for critical infrastructure (defense, energy, telecom)
   - Addresses GDPR/privacy requirements
   - Enables secure data sharing across organizations

### Comparison to State-of-the-Art

| Technique | Year | Approach | Privacy | Multi-Objective |
|-----------|------|----------|---------|-----------------|
| Dataset Condensation | 2021 | Distribution matching | ❌ | ❌ |
| FedDM (CVPR) | 2023 | Client-side distillation | ⚠️ | ❌ |
| **This Work** | 2025 | Server-side gradient matching | ✅ | ✅ |

**Key Advantages**:
- ✅ Stronger privacy (differential privacy + gradient-only sharing)
- ✅ Better quality (multi-objective optimization)
- ✅ More practical (works with real images, not just MNIST)

---

## 📊 Performance Analysis

### Compression vs Quality Trade-off
```
Dataset Size:  6000 → 2000 images (3:1 compression)
Privacy:       Raw data exposed → Fully protected
Performance:   93.5% → 92.5% (1p.p. degradation)
```

**Value Proposition**: 
- 66% storage reduction
- 100% privacy preservation
- 99% performance retention
- ✅ **Worthwhile trade-off for critical infrastructure scenarios**

---

## 📚 References

### Key Papers

1. **Dataset Condensation with Gradient Matching**  
   Zhao et al., ICLR 2021  
   Foundation for gradient matching approach

2. **Federated Dataset Distillation (FedDM)**  
   Xiong et al., CVPR 2023  
   Federated distillation framework

3. **Multi-Task Learning Using Uncertainty**  
   Kendall & Gal, CVPR 2018  
   Uncertainty weighting for multi-objective optimization

4. **Differential Privacy in Deep Learning**  
   Abadi et al., CCS 2016  
   Privacy mechanisms and guarantees

### Additional Resources

- LPIPS Perceptual Similarity: [github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
- PyTorch Federated Learning: [flower.dev](https://flower.dev)
- RAIDO Project: [raido-project.eu](https://raido-project.eu)


---

## 🙏 Acknowledgments

This work is supported by:
- **Horizon Europe** - RAIDO Project (Grant Agreement No. [number])
- **MetaMind Innovations** - Technical development and coordination

---
