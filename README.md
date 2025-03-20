# Machine Learning Methods - Experimenting with MLPs & CNNs

## Project Summary

In this project, I experimented with **Multi-Layer Perceptrons (MLPs)** and **Convolutional Neural Networks (CNNs)**. The study begins with MLPs applied to a dataset of European countries and later transitions to **image classification** using CNNs to detect deepfake-generated images.

---

## Multi-Layer Perceptrons (MLPs)

### Dataset
The dataset consists of tables with three columns:
- **Longitude**
- **Latitude**
- **Country** (encoded as an integer)

The goal is to classify a city into its corresponding country based on geographical coordinates.

### Optimizing MLP Training  
_Code implementation: `NN_tutorial.py`_

#### Learning Rate
A high learning rate can lead to **oscillations in loss**, making training unstable. Conversely, a lower learning rate ensures **more stable learning**. The figure below demonstrates these effects:

<p align="center">
  <img src="path/to/learning_rate_plot.png" alt="Learning Rate Effects" width="500"/>
</p>

#### Epochs
Too few epochs prevent the model from learning patterns effectively, leading to high losses. Too many epochs cause **overfitting**, observed around the **80-epoch mark**.

<p align="center">
  <img src="path/to/epochs_plot.png" alt="Epochs Effect" width="500"/>
</p>

#### Batch Normalization
Batch normalization stabilizes training, leading to faster convergence compared to models without it.

<p align="center">
  <img src="path/to/batch_norm_plot.png" alt="Batch Normalization" width="500"/>
</p>

#### Batch Size
- **Larger batch sizes** achieve **higher test accuracy**.
- **Smaller batch sizes** result in **slower training speeds per epoch**.
- **Stability**: Larger batch sizes show **smoother loss curves** compared to smaller ones.

### Evaluating MLP Performance  
_Code implementation: `main.py`_

- **Best model:** Achieved **validation accuracy ≈ 0.75**, with training, validation, and test losses stabilizing around **0.25**.
- **Worst model:** Showed validation accuracy **≈ 0.50**, with higher loss convergence (**≈ 1.0**).

#### Depth vs. Accuracy
- **Optimal depth:** **2 hidden layers**
- More layers lead to **diminishing returns** due to the **vanishing gradient problem**.

<p align="center">
  <img src="path/to/depth_vs_accuracy.png" alt="Depth vs Accuracy" width="500"/>
</p>

#### Width vs. Accuracy
- **Optimal width:** **30 neurons per hidden layer**
- Too few or too many neurons reduce accuracy.

<p align="center">
  <img src="path/to/width_vs_accuracy.png" alt="Width vs Accuracy" width="500"/>
</p>

#### Monitoring Gradients
- Without batch normalization: **Vanishing gradients** in the early layers.
- With batch normalization: **Gradients explode**, requiring careful tuning.

#### Implicit Representation
Transforming input coordinates using **sine and cosine functions** improves decision boundaries, allowing the model to learn more complex patterns.

---

## Convolutional Neural Networks (CNNs)

### Task: Deepfake Image Classification
Using CNNs, I tackled **binary classification** to distinguish real human faces from deepfake-generated images. The dataset is available [here](https://drive.google.com/file/d/1KBT7gpeo2fDj8-J_pc3dFWu4LLEDwKKp/view).

### Model Comparisons

#### XGBoost (Baseline)
- **Accuracy:** 73.5%

#### Training from Scratch (ResNet18)
- **Epoch 1 Loss:** 0.7758
- **Validation Accuracy:** 52.5%
- **Test Accuracy:** 52.25%
- **Learning Rate:** 0.01

#### Linear Probing (Pretrained ResNet18)
- **Epoch 1 Loss:** 0.6795
- **Validation Accuracy:** 69.5%
- **Test Accuracy:** 72.5%
- **Learning Rate:** 0.01

#### Fine-Tuning (Pretrained ResNet18)
- **Epoch 1 Loss:** 0.6244
- **Validation Accuracy:** 74.0%
- **Test Accuracy:** 77.5%
- **Learning Rate:** 0.001

### Best vs. Worst Model Comparison
- **Best model:** **Fine-tuned ResNet18** (Test accuracy = **77.5%**)
- **Worst model:** **Training from scratch** (Test accuracy = **52.25%**)

### Sample Analysis
Five images correctly classified by the **Fine-tuned model** but misclassified by the **Training from scratch model**:

<p align="center">
  <img src="path/to/correctly_classified_samples.png" alt="Correctly Classified Samples" width="500"/>
</p>

---

## Results & Insights

### Key Findings
- **MLPs work well for structured tabular data**, but require careful tuning of depth, width, and batch normalization.
- **CNNs are highly effective for image classification**, especially with **transfer learning** (e.g., **fine-tuning ResNet18**).
- **Fine-tuning a pretrained model significantly outperforms training from scratch**.
- **Implicit representation techniques enhance MLPs' ability to capture complex patterns.**

### Future Work
- Further optimize CNN performance to exceed **97% test accuracy**.
- Experiment with **attention-based architectures** such as Vision Transformers (ViTs) for image classification.
- Extend MLP analysis to **higher-dimensional structured data.**

---

## How to Run the Code

### Setup
Ensure you have the necessary dependencies:
```bash
pip install numpy pandas torch torchvision xgboost matplotlib
```

### Running MLP Experiments
```bash
python NN_tutorial.py
python main.py
```

### Running CNN Experiments
```bash
python cnn.py
```

---

## Author
**Your Name**  
**GitHub:** [YourGitHubProfile](https://github.com/yourgithubprofile)
