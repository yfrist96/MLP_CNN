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
  <img src="https://github.com/user-attachments/assets/b8678682-be5e-4385-abb5-5169a5f19325" alt="Learning Rate Effects" width="500"/>
</p>

#### Epochs
Too few epochs prevent the model from learning patterns effectively, leading to high losses. Too many epochs cause **overfitting**, observed around the **80-epoch mark**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d28e0919-9a3f-426b-8c71-c9ad2011d8bc" alt="Epochs Effect" width="500"/>
</p>

#### Batch Normalization
Batch normalization stabilizes training, leading to faster convergence compared to models without it.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7041ce4d-e983-4346-bcad-267e8ac8cf3c" alt="Batch Normalization" width="500"/>
</p>

#### Batch Size
- **Larger batch sizes** achieve **higher test accuracy**.
- **Smaller batch sizes** result in **slower training speeds per epoch**.
- **Stability**: Larger batch sizes show **smoother loss curves** compared to smaller ones.

<p align="center">
  <img src="https://github.com/user-attachments/assets/05a82609-6e3a-4863-a9f0-85d826543668" alt="Batch Normalization" width="500"/>
</p>

### Evaluating MLP Performance  
_Code implementation: `main.py`_

- **Best model:** Achieved **validation accuracy ≈ 0.75**, with training, validation, and test losses stabilizing around **0.25**.

<p align="center">
  <img alt="Light" src="https://github.com/user-attachments/assets/bb060533-db8f-492b-a617-07506ae994c0" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/user-attachments/assets/1d846194-49e8-49b8-9c25-b1b30be0f142" width="45%">
</p>

- **Worst model:** Showed validation accuracy **≈ 0.50**, with higher loss convergence (**≈ 1.0**).

<p align="center">
  <img alt="Light" src="https://github.com/user-attachments/assets/380156a6-c3bf-4754-88bf-4b704996de17" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/user-attachments/assets/20d2207b-c43c-4ee1-a1eb-fc368a9b6a59" width="45%">
</p>

#### Depth vs. Accuracy
- **Optimal depth:** **2 hidden layers**
- More layers lead to **diminishing returns** due to the **vanishing gradient problem**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b49e8f66-88f8-40f0-b9a2-74ef0b959fc1" alt="Depth vs Accuracy" width="500"/>
</p>

#### Width vs. Accuracy
- **Optimal width:** **30 neurons per hidden layer**
- Too few or too many neurons reduce accuracy.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b5ab32ee-e76a-476f-b803-ce928a994fe9" alt="Width vs Accuracy" width="500"/>
</p>

#### Monitoring Gradients
- Without batch normalization: **Vanishing gradients** in the early layers.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f9ae681b-96ea-441d-8dd8-775fbe09811c" alt="Width vs Accuracy" width="500"/>
</p>

- With batch normalization: **Gradients explode**, requiring careful tuning.

<p align="center">
  <img src="https://github.com/user-attachments/assets/40b863ad-787b-4b29-91ea-f2d2f0945382" alt="Width vs Accuracy" width="500"/>
</p>

#### Implicit Representation
Transforming input coordinates using **sine and cosine functions** improves decision boundaries, allowing the model to learn more complex patterns.

<p align="center">
  <img alt="Light" src="https://github.com/user-attachments/assets/c48859d0-0978-452f-8f75-8d8ab0fc5500" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/user-attachments/assets/9ffa6017-fa1f-4b30-acc3-af1958bb93ae" width="45%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/37526961-0607-452a-8709-6afd1d745644" alt="Width vs Accuracy" width="500"/>
</p>

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
