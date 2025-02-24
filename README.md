# **Hybrid Sampling in Machine Learning**
Hybrid sampling is a technique used to handle class imbalance in datasets by combining both **oversampling** and **undersampling** methods. It aims to improve the performance of classification models when dealing with imbalanced data, where one class significantly outnumbers the others.

---

## **Why Use Hybrid Sampling?**
- **Oversampling** alone (e.g., **SMOTE**) can increase redundancy and overfitting, especially when synthetic samples closely resemble existing ones.
- **Undersampling** alone (e.g., **Random Undersampling**) may remove valuable information by discarding important instances from the majority class.
- **Hybrid methods** strike a balance, improving generalization while maintaining the dataset’s original characteristics.

---

## **Types of Hybrid Sampling Techniques**
### 1. **SMOTE + Tomek Links**  
   - **SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic minority class samples.  
   - **Tomek Links** is an undersampling technique that removes samples from the majority class that are closest to the minority class, refining the decision boundary.

### 2. **SMOTE + Edited Nearest Neighbors (ENN)**  
   - **ENN** removes noisy majority class samples after applying SMOTE, helping reduce overfitting and increasing model robustness.

### 3. **SMOTE + NearMiss**  
   - **NearMiss** is an undersampling technique that selects majority class samples closest to the minority class to ensure better class separation.

### 4. **BalanceCascade**  
   - Iteratively undersamples majority class instances in different subsets to maintain diversity in training.

### 5. **Adaptive Synthetic Sampling (ADASYN) + Undersampling**  
   - **ADASYN** is an advanced SMOTE variant that generates synthetic data more adaptively, focusing on hard-to-learn samples, combined with undersampling for a balanced dataset.

---

## **Advantages of Hybrid Sampling**
✅ Reduces overfitting from excessive oversampling  
✅ Preserves important patterns from the majority class  
✅ Improves model generalization by refining the decision boundary  
✅ Works well with noisy datasets (especially when using ENN or Tomek Links)

---

## **Applications of Hybrid Sampling**
- **Medical Diagnosis:** Handling rare disease cases in datasets  
- **Fraud Detection:** Addressing the imbalance in fraud vs. non-fraud transactions  
- **Fault Detection:** Manufacturing defect classification  
- **Text Classification:** Identifying spam or fake news  
