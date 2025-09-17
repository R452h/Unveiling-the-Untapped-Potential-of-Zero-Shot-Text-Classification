# Unveiling the Untapped Potential of Zero-Shot Text Classification

A project exploring the power of **Zero-Shot Learning (ZSL)** for sentiment classification using the **facebook/bart-large-mnli** model from Hugging Face Transformers.  
We compare ZSL with traditional supervised approaches such as **Naive Bayes**, **Support Vector Machines (SVM)**, and **Recurrent Neural Networks (RNN)** across multiple datasets.

---

## 🚀 Project Overview
Traditional NLP text classification requires **large amounts of labeled data** and frequent retraining for new categories.  
Zero-Shot Text Classification overcomes these limitations by leveraging pre-trained language models to classify text **without explicit training** on each label.

This project:
- Implements **ZSL and traditional models** (Naive Bayes, SVM, RNN).
- Evaluates performance using **Accuracy, Precision, Recall, and F1-score**.
- Visualizes results with **graphs** for a clear comparison.
- Demonstrates **real-life applications** in sentiment analysis and links to **Sustainable Development Goals (SDG 3 & SDG 8).**

---

## 📊 Datasets
We experimented with three datasets:

1. **Amazon Reviews Dataset (10,000 rows)**  
   - Columns: `text`, `label` (positive/negative).  

2. **Twitter Sentiment Dataset (7,920 rows)**  
   - Columns: `id`, `tweet`, `label` (0 = positive, 1 = negative).  

3. **Restaurant Food Reviews Dataset (1,000 rows)**  
   - Columns: `Review`, `Liked` (0 = negative, 1 = positive).  

---

## 🛠️ Methodology
- **Zero-Shot Learning (ZSL):**  
  Used `facebook/bart-large-mnli` model from Hugging Face Transformers.  
  Labels are framed as hypotheses (e.g., *“This text is positive”*) and compared with input text for entailment.

- **Naive Bayes:**  
  Applied Multinomial NB with TF-IDF vectorization.

- **Support Vector Machine (SVM):**  
  Linear kernel, TF-IDF features.

- **Recurrent Neural Network (RNN):**  
  Bidirectional GRU with embedding layers.

---

## 📈 Results

| Dataset | Model        | Accuracy | Precision | Recall | F1-Score |
|---------|-------------|----------|-----------|--------|----------|
| Amazon Reviews | **ZSL** | **0.92** | 0.95 | 0.88 | 0.91 |
|             | Naive Bayes | 0.83 | 0.83 | 0.83 | 0.83 |
|             | SVM         | 0.87 | 0.87 | 0.87 | 0.87 |
|             | RNN         | 0.81 | 0.78 | 0.89 | 0.69 |
| Twitter Data | **ZSL** | **0.93** | 0.84 | 0.97 | 0.90 |
|             | Naive Bayes | 0.89 | 0.89 | 0.89 | 0.89 |
|             | SVM         | 0.89 | 0.89 | 0.89 | 0.89 |
|             | RNN         | 0.87 | 0.78 | 0.73 | 0.75 |
| Restaurant Reviews | **ZSL** | **0.98** | 0.97 | 0.98 | 0.98 |
|             | Naive Bayes | 0.83 | 0.83 | 0.81 | 0.82 |
|             | SVM         | 0.79 | 0.80 | 0.79 | 0.79 |
|             | RNN         | 0.78 | 0.78 | 0.80 | 0.79 |

📌 **ZSL consistently outperformed traditional models**, achieving near-perfect results on restaurant reviews.

---

## 🌍 Real-Life Applications
- Social media sentiment monitoring.  
- Customer feedback classification (positive/negative).  
- Product review analysis for businesses.  
- Public opinion mining for **healthcare** and **job market** insights.  

### Contribution to Sustainable Development Goals
- **SDG 3: Good Health and Well-Being** – Analyze public opinion on healthcare initiatives.  
- **SDG 8: Decent Work and Economic Growth** – Assess sentiment around job creation and unemployment.  

---

## 📦 Tech Stack
- **Python 3.x**
- **Hugging Face Transformers**
- **scikit-learn**
- **TensorFlow / PyTorch**
- **Pandas, NumPy, Matplotlib**

---

## 📌 Future Scope
- Extend ZSL to **multi-class classification** (beyond binary sentiment).  
- Optimize computational efficiency using **parallel processing**.  
- Explore **few-shot learning** to complement ZSL.  

---

## 👨‍💻 Author
- **Sushant Raj** 

## 📜 References
Key references include recent works on ZSL in **healthcare, social media, and news classification**.  
(Full reference list is available in the project report.)
<img width="1007" height="543" alt="image" src="https://github.com/user-attachments/assets/74ad110e-ff20-450a-8197-af5ad9f099f2" />
/example.png)
<img width="925" height="516" alt="image" src="https://github.com/user-attachments/assets/14296bbe-fa7a-465f-ab50-31bec495eed2" />
<img width="975" height="549" alt="image" src="https://github.com/user-attachments/assets/7334f8bc-0b68-47bb-a005-12ee7807f68b" />
<img width="947" height="626" alt="image" src="https://github.com/user-attachments/assets/f8d16434-f1c5-46e9-a554-006eb3cb7fe0" />
<img width="960" height="416" alt="image" src="https://github.com/user-attachments/assets/69647694-ba0f-49ce-810b-715ceb49cfbb" />
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/a2aa7d21-fb39-492a-8a60-3fcdace6ba82" />
<img width="948" height="571" alt="image" src="https://github.com/user-attachments/assets/7efa5dfc-b1f2-4609-a226-20dc687bc952" />
<img width="956" height="538" alt="image" src="https://github.com/user-attachments/assets/500c893d-8d2d-40f0-8eaa-2fb14a83913d" />
<img width="1014" height="561" alt="image" src="https://github.com/user-attachments/assets/82ef1fcf-2eec-4981-80de-8fa061170fd8" />
<img width="1014" height="570" alt="image" src="https://github.com/user-attachments/assets/2a2e291d-6b9c-4ba0-81c8-2cc6d8050055" />
<img width="1013" height="570" alt="image" src="https://github.com/user-attachments/assets/3390443a-d19b-4d19-b941-c330490faf02" />
<img width="1009" height="567" alt="image" src="https://github.com/user-attachments/assets/7678aef5-5a0d-45b6-8e70-ab76dbc373c0" />
<img width="1920" height="1080" alt="Screenshot (1)" src="https://github.com/user-attachments/assets/fba2d6ec-d1a9-48fa-b3f3-bb3ab8291a4f" />













