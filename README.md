# 🛍️ E-Commerce Customer Behavior Clustering and Segmentation

This project implements **K-Means Clustering** on an online store's transactional data to perform **Market Segmentation**. The goal is to move beyond simple demographic data and classify customers based on their purchasing behavior, enabling targeted marketing strategies, optimizing inventory, and enhancing customer loyalty programs.

## ⚙️ Methodology & Analysis Flow

The segmentation process strictly followed data science best practices:

1.  **Data Preprocessing:** Cleaning and structuring raw transactional data.
2.  **Feature Engineering (RFM):** Constructing behavioral metrics crucial for clustering.
3.  **Statistical Transformation:** Applying Log Transformation and Standardization to normalize features.
4.  **Optimal K Determination:** Using the **Elbow Method** to identify the ideal number of clusters.
5.  **K-Means Clustering:** Model execution and cluster assignment.
6.  **Cluster Interpretation:** Analyzing the statistical profile of each segment to derive business insights.

---

## 🔬 Feature Engineering: RFM Model

To capture the true essence of customer behavior, three critical features based on the **RFM (Recency, Frequency, Monetary)** model were engineered from the raw transactional data.

### 1. Engineered Features & Rationale

| Feature Name | Conceptual Definition | Construction Rationale |
| :--- | :--- | :--- |
| **Recency** | Number of days since the customer's last purchase. | **Goal:** Identify recently active customers. **Logic:** Calculated as the difference between a **Reference Date** (e.g., the day after the last transaction in the dataset) and the customer's last transaction date. Lower values indicate higher engagement. |
| **Frequency** | Total count of orders placed by the customer within the observed period. | **Goal:** Measure customer loyalty and transactional volume. **Logic:** Computed by counting the number of unique order IDs associated with each `CustomerID`. Higher values indicate more loyal customers. |
| **Monetary** | Total money spent by the customer over the observed period. | **Goal:** Quantify the customer's financial value to the business. **Logic:** Calculated by summing the total order value for each customer. Higher values identify the most profitable segment. |

### 2. Statistical Pre-processing

Feature scaling is **critical** for distance-based algorithms like K-Means.

| Statistical Operation | Rationale & Impact |
| :--- | :--- |
| **Log Transformation** | **Rationale:** RFM metrics inherently exhibit **high positive skewness** (a few customers buy very frequently/spend a lot, while most are infrequent/low-value). This transformation ($\log(x)$ or $\log(1+x)$) was applied to **`Frequency`** and **`Monetary`** to approximate a more **Normal Distribution**, mitigating the disproportionate influence of outliers. |
| **Standard Scaling** | **Rationale:** After transformation, the features were scaled using **`StandardScaler`** (mean = 0, std = 1). This ensures that the distance calculation in K-Means is not dominated by the magnitude of a single feature (e.g., `Monetary` value), allowing all three features to contribute equally to the cluster formation. |

---

## 🔢 Clustering Analysis & Model Results

### 1. Determining Optimal Cluster Count (K)

* **Statistical Metric:** **WCSS (Within-Cluster Sum of Squares)**.
* **Method:** The **Elbow Method** was employed by plotting WCSS against the number of clusters ($K$).
* **Result:** The "bend" or "elbow point" in the plot occurred at **$K=4$**, indicating that adding more than four clusters provides diminishing returns in reducing internal cluster variance. **Four clusters** were thus selected for the final model.

### 2. Cluster Interpretation (Profile Analysis)

The final 4 clusters were analyzed based on the **average RFM score** of their members to assign a business-relevant profile:

| Cluster ID | Avg. Recency (Days) | Avg. Frequency (Orders) | Avg. Monetary ($) | Behavioral Profile | Strategic Application |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | Low (Newest) | High (Frequent) | High (High Spenders) | **Champions / Loyalists** | Retention and Reward: Offer VIP programs, early access to new products. |
| **Cluster 1** | High (Oldest) | Low (Infrequent) | Low (Lowest Spenders) | **Lost Customers / Hibernating** | Re-activation: Offer aggressive discounts or personalized win-back campaigns. |
| **Cluster 2** | Medium | Low | Medium | **At-Risk / Needs Attention** | Prevention: Implement timely, targeted messaging to prevent churn and increase frequency. |
| **Cluster 3** | Low | Medium | Medium | **New/Growing Potential** | Growth: Focus on product education and cross-selling to increase Monetary value. |

---

## 🛠️ Requirements & Setup

1.  **Clone the repository:** `git clone YOUR_REPO_URL`
2.  **Install dependencies:** `pip install pandas numpy matplotlib seaborn scikit-learn`
3.  **Run the analysis:** Open and execute `Clustering_customer.ipynb` in a Jupyter environment.
