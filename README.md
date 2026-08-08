# 🏨 Investigate Hotel Business Using Data Visualization

## 📌 Project Overview

This project analyzes hotel booking and cancellation behavior using the Hotel Booking Demand Dataset (2017–2019). The goal is to help hotel managers understand customer booking patterns, identify cancellation risks, optimize revenue, and make data-driven business decisions.

The dashboard was developed using **Python, Streamlit, Plotly, Pandas, NumPy, and Scikit-Learn** and transforms a traditional notebook analysis into an interactive Business Intelligence application.

---

## 🎯 Business Questions Answered

### 1️⃣ Which hotel type do customers book most often?

* Compare booking volume between City Hotels and Resort Hotels.
* Analyze monthly demand patterns and seasonality.
* Identify peak and low-demand periods.

### 2️⃣ Does length of stay affect cancellation rate?

* Measure cancellation behavior across different stay durations.
* Compare trends between hotel types.
* Discover high-risk booking segments.

### 3️⃣ Does lead time affect cancellation rate?

* Analyze how far in advance customers book.
* Identify lead-time ranges with the highest cancellation risk.
* Compare booking stability across hotel types.

---

# 🚀 Live Dashboard

### Streamlit Application

https://investigate-hotel-business-using-data-visualization-alhvl9osag.streamlit.app/

---

# 📊 Interactive Dashboard Features

## Executive Overview

Provides an instant snapshot of business performance through KPI cards:

* Total Bookings
* Cancellation Rate
* Average Lead Time
* Average Daily Rate (ADR)
* Average Stay Length
* Estimated Revenue

---

## Hotel Type & Seasonality Analysis

### Interactive Visuals

* Booking Share Pie Chart
* Monthly Booking Trend Line Chart
* Booking Heatmap (Month × Year)

### Insights Generated

* Most popular hotel type
* Seasonal booking peaks
* Demand fluctuations throughout the year

---

## Stay Duration vs Cancellation Analysis

### Interactive Visuals

* Cancellation Rate by Hotel Type
* Dynamic Stay Length Slider
* Cancellation Trend by Total Nights
* Stay Duration Group Comparison

### Insights Generated

* Impact of stay length on cancellation behavior
* Risk differences between hotel types
* Identification of long-stay risk segments

---

## Lead Time vs Cancellation Analysis

### Interactive Visuals

* Lead Time Bucket Analysis
* Cancellation Rate Comparison
* Lead Time Heatmap

### Insights Generated

* High-risk booking windows
* Booking behavior patterns
* Relationship between planning horizon and cancellation probability

---

## Revenue Intelligence Dashboard

### Interactive Visuals

* Revenue by Hotel Type
* Monthly Revenue Trend
* Revenue Lost to Cancellations

### KPIs

* Estimated Total Revenue
* Revenue from Honored Bookings
* Revenue Lost Due to Cancellations

### Business Value

Helps management understand the financial impact of cancellations and booking patterns.

---

## 🌍 Global Customer Analysis

### Interactive Visuals

* World Choropleth Map
* Top Customer Countries

### Insights Generated

* Geographic demand distribution
* Key international markets
* Regional customer concentration

---

## 👥 Customer Segmentation

Customers are automatically categorized into:

* Family
* Couple
* Solo / Other

### Interactive Visuals

* Customer Segment Distribution
* Cancellation Rate by Segment

### Business Value

Supports targeted marketing campaigns and personalized offers.

---

## 📈 Market Intelligence

Analyze performance across:

* Market Segments
* Distribution Channels
* Customer Types
* Deposit Types

### Interactive Visuals

* Booking Volume Analysis
* Revenue Analysis
* Cancellation Rate Analysis

### Business Value

Identifies the most profitable and highest-risk customer acquisition channels.

---

## 🔗 Correlation Analysis

### Interactive Visuals

* Dynamic Correlation Heatmap
* User-Selected Variables

### Insights Generated

* Strongest drivers of cancellations
* Relationships among booking variables
* Hidden business patterns

---

## 🔮 Forecasting Module

### Interactive Visuals

* Historical Booking Trends
* Future Booking Projection

### Features

* Adjustable forecast horizon
* Trend-based projection model

### Business Value

Supports demand planning and resource allocation.

---

## 🧪 What-If Simulator

### Interactive Controls

Adjust expected cancellation reduction percentage.

### Simulates

* Saved Bookings
* Reduced Cancellations
* Recovered Revenue

### Business Value

Measures the potential impact of policy changes before implementation.

---

## ⚖ Hotel Comparison Dashboard

### Interactive Visuals

* Radar Chart Comparison
* Performance Benchmarking

### Metrics Compared

* ADR
* Cancellation Rate
* Lead Time
* Stay Length

### Business Value

Provides a side-by-side comparison between City and Resort Hotels.

---

# 🤖 Machine Learning Cancellation Prediction

A Random Forest model predicts cancellation probability for future bookings.

### Model Features

* Lead Time
* Stay Length
* ADR
* Customer Characteristics
* Booking History
* Market Information

### Outputs

* Cancellation Probability
* Risk Classification

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk

### Additional Visuals

* Feature Importance Analysis
* Risk Distribution Charts

---

# 🔥 Revenue-at-Risk Analysis

### Interactive Visuals

* Risk Band Distribution
* Revenue-at-Risk Analysis
* High-Risk Booking Table

### Business Value

Allows management to proactively protect revenue from likely cancellations.

---

# 🤖 AI Business Assistant

Built-in intelligent assistant capable of answering questions such as:

* Which hotel has the highest cancellation rate?
* Which month generates the most revenue?
* How can we reduce cancellations?
* Which customer segment should we target?

The assistant responds directly from the filtered dataset.

---

# 🧹 Data Quality Dashboard

Provides transparency into the data cleaning process.

### Data Quality Metrics

* Missing Values
* Duplicate Records
* Invalid ADR Values
* Zero-Guest Bookings
* Rows Removed During Cleaning

---

# 📥 Export & Reporting

Users can:

* Download filtered datasets
* Generate reports
* Export insights for business presentations

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly
* Streamlit

### Machine Learning

* Scikit-Learn
* Random Forest Classifier

### Deployment

* Streamlit Community Cloud

---

# 📂 Dataset

**Hotel Booking Demand Dataset**

* Period: 2017–2019
* Records: ~119,000 hotel bookings
* Hotels:

  * City Hotel
  * Resort Hotel

Key variables include:

* Lead Time
* ADR
* Cancellation Status
* Stay Duration
* Market Segment
* Customer Type
* Country
* Deposit Type

---

# 💡 Key Business Recommendations

✅ Introduce stricter cancellation policies for long lead-time bookings.

✅ Deploy automated reminder campaigns before arrival dates.

✅ Focus marketing efforts on high-value customer segments.

✅ Optimize pricing during seasonal demand peaks.

✅ Prioritize intervention on high-risk bookings identified by the ML model.

---

# 👨‍💻 Author

**P Suman Sangeet**

PGDM (Big Data Analytics)

Data Analyst | Business Intelligence | Machine Learning | Streamlit Developer

---

⭐ If you found this project useful, consider giving the repository a star.
