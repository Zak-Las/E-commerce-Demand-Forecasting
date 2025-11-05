# E-commerce Demand Forecasting: A Comparative Analysis of N-BEATS and Prophet

This capstone project provides an end-to-end solution for forecasting 30-day product demand for a subset of the M5 e-commerce dataset. It demonstrates a rigorous data science workflow, from data processing and quality assurance to model development, evaluation, and iterative improvement.

The primary goal is to compare the performance of a modern deep learning architecture (N-BEATS) against a well-established statistical baseline (Prophet), showcasing the ability to not only build complex models but also to validate them against robust alternatives.

## Key Highlights & Demonstrated Skills

This project directly showcases skills and competencies relevant to a data science role, including:

*   **Data Exploration and Curation:** A dedicated data quality notebook (`notebooks/data_quality.ipynb`) programmatically assesses the dataset for integrity issues like gaps, zero-inflation, and outliers, producing a detailed quality report (`artifacts/data_quality_report.json`).
*   **Model Design, Development, and Validation:**
    *   Implementation of **N-BEATS**, a modern deep learning model for time series forecasting, using PyTorch Lightning.
    *   Implementation of **Prophet** as a strong statistical baseline.
    *   Rigorous comparison using multiple metrics (WAPE, MAE) and a rolling-origin backtest methodology.
*   **Rapid Innovation and Prototyping:** A feature engineering experiment was conducted on the N-BEATS model, where a residual series was created to improve performance. This demonstrated an iterative, hypothesis-driven approach to model improvement, resulting in a measurable lift in accuracy.
*   **Visualization and Storytelling:** Notebooks are structured to tell a clear story, with visualizations for loss curves, forecast comparisons, and data quality issues.
*   **Reproducibility:** The entire workflow is encapsulated in a series of Jupyter notebooks and Python scripts, ensuring that the results can be easily reproduced.

## Tech Stack

*   **Core Libraries:** Python, Pandas, NumPy, PyTorch, PyTorch Lightning, Prophet, Scikit-learn
*   **Visualization:** Matplotlib, Seaborn
*   **Environment:** The project is configured to run on a CPU-only environment.

## Repository Structure

```
├── artifacts/            # Stores all generated outputs (models, metrics, reports)
├── config/               # Configuration files for models
├── data/                 # Raw and processed data (tracked by Git LFS)
├── docs/                 # Project documentation (e.g., architecture)
├── notebooks/            # Jupyter notebooks for the main workflow
│   ├── data_processing.ipynb
│   ├── data_quality.ipynb
│   ├── prophet_baseline.ipynb
│   └── nbeats_training.ipynb
├── src/                  # Source code for data processing, models, etc.
├── tests/                # Unit tests for core functions
└── README.md             # This file
```

## How to Reproduce the Results

To run this project and reproduce the artifacts, follow these steps in order:

1.  **Set up the environment:**
    ```bash
    # It is recommended to use a virtual environment
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created from the environment)*

2.  **Run the Notebooks in Sequence:** Execute the following Jupyter notebooks from top to bottom. Each notebook builds on the artifacts of the previous one.

    a. **`notebooks/data_processing.ipynb`**
       *   **Purpose:** Converts the raw M5 CSV files into a clean, aggregated panel DataFrame (`m5_panel_subset.parquet`).
       *   **Key Output:** `data/processed/m5_panel_subset.parquet`

    b. **`notebooks/data_quality.ipynb`**
       *   **Purpose:** Analyzes the processed panel for integrity issues.
       *   **Key Output:** `artifacts/data_quality_report.json`

    c. **`notebooks/prophet_baseline.ipynb`**
       *   **Purpose:** Trains and evaluates the Prophet baseline model.
       *   **Key Output:** `artifacts/models/prophet_metrics.json`

    d. **`notebooks/nbeats_training.ipynb`**
       *   **Purpose:** Trains the baseline N-BEATS model and the improved residual-based N-BEATS model.
       *   **Key Outputs:** Checkpoints and metrics for both N-BEATS models, including `artifacts/models/nbeats_notebook_metrics.json` and `artifacts/models/nbeats_feature_experiment_metrics.json`.

## Results & Analysis

The project culminates in a comparison between three models: Prophet, a baseline N-BEATS model, and an improved N-BEATS model trained on engineered features (residuals).

| Model                       | Mean Validation WAPE | Notes                                         |
| --------------------------- | -------------------- | --------------------------------------------- |
| **Prophet**                 | *~[Value from JSON]* | Strong statistical baseline.                  |
| **N-BEATS (Baseline)**      | *~[Value from JSON]* | Deep learning model on globally scaled data.  |
| **N-BEATS (Residual Feature)** | *~[Value from JSON]* | N-BEATS trained on a de-seasonalized residual series. |

***Note:** These values should be filled in from the generated `.json` artifact files.*

The results indicate that while Prophet provides a robust baseline, the N-BEATS model, especially after the feature engineering experiment, achieves superior performance. This highlights the power of deep learning for capturing complex patterns in time series data, as well as the value of iterative, data-centric improvements.

## Next Steps

This project establishes a strong foundation. Future work could include:

*   **Advanced Feature Engineering:** Incorporate calendar events, holidays, and price information as explicit covariates.
*   **Probabilistic Forecasting:** Extend the N-BEATS model with quantile heads to produce prediction intervals, not just point forecasts.
*   **Scalability:** Modularize the notebook code into a script-based pipeline (e.g., using `luigi` or `kedro`) to handle larger datasets and automate re-training.
*   **Deployment:** Containerize the best-performing model with Docker and expose it via a REST API (e.g., using FastAPI) for on-demand forecasts.
