# California Housing Streamlit Trainer

This small Streamlit app trains several regression models on the California Housing dataset and saves the best model to `best_model.pkl`.

Setup
1. Create a virtual environment (recommended) and activate it.
2. Install dependencies:

   pip install -r requirements.txt

Run

   streamlit run streamlit_app.py

What it does
- Loads the scikit-learn California housing dataset
- Trains Linear Regression, Random Forest, and Gradient Boosting
- Shows metrics and lets you download the best model
