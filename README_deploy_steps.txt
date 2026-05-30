Deployment files for Streamlit Cloud.

Upload these files to the root of your GitHub repository:
1. app.py
2. GBDT.pkl
3. requirements.txt

In Streamlit Cloud:
- Main file path: app.py
- Recommended Python version: 3.11 or 3.12

This version fixes:
- SHAP beeswarm feature display order
- Hidden leading/trailing spaces in model feature names
- Model prediction still uses original feature names saved in GBDT.pkl
