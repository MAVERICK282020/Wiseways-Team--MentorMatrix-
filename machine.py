from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

# 1. NVIDIA AI Mentor Setup
client = OpenAI( 
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "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"
)

# 2. Data Cleaning Utility
def clean_cols(name):
    return re.sub(r'[^a-zA-Z0-9\s]', '', name).strip()

# 3. Load and Process Dataset
try:
    df_raw = pd.read_csv("Book4.csv", encoding='latin1', engine='python')
    df_raw = pd.read_csv("Book3.csv", encoding='latin1', engine='python')
    df_raw = pd.read_csv("Book2.csv", encoding='latin1', engine='python')
    df_raw = pd.read_csv("Book1.csv", encoding='latin1', engine='python')
    df_raw = pd.read_csv("uptac2.csv", encoding='latin1', engine='python')
    df_raw = pd.read_csv("JEE_Rank_2016_2024.csv", encoding='latin1', engine='python')
    df_raw.columns = [clean_cols(col) for col in df_raw.columns]
    
    mapping = {
        'Institute': 'college',
        'Program': 'branch',
        'Opening Rank': 'opening_rank',
        'Closing Rank': 'closing_rank'
    }
    df = df_raw.rename(columns=mapping)
    df = df[['college', 'branch', 'opening_rank', 'closing_rank']]

    # Rank cleaning
    df['opening_rank'] = pd.to_numeric(df['opening_rank'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
    df['closing_rank'] = pd.to_numeric(df['closing_rank'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')

    df = df.dropna()

    # --- CS FIX: REMOVE DUPLICATES ---
    # Ek hi college aur branch ki multiple entries (Rounds) ko hata kar sirf last entry rakhenge
    df = df.drop_duplicates(subset=['college', 'branch'], keep='last')

    # Branch Encoding
    df['branch_code'] = df['branch'].astype('category').cat.codes
    reverse_branch_mapping = dict(enumerate(df['branch'].astype('category').cat.categories))
    branch_mapping = {v: k for k, v in reverse_branch_mapping.items()}

    print(f"✅ Dataset Loaded: {len(df)} Unique College-Branch Pairs Found.")

except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    df = pd.DataFrame()

# 4. Model Training
if not df.empty:
    X = df[['opening_rank', 'branch_code']]
    y = df['closing_rank']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("✅ ML Model Trained and Ready")
else:
    model = None

# --- API ROUTES ---

@app.route("/")
def home():
    return "WiseWays Machine Engine is Running 🚀"

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    query = data.get("query", "")
    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": query}],
            temperature=0.5
        )
        return jsonify({"response": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    try:
        rank = int(data.get("rank", 0))
        branch_input = data.get("branch", "").strip()
        
        temp_df = df.copy()
        
        # 5. Recommendation logic using Model + Distance
        if branch_input and branch_input in branch_mapping:
            b_code = branch_mapping[branch_input]
            pred = model.predict([[rank, b_code]])[0]
            temp_df['diff'] = abs(temp_df['closing_rank'] - pred)
        else:
            # Agar branch nahi di, toh seedha rank difference dekhenge
            temp_df['diff'] = abs(temp_df['closing_rank'] - rank)

        # Sort by proximity and take top 5
        results = temp_df.sort_values('diff').head(5).to_dict(orient="records")
        return jsonify({"colleges": results})
    except Exception as e:
        print(f"Recommend Error: {e}")
        return jsonify({"error": "Invalid Input"}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)