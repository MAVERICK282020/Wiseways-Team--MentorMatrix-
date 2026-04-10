WiseWays 
| AI-Powered Career & College Guidance Platform
WiseWays is a full-stack AI decision-making engine designed to help students navigate the complex landscape of college admissions and career planning. Unlike traditional portals that only provide rankings, WiseWays uses Machine Learning to predict college matches based on real admission data (JEE/UPTAC) and provides an AI Career Mentor for personalized guidance.

🚀 Key Features

College Prediction Engine: Uses a RandomForestRegressor model to predict admission possibilities based on entrance ranks, categories, and branch preferences.
AI Career Mentor: An integrated chatbot powered by NVIDIA’s Llama-3.1-8B to answer career doubts, branch comparisons, and industry trends.
Skill Roadmaps: Dynamic 4-year learning paths for recommended colleges to help students prepare for placements from Day 1.
Reality Score (ROI): Analyzes the trade-off between college fees and average placement packages to provide a "value-for-money" metric.
Voice Integration: Built-in Speech-to-Text (STT) and Text-to-Speech (TTS) for an accessible mentoring experience.

🛠️ Tech Stack

Backend
Framework: Flask
Machine Learning: Scikit-learn (Random Forest), Pandas, NumPy
LLM Integration: OpenAI SDK (NVIDIA NIM Endpoint for Llama-3.1)
Data Processing: Regular Expressions (Regex) for rank cleaning and data normalization.

Frontend
Styling: Tailwind CSS
Interactivity: Vanilla JavaScript (Async/Await API handling)
Icons/Fonts: Google Fonts (Inter), HeroIcons

📁 Project Structure
machine.py: The main Flask server containing ML recommendation logic and AI mentor API.
model.py: Dedicated script for training the Random Forest model and handling branch encoding.
online.html: The complete responsive frontend dashboard.
JEE_Rank_2016_2024.csv / uptac2.csv: Datasets containing historical opening and closing ranks.
