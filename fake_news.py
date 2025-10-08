import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import json
import requests
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import time # Import time for spinner (optional, but good practice)

# NOTE ON SECRETS: The app will now read secrets from st.secrets, 
# assuming you removed the [vars] header from secrets.toml.

# Load environment variables (kept for local testing consistency, ignored by Streamlit Cloud)
load_dotenv() 

# Validate environment variables (CRITICAL: Using st.secrets for Streamlit Cloud)
# Since you removed [vars], access is direct: st.secrets.KEY
try:
    GROQ_API_KEY = st.secrets.GROQ_API_KEY 
    GOOGLE_API_KEY = st.secrets.GOOGLE_API_KEY
    CSE_ID = st.secrets.CSE_ID
except AttributeError:
    st.error("⚠️ Streamlit Secrets are missing. Please ensure GROQ_API_KEY, GOOGLE_API_KEY, and CSE_ID are in your Streamlit Cloud secrets configuration.")
    st.stop()

# --- START: FINAL ROBUST CLIENT INITIALIZATION FIX ---
# This complex try/except block resolves the final 'proxies' TypeError 
# caused by Streamlit Cloud's deployment environment.
client = None
try:
    # First attempt: standard initialization (might fail due to proxies)
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
except TypeError as e:
    # If it fails with a TypeError, it's almost certainly the 'proxies' issue.
    if 'proxies' in str(e):
        # Clear the conflicting environment variables
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if 'https_proxy' in os.environ:
            del os.environ['https_proxy']
        
        # Second attempt to initialize the client after cleaning the environment
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        # If it's another TypeError, raise the original error
        raise e
except Exception as e:
    st.error(f"FATAL CLIENT INITIALIZATION ERROR: {e}")
    st.stop()
# --- END: FINAL ROBUST CLIENT INITIALIZATION FIX ---


# Load dataset
@st.cache_data
def load_news_data():
    try:
        return pd.read_csv("news_data.csv")
    except FileNotFoundError:
        st.error("Error: 'news_data.csv' not found. Please ensure the file is in the same directory.")
        st.stop()

df = load_news_data()

# Search similar articles from dataset
def search_similar_articles(news_text, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = df['News'].tolist() + [news_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_indices = cosine_sim.argsort()[0, -top_n:][::-1]
    similar = df.iloc[top_indices][['News', 'Status']].to_dict(orient='records')
    return similar

# Google Custom Search API integration (Corrected: uses requests, no model call)
def google_search(query, num_results=3):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": CSE_ID,
        "num": num_results
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status() 
        search_results = response.json().get('items', [])
        
        formatted_results = []
        for item in search_results:
            formatted_results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link")
            })
        return formatted_results
        
    except requests.exceptions.RequestException as e:
        return f"Google Search Error: {e}"

# Define tools Groq can call
functions = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_articles",
            "description": "Finds the most relevant articles in the dataset for comparison",
            "parameters": {
                "type": "object",
                "properties": {
                    "news_text": {"type": "string", "description": "User provided news article"},
                    "top_n": {"type": "integer", "default": 5}
                },
                "required": ["news_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Searches the web for credible sources related to a news article",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query text, usually the article headline or summary"},
                    "num_results": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

# Input validation function
def is_valid_input(text):
    if not text or text.isspace():
        return False
    non_alphanumeric = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    total_chars = len(text.strip())
    if non_alphanumeric / total_chars > 0.5 or not re.search(r'[a-zA-Z0-9]', text):
        return False
    return True

# Streamlit UI
st.title("Fake News Detector with GEN AI")
user_input = st.text_area("Enter the news article you want to verify:")

if st.button("Analyze"):
    if not is_valid_input(user_input):
        st.error("⚠️ Enter proper input. The input contains invalid characters or is empty.")
        st.markdown("Please provide a meaningful news article with text, not just special characters or spaces.")
        st.stop()

    with st.spinner("Analyzing the news article..."):
        # Escape special characters 
        safe_input = html.escape(user_input).replace('"', '\\"')
        messages = [{"role": "user", "content": f"Is this news real or fake? {safe_input}"}]

        # Step 1: Call Groq with tools available
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=functions,
                tool_choice="auto"
            )
        except Exception as e:
            st.error(f"⚠️ Failed to call Groq API (Step 1): {e}")
            st.stop()

        # Check for tool calls (CRITICAL INDENTATION FIX START)
        tool_calls = response.choices[0].message.tool_calls

        if not tool_calls:
            # If no tools were called, the model is trying to respond directly 
            final_response = response 
            final_messages = messages
        else:
            # Tool(s) were called. Proceed with the tool execution logic.
            tool_responses = []

            for call in tool_calls:
                # Use a try block to safely parse the arguments
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    st.error("⚠️ Model returned malformed JSON arguments for tool call.")
                    st.stop()

                if call.function.name == "search_similar_articles":
                    result = search_similar_articles(**args)
                elif call.function.name == "google_search":
                    result = google_search(**args)
                else:
                    continue

                tool_responses.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": call.function.name,
                    "content": json.dumps(result)
                })

            # Combine all messages (user input, model's tool call, and tool results)
            final_messages = messages + [response.choices[0].message] + tool_responses

            # Step 2: Final Groq response with all tool evidence
            try:
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=final_messages + [
                        
                            {
                                "role": "user",
                                "content": f"""
                                You are an AI model designed to assess the authenticity of news articles.

                                Please analyze the following news article and respond in the structured JSON format below.

                                News Article:
                                {safe_input}

                                Respond strictly in the following JSON format:

                            {{
                                "verdict": "Real or Fake",
                                "confidence_score": "X%",
                                "justification": {{
                                    "key_indicators": [
                                    "...",
                                    "..."
                                    ],
                                    "model_analysis": "..."
                            }},
                                "credible_sources": [
                            {{
                                    "title": "...",
                                    "url": "...",
                                    "relevance": "..."
            }}
          ]
        }}
        """
                            }
                        ]
                    )
            except Exception as e:
                    st.error(f"⚠️ Failed to get final Groq response (Step 2): {e}")
                    st.stop()
        # (CRITICAL INDENTATION FIX END)

    # Display and Parse Results
    try:
        # Robust JSON parsing (handles extra text/markdown wrappers)
        raw_content = final_response.choices[0].message.content
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        
        if match:
            json_string = match.group(0)
        else:
            json_string = raw_content

        result_json = json.loads(json_string)
        
        verdict = result_json.get("verdict", "").strip().lower()
        confidence = result_json.get("confidence_score", "")
        justification = result_json.get("justification", {})
        credible_sources = result_json.get("credible_sources", [])

        # Map verdict to colored string for Streamlit markdown
        if verdict == "fake":
            verdict_str = ":red[FAKE]"
        elif verdict == "real":
            verdict_str = ":green[REAL]"
        else:
            verdict_str = ":orange[UNKNOWN]"

        # Display results
        st.subheader("🧠 AI Verdict")
        st.markdown(f"### Verdict: {verdict_str}")
        st.markdown(f"**Confidence Score:** {confidence}")

        st.subheader("🔍 Justification")
        if "key_indicators" in justification:
            st.write("**Key Indicators:**")
            for indicator in justification["key_indicators"]:
                st.markdown(f"- {indicator}")
        if "model_analysis" in justification:
            st.write("**Model Analysis:**")
            st.markdown(justification["model_analysis"])

        st.subheader("🌐 Credible Sources")
        if credible_sources:
            for source in credible_sources:
                title = source.get("title", "No title")
                url = source.get("url", "#")
                relevance = source.get("relevance", "")
                st.markdown(f"- [{title}]({url}) — {relevance}")

        st.markdown("---")

    except Exception as e:
        st.error("⚠️ Failed to parse the AI result. Please try again. (Model output may be malformed)")
        st.text(f"Raw AI Output: {final_response.choices[0].message.content}")
        st.text(f"Error: {e}")
