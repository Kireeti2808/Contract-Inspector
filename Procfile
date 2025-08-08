# This command starts your user-facing Streamlit application.
web: streamlit run app.py --server.port $PORT --server.enableCORS false

# This command starts your background webhook listener.
worker: uvicorn listener:app --host 0.0.0.0 --port 8001
