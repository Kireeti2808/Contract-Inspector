# listener.py
from fastapi import FastAPI, Request, HTTPException
import subprocess
import hmac
import hashlib
import os

# 🔑 IMPORTANT: Set this as a secret in your deployment environment!
# This secret must MATCH the one you set in the GitHub webhook settings.
from dotenv import load_dotenv
load_dotenv()
WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")

app = FastAPI()

@app.post("/webhook")
async def receive_github_webhook(request: Request):
    """
    Listens for GitHub webhooks, verifies them, and triggers the index build.
    """
    # 1. Verify the signature
    signature = request.headers.get('X-Hub-Signature-256')
    if not signature:
        raise HTTPException(status_code=403, detail="X-Hub-Signature-256 header is missing!")

    if not WEBHOOK_SECRET:
         raise HTTPException(status_code=500, detail="Webhook secret not configured on server.")

    body = await request.body()
    expected_signature = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=403, detail="Request signature does not match!")

    # 2. Pull the latest changes from the repository
    print("Signature verified. Pulling latest changes from Git...")
    try:
        subprocess.run(["git", "fetch", "--all"], check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=True) # Or your branch name
    except subprocess.CalledProcessError as e:
        print(f"Error pulling from Git: {e}")
        raise HTTPException(status_code=500, detail="Failed to pull from Git.")

    # 3. Run the index-building script
    print("Git pull successful. Running build_index.py...")
    try:
        # We run this in a non-blocking way, but for simplicity here, it's blocking
        subprocess.run(["python", "build_index.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running build_index.py: {e}")
        raise HTTPException(status_code=500, detail="Index build script failed.")

    print("✅ Webhook processed successfully. Index is being updated.")
    return {"status": "success"}
