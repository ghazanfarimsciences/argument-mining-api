# Argument Mining API - Changes Documentation

## Overview
This document tracks all changes made to the argument-mining-api repository during the setup and development process.

## Initial Setup Changes

### 1. Environment Configuration
**File:** `app/argmining/config.py`
- **Purpose:** Loads environment variables for API keys
- **Changes:** No modifications needed - existing configuration was correct
- **Key Variables:**
  - `OPENAI_KEY`: OpenAI API key for model access
  - `HF_TOKEN`: Hugging Face token for model access

### 2. CORS Configuration
**File:** `app/main.py`
**Changes Made:**
```python
# Added CORS middleware to allow frontend requests
from fastapi.middleware.cors import CORSMiddleware

# Added CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
**Purpose:** Enable frontend to communicate with backend API

### 3. API Endpoint Structure
**File:** `app/api/routes/chat.py`
**Existing Structure:**
- **Endpoint:** `POST /chat/send`
- **Request Format:**
  ```json
  {
    "session_id": "string",
    "model": "modernbert|openai|tinyllama",
    "message": "string"
  }
  ```
- **Response Format:**
  ```json
  {
    "message": "string",
    "session_id": "string", 
    "model": "string",
    "output": {
      "original_text": "string",
      "claims": [{"id": "string", "text": "string"}],
      "premises": [{"id": "string", "text": "string"}],
      "stance_relations": [{"claim_id": "string", "premise_id": "string", "stance": "string"}]
    }
  }
  ```

### 4. Dependencies
**File:** `requirements.txt`
**Changes Made:**
```diff
- numpy==2.3.1
+ numpy==1.26.4
```
**Reason:** Fixed compatibility issue with Python 3.10.12

## Backend Features

### 1. Model Support
- **modernbert**: Hugging Face model for argument mining
- **openai**: OpenAI API for argument mining  
- **tinyllama**: Lightweight model for argument mining

### 2. Argument Mining Pipeline
**File:** `app/api/services/model_client.py`
**Functionality:**
- ADU (Argument Discourse Unit) classification
- Claim and premise identification
- Stance relation detection
- Structured output generation

### 3. Error Handling
- Proper HTTP status codes
- Detailed error messages for debugging
- Graceful handling of API key issues

## Server Configuration

### 1. Development Server
**Command:** `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`
**Features:**
- Hot reload for development
- CORS enabled
- Health check endpoint at `/health/`

### 2. Environment Variables
**Required:**
```bash
export OPEN_AI_KEY="your_openai_api_key"
export HF_TOKEN="your_huggingface_token"
```

## API Testing

### 1. Health Check
```bash
curl -X GET http://127.0.0.1:8000/health/
```

### 2. Chat Endpoint Test
```bash
curl -X POST http://127.0.0.1:8000/chat/send \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test123",
    "model": "openai", 
    "message": "Climate change is caused by human activities."
  }'
```

## Issues Resolved

### 1. API Key Configuration
- **Issue:** Backend was using test keys causing 401 errors
- **Solution:** Set real API keys in environment variables
- **Result:** Successful model processing and response generation

### 2. CORS Issues
- **Issue:** Frontend couldn't communicate with backend due to CORS
- **Solution:** Added CORS middleware with proper origins
- **Result:** Seamless frontend-backend communication

### 3. Dependency Compatibility
- **Issue:** numpy version 2.3.1 incompatible with Python 3.10.12
- **Solution:** Downgraded to numpy 1.26.4
- **Result:** Successful dependency installation

## Current Status
- ✅ Backend server running on port 8000
- ✅ CORS properly configured
- ✅ All three models (modernbert, openai, tinyllama) supported
- ✅ Proper error handling and logging
- ✅ Structured argument mining output
- ✅ Health check endpoint working

## Notes
- Backend architecture was not modified as per user requirements
- Only configuration and CORS changes were made
- All existing functionality preserved
- API response format maintained for frontend compatibility 