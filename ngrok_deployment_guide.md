
# Ngrok Deployment Guide for 3D Pose Estimation API

## Overview
This guide will help you deploy your FastAPI application using ngrok for remote access and testing.

## Prerequisites
1. Install ngrok: https://ngrok.com/download
2. Your FastAPI application running on port 8000
3. Python dependencies installed

## Step 1: Start Your FastAPI Server

```bash
# Navigate to your project directory
cd /home/minhdao/projects/Word/NCKH/PoseEstimation3D

# Start the FastAPI server
python main_api.py
```

Your API should now be running on `http://127.0.0.1:8000`

## Step 2: Test Local API

Before using ngrok, test your API locally:

```bash
# Test the CORS configuration
python test_cors.py

# Or access the API documentation
# Open http://127.0.0.1:8000/docs in your browser
```

## Step 3: Start Ngrok

### Option A: Basic Ngrok Setup
```bash
# Start ngrok for port 8000
ngrok http 8000
```

### Option B: Advanced Ngrok Setup with Custom Subdomain
```bash
# Start ngrok with custom subdomain (if you have ngrok+ account)
ngrok http 8000 --subdomain=your-custom-subdomain
```

### Option C: Ngrok Configuration File
Create a `ngrok.yml` file:
```yaml
authtoken: your-ngrok-auth-token
tunnels:
  api:
    addr: 8000
    proto: http
    subdomain: your-custom-subdomain
```

Then run:
```bash
ngrok start --all
```

## Step 4: Access Your API

When ngrok starts, it will show you a URL like:
- `https://your-random-string.ngrok.io` (public URL)
- `http://127.0.0.1:4040` (ngrok dashboard)

## Step 5: Test with Ngrok URL

Use the ngrok URL to test your API:

```bash
# Test with ngrok URL
curl -X GET "https://your-random-string.ngrok.io/" \
     -H "accept: application/json" \
     -H "Origin: http://localhost:3000"

# Or test with Python
python test_cors.py
```

## CORS Configuration for Ngrok

Your API is already configured to handle CORS properly:

```python
# CORS Origins include:
- "http://localhost:3000"      # Local React/Vue/Angular app
- "http://127.0.0.1:3000"      # Local React/Vue/Angular app
- "http://localhost:8000"      # Local API server
- "http://127.0.0.1:8000"      # Local API server
- "*"                          # Allow all origins (ngrok and testing)
```

## Common Issues and Solutions

### 1. CORS Errors
**Problem**: `Access-Control-Allow-Origin` header missing
**Solution**:
- Ensure your API is running with the updated CORS configuration
- Check that your client is using the correct ngrok URL
- Verify that the `Origin` header matches one of the allowed origins

### 2. Ngrok Connection Issues
**Problem**: Ngrok fails to connect to your local server
**Solution**:
- Make sure your FastAPI server is running on port 8000
- Check if port 8000 is available (not used by another application)
- Try restarting ngrok: `ngrok http 8000`

### 3. File Upload Issues
**Problem**: File uploads fail through ngrok
**Solution**:
- Ngrok has file size limitations (usually 500MB for free version)
- Check that your video files are within acceptable size limits
- Ensure the `uploads` directory exists and has write permissions

### 4. Performance Issues
**Problem**: Slow response times through ngrok
**Solution**:
- Ngrok adds some latency due to the tunnel
- For production use, consider a proper deployment solution
- Optimize your API processing pipeline

## Testing Your API with Ngrok

### 1. API Documentation
Access your API documentation at:
```
https://your-random-string.ngrok.io/docs
```

### 2. Test Endpoints
```bash
# Test root endpoint
curl -X GET "https://your-random-string.ngrok.io/" \
     -H "accept: application/json"

# Test pose comparison endpoint
curl -X POST "https://your-random-string.ngrok.io/api/compare_pose" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "user_image": "base64_encoded_image",
       "reference_video_path": "res/input/video.mp4",
       "reference_frame_index": 100
     }'
```

### 3. Frontend Integration
When integrating with a frontend application, use the ngrok URL:

```javascript
// Example React/Vue/Angular API call
const response = await fetch('https://your-random-string.ngrok.io/api/compare_pose', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    user_image: base64Image,
    reference_video_path: 'res/input/video.mp4',
    reference_frame_index: 100
  })
});
```

## Production Deployment Considerations

For production deployment, consider these alternatives to ngrok:

1. **Cloud Services**: AWS, Google Cloud, Azure
2. **Platform as Service**: Heroku, Render, Vercel
3. **Docker Containers**: Docker, Kubernetes
4. **Traditional Hosting**: Nginx, Apache

## Troubleshooting Checklist

1. ✅ FastAPI server is running on port 8000
2. ✅ CORS configuration is properly set
3. ✅ Ngrok is properly installed and authenticated
4. ✅ File permissions are set correctly
5. ✅ Dependencies are installed
6. ✅ Test endpoints work locally before using ngrok
7. ✅ Ngrok URL is used in client applications

## Summary

Your FastAPI application is now properly configured for CORS and ready for ngrok deployment:

1. **Start your API**: `python main_api.py`
2. **Start ngrok**: `ngrok http 8000`
3. **Use the ngrok URL** for testing and development
4. **Test with**: `python test_cors.py`

The CORS configuration allows requests from localhost, your frontend development server, and any ngrok URLs, making it perfect for development and testing scenarios.