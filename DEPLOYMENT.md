# 🚀 Deployment Guide - DocsReview RAG + LLM

## Streamlit Cloud Deployment

This guide will help you deploy your DocsReview app to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: You need a GitHub account
2. **NVIDIA API Key**: Get your API key from [NVIDIA AI](https://nvidia.com/ai)
3. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## 🔧 Setup Steps

### Step 1: Prepare Your Repository

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: DocsReview RAG + LLM app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

### Step 2: Configure Secrets

1. **Go to your Streamlit Cloud dashboard**
2. **Click "New app"**
3. **Connect your GitHub repository**
4. **Add secrets** in the Streamlit Cloud interface:

   ```toml
   NVIDIA_API_KEY = "your_actual_nvidia_api_key_here"
   ```

### Step 3: Deploy Configuration

- **Main file path**: `app.py`
- **Python version**: 3.9+
- **Branch**: `main`

## 🌐 App URL

Once deployed, your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

## 🔍 Features After Deployment

✅ **Document Analysis**: Upload PDFs or enter URLs  
✅ **RAG + LLM Integration**: Intelligent document querying  
✅ **Executive Summaries**: Professional document summaries  
✅ **Chat Interface**: Ask specific questions about documents  
✅ **Multi-language Support**: Spanish and English  
✅ **OCR Support**: Extract text from scanned documents  

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are in `requirements.txt`
2. **API Key Issues**: Verify your NVIDIA API key is correct
3. **Memory Issues**: The app uses ~2GB RAM for embeddings
4. **Timeout**: Large documents may take time to process

### Performance Tips

- **Smaller Documents**: Process documents under 50 pages for best performance
- **Clear Cache**: Use the "🔄 Reiniciar RAG" button if issues occur
- **Monitor Usage**: Check your NVIDIA API usage limits

## 📊 Monitoring

- **App Status**: Check Streamlit Cloud dashboard
- **Logs**: View logs in Streamlit Cloud interface
- **Usage**: Monitor API calls and performance

## 🔒 Security Notes

- **API Keys**: Never commit API keys to GitHub
- **Secrets**: Use Streamlit Cloud secrets management
- **Data**: Documents are processed in memory only

## 🎯 Next Steps

After deployment, you can:

1. **Share the URL** with your team
2. **Embed in other websites** using iframe
3. **Create API endpoints** for integration
4. **Scale up** if needed

## 📞 Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify your API key
3. Test locally first
4. Check GitHub repository permissions

---

**Happy Deploying! 🚀**