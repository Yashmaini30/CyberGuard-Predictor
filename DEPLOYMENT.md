# CyberGuard Predictor - Deployment Guide

## 🚀 FREE HOSTING OPTIONS

### 1. 🏆 RENDER.COM (RECOMMENDED)

**Steps:**
1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Click "New Web Service"
4. Connect your GitHub repo: `CyberGuard-Predictor`
5. Configure:
   - **Name:** cyberguard-predictor
   - **Region:** Singapore (closest to India)
   - **Branch:** main
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`

**Environment Variables:**
```
DAGSHUB_AUTH_TOKEN=your_token
DAGSHUB_USERNAME=mainiyash2
DAGSHUB_REPO=CyberGuard-Predictor
```

**Your live URL:** `https://cyberguard-predictor.onrender.com`

---

### 2. 🥈 RAILWAY.APP

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Deploy from GitHub
3. Select `CyberGuard-Predictor` repo
4. Auto-deploys with zero config!

**Your live URL:** `https://cyberguard-predictor-production.up.railway.app`

---

### 3. 🥉 HUGGING FACE SPACES

**Steps:**
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space
3. Choose "Docker" runtime
4. Upload your code
5. It automatically deploys!

**Your live URL:** `https://mainiyash2-cyberguard-predictor.hf.space`

---

## 📱 DEMO LINKS (Example URLs)

- **Main Dashboard:** `https://your-app.onrender.com/`
- **API Documentation:** `https://your-app.onrender.com/docs`
- **Risk Heatmap API:** `https://your-app.onrender.com/risk-heatmap`
- **Prediction API:** `https://your-app.onrender.com/predict-withdrawal-location`

---

## 🎯 FOR SIH 2024 SUBMISSION

1. **Deploy on Render.com** (most reliable)
2. **Create demo video** showing:
   - Dashboard with India map
   - Real-time prediction
   - Risk heatmap
   - API endpoints working
3. **Share live URL** in your SIH submission

---

## 💡 TIPS

- **No database needed** - App works in offline mode
- **Models included** - All ML models are in the repo
- **Free tier limits** - Render sleeps after 15 min inactivity
- **Custom domain** - Can add your own domain later

Ready to deploy? Choose **Render.com** and follow the steps above! 🚀