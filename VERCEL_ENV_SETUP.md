# Vercel Environment Variables Setup

## Frontend Environment Variables

In your Vercel project dashboard, you need to set these environment variables:

### Required Variables

1. **VITE_API_BASE_URL**
   - Value: `https://unmute-280906743817.asia-southeast1.run.app`
   - **IMPORTANT:** Must include `https://` protocol
   - **DO NOT** include trailing slash

2. **VITE_WS_BASE_URL**
   - Value: `wss://unmute-280906743817.asia-southeast1.run.app`
   - **IMPORTANT:** Use `wss://` (secure WebSocket) for HTTPS sites
   - **DO NOT** include trailing slash

## How to Set Environment Variables in Vercel

### Via Vercel Dashboard

1. Go to your project on [vercel.com](https://vercel.com)
2. Click on **Settings** tab
3. Click on **Environment Variables** in the left sidebar
4. Add each variable:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://unmute-280906743817.asia-southeast1.run.app`
   - **Environments**: Select all (Production, Preview, Development)
5. Click **Save**
6. Repeat for `VITE_WS_BASE_URL`

### Via Vercel CLI

```bash
vercel env add VITE_API_BASE_URL
# When prompted, enter: https://unmute-280906743817.asia-southeast1.run.app
# Select: Production, Preview, Development

vercel env add VITE_WS_BASE_URL
# When prompted, enter: wss://unmute-280906743817.asia-southeast1.run.app
# Select: Production, Preview, Development
```

## Verification

After setting the variables, you must **redeploy** your application for the changes to take effect:

```bash
# Trigger a new deployment
git commit --allow-empty -m "Update environment variables"
git push
```

Or use the Vercel dashboard:
1. Go to **Deployments** tab
2. Click the **three dots** on the latest deployment
3. Select **Redeploy**

## Troubleshooting

### Issue: GIF URLs are malformed

**Symptom:** Console shows errors like:
```
Failed to load resource: net::ERR_NAME_NOT_RESOLVED
unmute-280906743817.asia-southeast1.run.apphttps//storage.googleapis.com/...
```

**Cause:** `VITE_API_BASE_URL` is missing the `https://` protocol

**Fix:**
1. Update `VITE_API_BASE_URL` to include `https://`
2. Redeploy the application

### Issue: WebSocket connection fails with "Mixed Content" error

**Symptom:** Console shows:
```
Mixed Content: The page at 'https://...' was loaded over HTTPS, 
but attempted to connect to the insecure WebSocket endpoint 'ws://...'
```

**Cause:** `VITE_WS_BASE_URL` is using `ws://` instead of `wss://`

**Fix:**
1. Update `VITE_WS_BASE_URL` to use `wss://` (not `ws://`)
2. Redeploy the application

### Checking Current Environment Variables

To see what environment variables are currently set:

```bash
vercel env ls
```

### Removing Old/Wrong Variables

If you need to remove a variable:

```bash
vercel env rm VITE_API_BASE_URL production
```

Or use the Vercel dashboard to delete it.

## Backend Environment Variables on Cloud Run

Make sure your Cloud Run service also has the correct environment variables:

```bash
gcloud run services update unmute-backend \
  --set-env-vars="USE_GCS=true,GCS_BUCKET_NAME=unmute-datasets" \
  --region=asia-southeast1
```

Verify with:
```bash
gcloud run services describe unmute-backend \
  --region=asia-southeast1 \
  --format='value(spec.template.spec.containers[0].env)'
```

## Complete Checklist

- [ ] `VITE_API_BASE_URL` set to `https://unmute-280906743817.asia-southeast1.run.app`
- [ ] `VITE_WS_BASE_URL` set to `wss://unmute-280906743817.asia-southeast1.run.app`
- [ ] Both variables applied to Production, Preview, and Development
- [ ] Application redeployed after setting variables
- [ ] Backend `USE_GCS=true` on Cloud Run
- [ ] Backend `GCS_BUCKET_NAME=unmute-datasets` on Cloud Run
- [ ] GCS bucket has public access configured
- [ ] GCS bucket has CORS configured for your Vercel domain

## Testing

After deployment, check the browser console to verify URLs are correct:

**Good URLs:**
```
https://storage.googleapis.com/unmute-datasets/sgsl_dataset/want/want.gif?t=...
```

**Bad URLs (missing protocol):**
```
unmute-280906743817.asia-southeast1.run.apphttps//storage.googleapis.com/...
```

**Bad URLs (wrong protocol):**
```
http://unmute-280906743817.asia-southeast1.run.app/static/sgsl_dataset/want/want.gif
```
