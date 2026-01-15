# Google Cloud Storage Setup for Datasets

This guide explains how to store `sgsl_dataset` and `sgsl_processed` on Google Cloud Storage (GCS) for production deployment.

## Prerequisites

1. Google Cloud account with a project
2. `gcloud` CLI installed and authenticated
3. `gsutil` command available

## Step 1: Create a GCS Bucket

```bash
# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Create a bucket in the same region as your Cloud Run service
gsutil mb -l asia-southeast1 gs://unmute-datasets

# Or use a different name
gsutil mb -l asia-southeast1 gs://your-bucket-name
```

## Step 2: Upload Datasets

```bash
# Upload both dataset directories (this may take a while)
gsutil -m cp -r sgsl_dataset gs://unmute-datasets/
gsutil -m cp -r sgsl_processed gs://unmute-datasets/

# Verify upload
gsutil ls gs://unmute-datasets/
```

## Step 3: Make Files Publicly Accessible

For the frontend to directly access GIF files, the bucket needs public read access:

```bash
# Make entire bucket publicly readable
gsutil iam ch allUsers:objectViewer gs://unmute-datasets

# Or set uniform bucket-level access first (recommended)
gsutil uniformbucketlevelaccess set on gs://unmute-datasets
gsutil iam ch allUsers:objectViewer gs://unmute-datasets
```

### Alternative: Use Signed URLs (More Secure)

If you don't want public access, you can modify the code to generate signed URLs. This requires service account credentials.

## Step 4: Configure Cloud Run Environment Variables

In your Cloud Run service, set these environment variables:

```
USE_GCS=true
GCS_BUCKET_NAME=unmute-datasets
```

### Using gcloud CLI:

```bash
gcloud run services update unmute-backend \
  --set-env-vars="USE_GCS=true,GCS_BUCKET_NAME=unmute-datasets" \
  --region=asia-southeast1
```

### Using Cloud Console:

1. Go to Cloud Run in Google Cloud Console
2. Select your service
3. Click "Edit & Deploy New Revision"
4. Go to "Variables & Secrets" tab
5. Add:
   - `USE_GCS` = `true`
   - `GCS_BUCKET_NAME` = `unmute-datasets`
6. Deploy

## Step 5: Set Up IAM (For Cloud Run)

Cloud Run needs permission to read from GCS. The default service account usually has this, but you can verify:

```bash
# Get the service account email
gcloud run services describe unmute-backend --region=asia-southeast1 --format='value(spec.template.spec.serviceAccountName)'

# Grant Storage Object Viewer role if needed
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_GCS` | Enable GCS storage (`true`/`false`) | `false` |
| `GCS_BUCKET_NAME` | Name of the GCS bucket | `unmute-datasets` |

## How It Works

When `USE_GCS=true`:

1. **Static files (GIFs)**: URLs point directly to GCS public URLs
   - Example: `https://storage.googleapis.com/unmute-datasets/sgsl_dataset/HELLO/HELLO.gif`

2. **Pickle files**: Downloaded from GCS on-demand by the backend
   - The backend uses `google-cloud-storage` Python library
   - Files are cached in memory using LRU cache

3. **vocab.json**: Loaded from GCS at startup

## Local Development

For local development, you have two options:

### Option 1: Use Local Files (Default)
Keep `USE_GCS=false` (or don't set it) and mount local directories:

```bash
# In docker-compose, volumes are already configured:
volumes:
  - ./sgsl_dataset:/app/sgsl_dataset:ro
  - ./sgsl_processed:/app/sgsl_processed:ro
```

### Option 2: Test with GCS
Set environment variables to test GCS integration locally:

```bash
export USE_GCS=true
export GCS_BUCKET_NAME=unmute-datasets
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

uvicorn backend.app:app --reload
```

## CORS Configuration for Frontend

Since GIFs are served directly from GCS, ensure CORS is configured on the bucket:

```bash
# Create cors.json file
cat > cors.json << 'EOF'
[
  {
    "origin": ["https://unmute-gamma.vercel.app", "http://localhost:5173"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type"],
    "maxAgeSeconds": 3600
  }
]
EOF

# Apply CORS configuration
gsutil cors set cors.json gs://unmute-datasets
```

## Troubleshooting

### "403 Forbidden" when accessing GCS files
- Check bucket permissions: `gsutil iam get gs://unmute-datasets`
- Ensure `allUsers:objectViewer` is set for public access

### "Could not authenticate" errors
- For Cloud Run: Check service account has `storage.objectViewer` role
- For local: Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

### Files not found in GCS
- Verify upload: `gsutil ls gs://unmute-datasets/sgsl_processed/`
- Check bucket name matches `GCS_BUCKET_NAME` env var

## Cost Considerations

GCS pricing includes:
- **Storage**: ~$0.02/GB/month for Standard storage
- **Operations**: ~$0.004 per 10,000 Class A operations (uploads)
- **Egress**: Free within same region, ~$0.12/GB to internet

For ~10GB of datasets with moderate traffic, expect ~$1-5/month.
