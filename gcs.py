# upload_to_gcs.py
from google.cloud import storage
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcs-key.json'

client = storage.Client()
bucket = client.bucket('unmute-datasets')

def upload_directory(local_path, gcs_path):
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            gcs_file = os.path.join(gcs_path, relative_path).replace('\\', '/')
            
            blob = bucket.blob(gcs_file)
            blob.upload_from_filename(local_file)
            print(f'Uploaded {local_file} to {gcs_file}')

upload_directory('sgsl_dataset', 'sgsl_dataset')