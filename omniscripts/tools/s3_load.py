from pathlib import Path
import re

import boto3
from botocore import UNSIGNED
from botocore.config import Config


def download_folder(bucket_name, s3_folder_path, local_dir, reload, pattern=".*"):
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(bucket_name)
    local_dir = Path(local_dir)

    compiled = re.compile(pattern)
    print(s3_folder_path, bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder_path):
        source = obj.key
        target = Path(local_dir) / Path(source).relative_to(s3_folder_path)

        print(f'Processing "{source}"...')
        if re.match(compiled, source) is None:
            print(f'Skipping "{source}", not matching "{pattern}"')
        elif target.exists() and not reload:
            print(f'Skipping "{source}", already exists locally')
        else:
            print(f'Loading "{source}" from S3 bucket "{bucket_name}"...')
            target.parent.mkdir(parents=True, exist_ok=True)
            bucket.download_file(source, str(target))
            print(f'Downloaded "{source}" from S3 bucket "{bucket_name}"')

    print(f'Done loading "s3://{bucket_name}/{s3_folder_path}"')
