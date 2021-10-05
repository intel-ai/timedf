import s3fs


class S3Client:
    s3_aws_com = ".s3.amazonaws.com"

    def __init__(self):
        self.fs = s3fs.S3FileSystem(anon=True)

    @classmethod
    def s3like(cls, filename: str):
        if filename.startswith("s3://"):
            return True
        elif filename.startswith("https://"):
            if cls.s3_aws_com not in filename:
                return False
            return True
        else:
            return False

    def _prepare_s3_link(self, https_link: str):
        filename = https_link.replace("https://", "")
        filename = filename.replace(self.s3_aws_com, "")
        bucket_name = filename.split("/")[0]
        return bucket_name, filename

    def getsize(self, filename: str):
        if filename.startswith("https://"):
            _, filename = self._prepare_s3_link(filename)
        return self.fs.info(filename)["Size"]

    def glob(self, files_pattern: str):
        if files_pattern.startswith("https://"):
            bucket_name, s3_files_pattern = self._prepare_s3_link(files_pattern)
            return [
                f"https://{filename.replace(bucket_name, bucket_name+self.s3_aws_com)}"
                for filename in self.fs.glob(s3_files_pattern)
            ]
        else:
            return [f"s3://{filename}" for filename in self.fs.glob(files_pattern)]

    def du(self, start_path: str):
        if start_path.startswith("https://"):
            _, start_path = self._prepare_s3_link(start_path)
        return s3_client.fs.du(start_path) / 1024 / 1024


s3_client = S3Client()
