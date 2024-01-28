import urllib.request
import shutil


DATASET_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"


if __name__ == '__main__':
    urllib.request.urlretrieve(DATASET_URL, "maestro-v3.0.0.zip")
    shutil.unpack_archive("maestro-v3.0.0.zip", ".")
