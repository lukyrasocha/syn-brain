import gdown
import zipfile
from pathlib import Path
from loguru import logger

def download_extract_dataset(url, raw_data_path):
    raw_data_path = Path(raw_data_path)
    raw_data_path.mkdir(parents=True, exist_ok=True)

    if any(raw_data_path.iterdir()):
        logger.success(f"✅ Using existing data in '{raw_data_path}'")
        return

    file_path = raw_data_path / "downloaded_file"

    logger.info(f"Downloading dataset from Google Drive...")
    gdown.download(url, str(file_path), quiet=False)

    logger.success(f"✅ Downloaded to {file_path.name}")

    zipfile.ZipFile(file_path).extractall(raw_data_path)

    file_path.unlink()  # delete the downloaded archive
    logger.success("✅ Dataset extracted and ready")


    
if __name__ == "__main__":
    #DATASET_URL = "https://drive.google.com/uc?id=1cpjv0UYWgk9kRSGeBFxdK9HxCekGMDv-" # dataset divided into classes
    DATASET_ULR = "https://drive.google.com/file/d/1eLOdNZ2--gAzWKtL3nV4FdVQw59XEEdh"
    RAW_DATA_PATH = Path("./data/raw")
    
    download_extract_dataset(DATASET_URL, RAW_DATA_PATH)
