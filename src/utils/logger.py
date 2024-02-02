import logging
import os
from src.utils.constants import log_file_path

if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)

logger = logging.getLogger("general")
logging.basicConfig(filename=os.path.join(log_file_path, 'myapp.log'), level=logging.INFO)
