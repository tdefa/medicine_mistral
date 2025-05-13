%%
import json
import os
from glob import glob
import pathlib
import tqdm
import requests
import pandas as pd
import time
from requests.exceptions import RequestException
from tenacity import retry, wait_exponential, retry_if_exception_type




__file__ = "/hackaton_medi/baseline/request_fda.py"

if __name__ == "__main__":
    SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
    ROOT_DIR = SCRIPT_DIR.parent