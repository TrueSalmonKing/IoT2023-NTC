import gdown


url = 'https://drive.google.com/drive/folders/1z6VMfFyl01E0AT1PO9Csx9AJNuxlAQDf'

gdown.download_folder(url, quiet=False, use_cookies=False)