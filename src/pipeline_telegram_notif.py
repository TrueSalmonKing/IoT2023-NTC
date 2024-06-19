import json
import time
import sys
import subprocess
import os

import requests
import traceback
import os

class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, message):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message}
        response = requests.post(url, data=data)
        return response.json()

    def notify_error(self, error_message):
        self.send_message(f"Error occurred:\n{error_message}\n\nTraceback:\n{traceback.format_exc()}")

    def upload_file(self, file_path, caption=None):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"
        files = {"document": open(file_path, "rb")}
        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption
        response = requests.post(url, files=files, data=data)
        return response.json()

    def upload_pdf_files(self, folder_path):
        pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]

        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            self.upload_file(file_path, caption=f"PDF: {pdf_file}")


def run_python_script(script_path):
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_script.py <script_path>")
        sys.exit(1)
    
    script_path = sys.argv[1]


    config_file = "../telegram_config"
    if not os.path.exists(config_file):
        print(
            """Please include the telegram configuration file containing the bot token and the chat id in JSON form:\n
            {
                "bot_token": "",
                "chat_id": ""
            }
            """
        )
        exit(0)
    with open(config_file, "r") as f:
        config = json.load(f)

    notifier = TelegramNotifier(config["bot_token"], config["chat_id"])

    try:
        start_time = time.time()
        run_python_script(script_path)
        execution_time = time.time() - start_time
        notifier.send_message(f"Script ran by {os.getlogin()} execution finished successfully in {execution_time} seconds.")
        #notifier.upload_pdf_files('../figures/')
    except Exception as e:
        notifier.notify_error(str(e))