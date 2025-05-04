import os
import requests

def download_font():
    font_path = r'C:\Users\zyq\AppData\Roaming\Ultralytics\Arial.ttf'
    if not os.path.exists(font_path):
        print("Downloading Arial.ttf...")
        # 使用备用下载链接
        font_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/Arial.ttf"
        try:
            response = requests.get(font_url)
            response.raise_for_status()
            with open(font_path, 'wb') as f:
                f.write(response.content)
            print(f"Font downloaded successfully to {font_path}")
        except Exception as e:
            print(f"Error downloading font: {e}")
    else:
        print("Arial.ttf already exists.")

if __name__ == "__main__":
    download_font() 