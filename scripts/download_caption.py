from SoccerNet.Downloader import SoccerNetDownloader

def download_soccer_data():
    downloader = SoccerNetDownloader(LocalDirectory="data/caption")
    downloader.downloadDataTask(task="caption-2023", split=["train", "valid", "test", "challenge"])

if __name__ == "__main__":
    download_soccer_data()
