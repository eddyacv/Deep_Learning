from SoccerNet.Downloader import SoccerNetDownloader

def download_soccer_data():
    downloader = SoccerNetDownloader(LocalDirectory="data/raw")
    downloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])

if __name__ == "__main__":
    download_soccer_data()
