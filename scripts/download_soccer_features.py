from SoccerNet.Downloader import SoccerNetDownloader

def download_soccer_data():
    downloader = SoccerNetDownloader(LocalDirectory="data/features")
    downloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train", "valid", "test"]) # download Features reduced with PCA

if __name__ == "__main__":
    download_soccer_data()
