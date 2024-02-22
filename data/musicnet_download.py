import urllib.request
import shutil


DATASET_URL = "https://zenodo.org/records/5120004/files/musicnet.tar.gz?download=1"
METADATA_URL = "https://zenodo.org/records/5120004/files/musicnet_metadata.csv?download=1"
MIDI_URL = "https://zenodo.org/records/5120004/files/musicnet_midis.tar.gz?download=1"

if __name__ == '__main__':
    urllib.request.urlretrieve(DATASET_URL, "musicnet.tar.gz")
    shutil.unpack_archive("musicnet.tar.gz", ".")

    urllib.request.urlretrieve(METADATA_URL, "musicnet_metadata.csv")

    urllib.request.urlretrieve(MIDI_URL, "musicnet_midis.tar.gz")
    shutil.unpack_archive("musicnet_midis.tar.gz", ".")
