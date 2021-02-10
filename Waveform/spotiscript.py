import spotipy
from tqdm import tqdm
import spoticonfig as cfg

#These are spotify authentication constants, which are stored in my private config file
CLIENT_ID = cfg.SPOTI_IDS["CLIENT_ID"]
CLIENT_SECRET = cfg.SPOTI_IDS["CLIENT_SECRET"]

#This is the authentication process
token = spotipy.oauth2.SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
cache_token = token.get_access_token(as_dict=False)
sp = spotipy.Spotify(cache_token)

#These are spotify playlist and artist URI-s
IRON_MAIDEN_URI = "spotify:artist:6mdiAmATAx73kdxrNrnlao"
SAKAL_URI = "spotify:artist:1G9Ij5rkolzKLw3aGyS2PQ"

THE_GREAT_BLUES_INSTRUMENTALS = "spotify:playlist:2kBZz17IAMPDPYvElRxvK7"
BLUES_GUITAR_INSTRUMENTALS = "spotify:playlist:3a54WQYSUPwjgGmfd4JIII"
BLUES_INSTRUMENTAL = "spotify:playlist:3IK20G5ifusF0vyUdU2Vj6"
INSTRUMENTAL_GUITART_BLUES_ROCK = "spotify:playlist:1vYJareH2coOMwhDzmixx7"
INSTRUMENTAL_ROCK = "spotify:playlist:2uhsnHgI4F2eFyvoMHY0GR"
CLASSIC_ROCK_INSTRUMENTALS = "spotify:playlist:7DNv3pBxYj3FDAGIl4QA0m"

CAUSA_SUI = "spotify:artist:1TAcaMoUlvLpTUzh18TzDY"
MY_SLEEPING_KARMA = "spotify:artist:4idYdwYKM9exGep2RkwHcE"
THE_RE_STONED = "spotify:artist:6bLvlTSqMrL6C2eFeAx0BY"


def get_all_preview_urls_of_artist(uri, album_count=50):
    urls = []
    artist_albums = sp.artist_albums(uri, limit=album_count)
    for album in artist_albums["items"]:
        tracks = sp.album_tracks(album["uri"])
        for track in tracks["items"]:
            if track["preview_url"] is not None:
                urls.append(track["preview_url"])

    return urls

def get_instrumental_songs_preview_urls_of_artist(uri, album_count=50):
    urls = []
    artist_albums = sp.artist_albums(uri, limit=album_count)
    for album in tqdm(artist_albums["items"]):
        tracks = sp.album_tracks(album["uri"])
        for track in tracks["items"]:
            if track["preview_url"] is not None and sp.audio_features(track["uri"])[0]["instrumentalness"] > 0.8:
                urls.append(track["preview_url"])

    return urls

def get_all_preview_urls_from_playlist(uri):
    urls = []
    tracks = sp.playlist_tracks(uri)
    for track in tracks["items"]:
        if track["track"]["preview_url"] is not None:
            urls.append(track["track"]["preview_url"])

    return urls


def download_dataset():
    urls = get_all_preview_urls_from_playlist(THE_GREAT_BLUES_INSTRUMENTALS) + get_all_preview_urls_from_playlist(BLUES_GUITAR_INSTRUMENTALS) + \
        get_all_preview_urls_from_playlist(BLUES_INSTRUMENTAL) + get_all_preview_urls_from_playlist(INSTRUMENTAL_GUITART_BLUES_ROCK) + \
        get_all_preview_urls_from_playlist(INSTRUMENTAL_ROCK) + get_all_preview_urls_from_playlist(CLASSIC_ROCK_INSTRUMENTALS) + \
           get_all_preview_urls_of_artist(THE_RE_STONED) + get_all_preview_urls_of_artist(CAUSA_SUI) + get_all_preview_urls_of_artist(MY_SLEEPING_KARMA)

    return urls
