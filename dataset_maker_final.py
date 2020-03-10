from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import unicodecsv as csv


client_credentials_manager = SpotifyClientCredentials(client_id = 'afba44bd661346288a3c04fe82cb3b78', client_secret='568d765d7f2940139c01d7bb9fc4399b')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False

startYear = 1990
endYear = 2019
songsbyyearids = []
count = 4 # change count to 14
limit = 22
artist_ids = []
print('Size Per Year:' + str(count*limit))    
for year in range(startYear,endYear):#change year to 2019
    for c in range(0,count):
        results = sp.search(q='year:' + str(year), type='track', limit = limit, offset =c*limit)
        for i, t in enumerate(results['tracks']['items']):
            songsbyyearids.append(t['uri'])
            
print('Total no. Of Songs From Years:' + str(len(songsbyyearids)))
billboardsongids = []
firstArtistBillboardOccurence = {}
for year in range(startYear,endYear):
    # yearWAP = {}
    yearAIB = []
    try:
        billboardPlaylistID  = (sp.search(q = "TOP " + str(year) + " - Hot 100 Billboard", type = 'playlist', limit = 1)['playlists']['items'][0]['id'])
    except:
        continue
        print(year)
    uri = 'spotify:user:spotifycharts:playlist:' + billboardPlaylistID
    username = uri.split(':')[2]
    playlist_id = uri.split(':')[4]
    results = sp.user_playlist(username, playlist_id)
    for i in results['tracks']['items']:
        trackURI = i['track']['uri']
        trackID = i['track']['id']
        if(trackID != None):
            # print(trackURI)
            # print(trackID)
            albumInfo =  sp.track(trackURI)['album']
            trackArtistID = albumInfo['artists'][0]['id']
            
            if albumInfo['artists'][0]['name'] == 'Various Artists':
                continue
            
            billboardsongids.append(trackURI)
            
            if(albumInfo['release_date_precision'] == 'year'):
                trackReleaseYear = int(albumInfo['release_date'])
            elif(albumInfo['release_date_precision'] == 'day'):
                trackReleaseYear = int(albumInfo['release_date'][0:4])
            if trackArtistID in firstArtistBillboardOccurence:
                firstArtistBillboardOccurence[trackArtistID] = int(min(firstArtistBillboardOccurence[trackArtistID], trackReleaseYear))
            else:
                firstArtistBillboardOccurence[trackArtistID] = int(trackReleaseYear)

print('Total No. Of Songs From Billboard: ' + str(len(billboardsongids)))

# Removing Songs in Year ID which are repeated in Billboard IDs
for billboardsong in billboardsongids:
    if(billboardsong in songsbyyearids):
        songsbyyearids.remove(billboardsong)

print('Final No. Of Songs: ' + str(len(billboardsongids) + len(songsbyyearids)))

af = []
for tid in songsbyyearids:
    af.append(sp.audio_features(tid)[0])
    trackinfo = sp.track(tid)
    af[-1]['popularity'] = trackinfo['popularity']
    artistID = trackinfo['album']['artists'][0]['id']
    artistInfo = sp.artist(artistID)
    af[-1]['artist_name'] = artistInfo['name'] # Can be removed later
    af[-1]['artist_popularity'] = artistInfo['popularity']
    # af[-1]['name'] = trackinfo['name'] # if we want to include the name of the song
    af[-1]['billboard'] = 0
    if(trackinfo['album']['release_date_precision'] == 'year'):
        af[-1]['year'] = int(trackinfo['album']['release_date'])
    elif(trackinfo['album']['release_date_precision'] == 'day'):
        af[-1]['year'] = int(trackinfo['album']['release_date'][0:4])
    del af[-1]['track_href']
    del af[-1]['id'] # if we need to extract more data we didn't think of before
    del af[-1]['uri']
    del af[-1]['analysis_url']
    del af[-1]['mode']
    del af[-1]['time_signature']
    del af[-1]['type']
    del af[-1]['key']
    if(artistID in firstArtistBillboardOccurence.keys()):
        if int(af[-1]['year'])>firstArtistBillboardOccurence[artistID]:
            af[-1]['artistFeatured'] = 1
        else:
            af[-1]['artistFeatured'] = 0 
    else:
        af[-1]['artistFeatured'] = 0
    
print(af[0])
for tid in billboardsongids:
    try:
        # af.append(sp.audio_features(tid)[0])
        # trackinfo = sp.track(tid)
        # af[-1]['popularity'] = trackinfo['popularity']
        # af[-1]['artist_popularity'] = sp.artist(trackinfo['album']['artists'][0]['id'])['popularity']
        # af[-1]['artist_name'] = sp.artist(trackinfo['album']['artists'][0]['id'])['name'] # Can be removed later
        # # af[-1]['name'] = trackinfo['name'] # if we want to include the name of the song
        # af[-1]['billboard'] = 1
        # if(trackinfo['album']['release_date_precision'] == 'year'):
        #     af[-1]['year'] = trackinfo['album']['release_date']
        # elif(trackinfo['album']['release_date_precision'] == 'day'):
        #     af[-1]['year'] = trackinfo['album']['release_date'][0:4]
        # del af[-1]['track_href']
        # del af[-1]['id'] # if we need to extract more data we didn't think of before
        # del af[-1]['uri']
        # del af[-1]['analysis_url']
        # del af[-1]['mode']
        # del af[-1]['time_signature']
        # del af[-1]['type']
        # del af[-1]['key']

        af.append(sp.audio_features(tid)[0])
        trackinfo = sp.track(tid)
        af[-1]['popularity'] = trackinfo['popularity']
        artistID = trackinfo['album']['artists'][0]['id']
        artistInfo = sp.artist(artistID)
        af[-1]['artist_name'] = artistInfo['name'] # Can be removed later
        af[-1]['artist_popularity'] = artistInfo['popularity']
        # af[-1]['name'] = trackinfo['name'] # if we want to include the name of the song
        af[-1]['billboard'] = 1
        if(trackinfo['album']['release_date_precision'] == 'year'):
            af[-1]['year'] = int(trackinfo['album']['release_date'])
        else:
            af[-1]['year'] = int(trackinfo['album']['release_date'][0:4])
        del af[-1]['track_href']
        del af[-1]['id'] # if we need to extract more data we didn't think of before
        del af[-1]['uri']
        del af[-1]['analysis_url']
        del af[-1]['mode']
        del af[-1]['time_signature']
        del af[-1]['type']
        del af[-1]['key']
        if(artistID in firstArtistBillboardOccurence):
            af[-1]['artistFeatured'] = 1 if int(af[-1]['year'])>int(firstArtistBillboardOccurence[artistID]) else 0
        else:
            af[-1]['artistFeatured'] = 0    
    except Exception as e:
        print(tid + " scene")
        print(e)
        
print(af[0])

af.pop(3106)

toCSV = af
keys = toCSV[0].keys()
print
with open('people1.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(toCSV)