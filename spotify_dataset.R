library(spotifyr)
library(tidyverse)

get_track_audio_features_over_100 <- function(ids) {
  
  ## spotifyr limits get_track_audio_features to 100 at a time
  ## this function loops through the full id list
  
  ids <- ids[!is.na(ids)]
  len <- length(ids)
  repetitions <- floor(len/100) * 100
  intervals <- c(seq(from = 0, to = repetitions, by = 100), len)
  
  features <- data.frame()
  for(r in seq_along(intervals)){
    start <- intervals[r]
    end <- intervals[r + 1] - 1
    if(is.na(end)) break
    
    inner_features <- get_track_audio_features(ids = ids[start:end])
    features <- rbind(features, inner_features)
    
  }
  
  return(features)
  
}

access_token <- get_spotify_access_token(client_id = Sys.getenv('SPOTIFY_CLIENT_ID'),
                                         client_secret = Sys.getenv('SPOTIFY_CLIENT_SECRET'))

# Get a list of genre-specific playlists
genres <- c('pop', 'country', 'rap', 'hiphop', 'rock', 'dance')

playlist_ids <- NULL

for(g in genres){
  
  out <- search_spotify(q = g, type = 'playlist')
  out <- out %>% 
    select(name, id) %>%
    mutate(genre = g)
  
  playlist_ids <- rbind(playlist_ids, out)
  
}

# get the track ids
playlist_songs <- NULL

for(p in seq_along(playlist_ids$id)){
  
  out <- get_playlist_tracks(playlist_id = playlist_ids$id[p])
  
  out <- out %>%
    filter(!is.na(track.id)) %>%
    # separate out the df column artists
    unnest(cols = 'track.artists') %>%
    group_by(track.id) %>%
    mutate(row_number = 1:n(),
           track.artist = name) %>%
    ungroup() %>%
    filter(row_number == 1) %>%
    select(track.id, track.name, track.artist, track.popularity, track.album.id, track.album.name, track.album.release_date) %>%
    mutate(playlist_name = playlist_ids$name[p],
           playlist_id = playlist_ids$id[p],
           playlist_genre = playlist_ids$genre[p]) 
  
  playlist_songs <- rbind(playlist_songs, out)
  
}

# get track audio features
playlist_audio <- get_track_audio_features_over_100(ids = playlist_songs$track.id)

# combine
playlist_songs <- playlist_songs %>%
  left_join(select(playlist_audio, -track_href, -uri, -analysis_url, -type, -time_signature), by = c('track.id' = 'id')) %>%
  unique()

#write.csv(playlist_songs, 'genre_songs.csv', row.names=FALSE)