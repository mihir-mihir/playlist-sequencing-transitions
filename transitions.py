#!/Users/mihir/Developer/ANRG/sra/bin/python3


# constructing the feature space #
##################################
"""
acoustic vectors: (R12) average timbre vectors across all segments in track
key/mode vectors: (R3) adjacent keys in the circle of fifths and relative major/minor keys are equidistant (see fig 2 in paper)
tempo: (R2) magnitude and phase representation loosely based on scheme used in paper (also fig 2), see comments for description of implemented scheme
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque

# connect to spotify
os.environ["SPOTIPY_CLIENT_ID"] = "a1cab982635644aabcd6fcdc5f65d57b"
os.environ["SPOTIPY_CLIENT_SECRET"] = "4d495bfc9b52424b9b344eb19286366e"
spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

playlist_id = "5KALzJukYbeZkt0lYGF6Pj"
playlist_fields = "tracks.items(track(id, name, album.name))"

# track data for each item is in item's 'track' property
playlist_tracks = [
    item["track"]
    for item in spotify.playlist(playlist_id=playlist_id, fields=playlist_fields)[
        "tracks"
    ]["items"]
]


# string of track ids
playlist_track_ids = [track["id"] for track in playlist_tracks]

# print(",".join(playlist_track_ids))

track_features = [
    {
        "id": id,
        "name": "h",
        "features": {
            "tempo": analysis["track"]["tempo"],
            "key": analysis["track"]["key"],
            "mode": analysis["track"]["mode"],
            "segments": [
                {
                    "start": segment["start"],
                    "duration": segment["duration"],
                    "timbre": segment["timbre"],
                }
                for segment in analysis["segments"]
            ],
        },
    }
    for id, analysis in [(id, spotify.audio_analysis(id)) for id in playlist_track_ids]
]

# might need to fix this math
# key/mode coordinates
N_KEYS = 12
KEY_RADIUS = 1  # distance from a point on the polygon to the center


def key_coordinates(key_num, mode):
    theta = 2 * math.pi / N_KEYS  # angle between each side
    x = KEY_RADIUS * math.cos(key_num * theta)
    y = KEY_RADIUS * math.sin(key_num * theta)
    return np.array([x, y, mode])


# visually testing the x, y coordinates to make sure they form 12-sided equilateral polygon (z coordinate will just be 0 or 1 depending on the mode)
coords = []
for i in range(N_KEYS):
    coords.append(key_coordinates(i, 0))

track_vectors = {
    feature["id"]: {
        # kind of lengthy compared to:
        # "tempo": np.array([math.log(feature["features"]["tempo"], 2) % 1, math.log(feature["features"]["tempo"], 2) // 1]),
        # used list comprehension so I didn't have to access feature["features"]["tempo"] twice -- is there a cleaner way to do this?
        # phase is equal to decimial part of the base-2 log of the tempo -- tempos x and y that are related by x = y * 2^n (when you double or half y a given amount of times, you get x), will have the same phase
        # magnitude is equal to the integer part of the base-2 log of the tempo -- if tempo x is double y, mag(x) - mag(y) = 1, if x is quadruple y, mag(x) - mag(y) = 2; in the tempo data from the songs, expecting
        # the bpms to range from ~40 to ~160, so the greatest difference in magnitudes between two tempo representations (phase/mag) is around 2 in the most extreme case where the tempos of two songs are near the extremes:
        # mag(160) - mag(40) ~=~ 7.322 - 5.322 = 2
        # phase of tempos in same "tempo octave" is the same, like it is in the paper, but including information about actual value of tempo with magnitude component
        "tempo": [
            np.array([tempo_log % 1, tempo_log // 1])
            for tempo_log in [
                math.log(tempo, 2) for tempo in [feature["features"]["tempo"]]
            ]
        ][0],
        # combining key and mode information to get 3D coordinates according to diagram in paper
        "key": key_coordinates(feature["features"]["key"], feature["features"]["mode"]),
        # averaging timbre vectors over all segments to get timbre vector from track for now
        "timbre": np.average(
            np.array(
                [segment["timbre"] for segment in feature["features"]["segments"]]
            ),
            axis=0,
        ),
    }
    for feature in track_features
}

# define weights for each of the three feature vectors associated with a track to be used when computing
# a weighted average of the distances between like feature vectors from two different tracks
weight = {}
weight["tempo"] = 0.3
weight["key"] = 0.4
weight["timbre"] = 0.3


# gives a weighted average of the euclidean distances between each track, using the weights for each of the three feature vectors associated with a track
# NOTE: probably need to apply some kind of scaling/normalization to each of the feature values,
#       don't want one feature to dominate just because its values happen to be a lot bigger than
#       the values for the other features
def track_distance(track1_id, track2_id):
    # uing the fact that euclidean distance is l2 norm, and default value of the ord parameter in np.linalg.norm is 2
    weighted_tempo_distance = weight["tempo"] * np.linalg.norm(
        track_vectors[track1_id]["tempo"] - track_vectors[track2_id]["tempo"]
    )
    weighted_key_distance = weight["key"] * np.linalg.norm(
        track_vectors[track1_id]["key"] - track_vectors[track2_id]["key"]
    )
    weighted_timbre_distance = weight["timbre"] * np.linalg.norm(
        track_vectors[track1_id]["timbre"] - track_vectors[track2_id]["timbre"]
    )

    return weighted_tempo_distance + weighted_key_distance + weighted_timbre_distance


# Greedy approximation algorithm for shortest Hamiltonian path (HAM-2 algorithm described in https://archives.ismir.net/ismir2017/paper/000086.pdf)
visited = deque([playlist_track_ids[0]])

while len(visited) < len(playlist_track_ids):
    min_distance_from_head = math.inf
    min_distance_from_tail = math.inf
    new_head = None
    new_tail = None
    for track_id in playlist_track_ids:
        if track_id in visited:
            continue

        # check distance from head
        d_head = track_distance(visited[0], track_id)
        d_tail = track_distance(visited[-1], track_id)

        if d_head < min_distance_from_head:
            min_distance_from_head = d_head
            new_head = track_id
        if d_tail < min_distance_from_tail:
            min_distance_from_tail = d_tail
            new_tail = track_id
    if min_distance_from_tail > min_distance_from_head:
        visited.append(new_tail)
    else:
        visited.appendleft(new_head)

# sequenced song ordering
for track_id in visited:
    track_name = spotify.track(track_id)["name"]
    print(track_name)

# distances between songs in the sequencing
for i in range(len(playlist_track_ids) - 1):
    dist = track_distance(playlist_track_ids[i], playlist_track_ids[i + 1])


# Scratch pad
# #######################################################################################

# visual test to make sure shape for key feature vector is correct
# ####################################################################
# plt.plot([coord[0] for coord in coords], [coord[1] for coord in coords], linestyle="", marker="o")
# plt.show()

# bpm polar coordinate scheme using custom ranges to determine magnitude - elected to use above scheme instead
# #############
# base_bpm = 40
# n_bpm_doublings = 2
# bpm_ranges = [range(base_bpm * (i + 1), 2 * base_bpm * (i + 1)) for i in range(n_bpm_doublings)]
