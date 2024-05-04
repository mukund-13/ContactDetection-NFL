

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.metrics import matthews_corrcoef

# Read in data files
BASE_DIR = "../input/nfl-player-contact-detection"

# Labels and sample submission
labels = pd.read_csv(f"{BASE_DIR}/train_labels.csv", parse_dates=["datetime"])

ss = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")

# Player tracking data
tr_tracking = pd.read_csv(
    f"{BASE_DIR}/train_player_tracking.csv", parse_dates=["datetime"]
)
te_tracking = pd.read_csv(
    f"{BASE_DIR}/test_player_tracking.csv", parse_dates=["datetime"]
)

# Baseline helmet detection labels
tr_helmets = pd.read_csv(f"{BASE_DIR}/train_baseline_helmets.csv")
te_helmets = pd.read_csv(f"{BASE_DIR}/test_baseline_helmets.csv")

# Video metadata with start/stop timestamps
tr_video_metadata = pd.read_csv(
    "../input/nfl-player-contact-detection/train_video_metadata.csv",
    parse_dates=["start_time", "end_time", "snap_time"],
)


example_video = "../input/nfl-player-contact-detection/train/58168_003392_Sideline.mp4"
output_video = video_with_helmets(example_video, tr_helmets)

frac = 0.65  # scaling factor for display
display(
    Video(data=output_video, embed=True, height=int(720 * frac), width=int(1280 * frac))
)

game_play = "58168_003392"
example_tracks = tr_tracking.query("game_play == @game_play and step == 0")
ax = create_football_field()
for team, d in example_tracks.groupby("team"):
    ax.scatter(
        d["x_position"],
        d["y_position"],
        label=team,
        s=65,
        lw=1,
        edgecolors="black",
        zorder=5,
    )
ax.legend().remove()
ax.set_title(f"Tracking data for {game_play}: at step 0", fontsize=15)
plt.show()

tr_tracking.head()

	game_play 	game_key 	play_id 	nfl_player_id 	datetime 	step 	team 	position 	jersey_number 	x_position 	y_position 	speed 	distance 	direction 	orientation 	acceleration 	sa
0 	58580_001136 	58580 	1136 	44830 	2021-10-10 21:08:20.900000+00:00 	-108 	away 	CB 	22 	61.59 	42.60 	1.11 	0.11 	320.33 	263.93 	0.71 	-0.64
1 	58580_001136 	58580 	1136 	47800 	2021-10-10 21:08:20.900000+00:00 	-108 	away 	DE 	97 	59.48 	26.81 	0.23 	0.01 	346.84 	247.16 	1.29 	0.90
2 	58580_001136 	58580 	1136 	52444 	2021-10-10 21:08:20.900000+00:00 	-108 	away 	FS 	29 	72.19 	31.46 	0.61 	0.06 	11.77 	247.69 	0.63 	-0.33
3 	58580_001136 	58580 	1136 	46206 	2021-10-10 21:08:20.900000+00:00 	-108 	home 	TE 	86 	57.37 	22.12 	0.37 	0.04 	127.85 	63.63 	0.69 	0.62
4 	58580_001136 	58580 	1136 	52663 	2021-10-10 21:08:20.900000+00:00 	-108 	away 	ILB 	48 	63.25 	27.50 	0.51 	0.05 	183.62 	253.71 	0.31 	0.31
Video Metadata

These files provide information that can be used to sync the video files with the NGS tracking data.

tr_video_metadata.head()

	game_play 	game_key 	play_id 	view 	start_time 	end_time 	snap_time
0 	58168_003392 	58168 	3392 	Endzone 	2020-09-11 03:01:43.134000+00:00 	2020-09-11 03:01:54.971000+00:00 	2020-09-11 03:01:48.134000+00:00
1 	58168_003392 	58168 	3392 	Sideline 	2020-09-11 03:01:43.134000+00:00 	2020-09-11 03:01:54.971000+00:00 	2020-09-11 03:01:48.134000+00:00
2 	58172_003247 	58172 	3247 	Endzone 	2020-09-13 19:30:42.414000+00:00 	2020-09-13 19:31:00.524000+00:00 	2020-09-13 19:30:47.414000+00:00
3 	58172_003247 	58172 	3247 	Sideline 	2020-09-13 19:30:42.414000+00:00 	2020-09-13 19:31:00.524000+00:00 	2020-09-13 19:30:47.414000+00:00
4 	58173_003606 	58173 	3606 	Endzone 	2020-09-13 19:45:07.527000+00:00 	2020-09-13 19:45:26.438000+00:00 	2020-09-13 19:45:12.527000+00:00

labels.head()

	contact_id 	game_play 	datetime 	step 	nfl_player_id_1 	nfl_player_id_2 	contact
0 	58168_003392_0_38590_43854 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	43854 	0
1 	58168_003392_0_38590_41257 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	41257 	0
2 	58168_003392_0_38590_41944 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	41944 	0
3 	58168_003392_0_38590_42386 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	42386 	0
4 	58168_003392_0_38590_47944 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	47944 	0
#Example Video with Labels

I#n this video you can see the labels for this play, approximately linked with the associated helmets within the video.

No#te in this video:

   # Black helmet boxes indicate that player is not in contact. A unique number (home/visiting combined with jersey number) is shown next to their helmet.
    #Green helmet box indicates the player is in contact with one or more players.
    #Red helmet box indicates the player is in contact with the ground (and possibly another player).
    #Blue lines show the link between players in contact with each other.

game_play = "58168_003392"
gp = join_helmets_contact(game_play, labels, tr_helmets, tr_video_metadata)

example_video = f"../input/nfl-player-contact-detection/train/{game_play}_Sideline.mp4"
output_video = video_with_contact(example_video, gp)

frac = 0.65  # scaling factor for display
display(
    Video(data=output_video, embed=True, height=int(720 * frac), width=int(1280 * frac))
)

Running for 58168_003392_Sideline.mp4

OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'

Your browser does not support the video tag.
Sample Submission

A sample_submission.csv file is provided. When your notebook is being processed on the test set this file will include every row required for a valid submission.

Note the contact_id column is a unique identifier consisting of a combination of game_play, step, nfl_player_id_1 and nfl_player_id_2

ss = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')
ss.head()

	contact_id 	contact
0 	58168_003392_0_38590_43854 	0
1 	58168_003392_0_38590_41257 	0
2 	58168_003392_0_38590_41944 	0
3 	58168_003392_0_38590_42386 	0
4 	58168_003392_0_38590_47944 	0
Baseline Approach

We can create a simple solution using only the tracking data and identifying an optimal threshold based on the training data for the competition metric.

This baseline submission does not consider player to ground contact, and predicts every player-to-ground row as non-contact.

The process is simple:

    For each contact_id we compute the seperation distance between players.
    We fill player-to-ground rows with a distance of 99 so they are treated as non-contact.
    We loop through thresholds between 0 and 5 yards and compute the competition metric at each threshold.
    We use the threshold that produces the best score when applying this to our test submission.

df_combo = compute_distance(labels, tr_tracking)

print(df_combo.shape, labels.shape)

df_dist = df_combo.merge(
    tr_tracking[["game_play", "datetime", "step"]].drop_duplicates()
)
df_dist["distance"] = df_dist["distance"].fillna(99)  # Fill player to ground with 99

(4721618, 12) (4721618, 7)

Note our dataframe now includes distance.

df_dist.head()

	contact_id 	game_play 	datetime 	step 	nfl_player_id_1 	nfl_player_id_2 	contact 	x_position_1 	y_position_1 	x_position_2 	y_position_2 	distance
0 	58168_003392_0_38590_43854 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	43854 	0 	40.33 	25.28 	41.99 	16.79 	8.650763
1 	58168_003392_0_38590_41257 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	41257 	0 	40.33 	25.28 	45.77 	15.59 	11.112592
2 	58168_003392_0_38590_41944 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	41944 	0 	40.33 	25.28 	42.00 	22.85 	2.948525
3 	58168_003392_0_38590_42386 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	42386 	0 	40.33 	25.28 	45.87 	23.89 	5.711716
4 	58168_003392_0_38590_47944 	58168_003392 	2020-09-11 03:01:48.100000+00:00 	0 	38590 	47944 	0 	40.33 	25.28 	42.10 	26.58 	2.196110

for dist in range(0, 5):
    score = matthews_corrcoef(df_dist["contact"], df_dist["distance"] <= dist)
    print(f"Threshold Distance: {dist} Yard - MCC Score: {score:0.4f}")

Threshold Distance: 0 Yard - MCC Score: 0.0000
Threshold Distance: 1 Yard - MCC Score: 0.5201
Threshold Distance: 2 Yard - MCC Score: 0.3592
Threshold Distance: 3 Yard - MCC Score: 0.2517
Threshold Distance: 4 Yard - MCC Score: 0.1935

Create Baseline Submission

Seeing above that the optimal threshold is 1 yard. We can now compute the distances on the test set and submit with the threshold of 1 yard.

Note: This is a code competition. When you submit, your model will be rerun on a set of 61 unseen plays located in a holdout test set. The publicly provided test videos are simply a set of mock plays (copied from the training set) which are not used in scoring.

ss = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")

THRES = 1

ss = expand_contact_id(ss)
ss_dist = compute_distance(ss, te_tracking, merge_col="step")

print(ss_dist.shape, ss.shape)

submission = ss_dist[["contact_id", "distance"]].copy()
submission["contact"] = (submission["distance"] <= THRES).astype("int")
submission = submission.drop('distance', axis=1)
submission[["contact_id", "contact"]].to_csv("submission.csv", index=False)

submission.head()

(49588, 11) (49588, 6)

	contact_id 	contact
0 	58168_003392_0_38590_43854 	0
1 	58168_003392_0_38590_41257 	0
2 	58168_003392_0_38590_41944 	0
3 	58168_003392_0_38590_42386 	0
4 	58168_003392_0_38590_47944 	0

