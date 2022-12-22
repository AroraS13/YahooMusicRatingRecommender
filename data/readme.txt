
Dataset: ydata-ymusic-user-song-ratings-meta-v1_0

Yahoo! Music user ratings of songs with song attributes, version 1.0

=====================================================================
This dataset is provided as part of the Yahoo! Research Alliance
Webscope program, to be used for approved non-commercial research
purposes by recipients who have signed a Data Sharing Agreement
with Yahoo!. This dataset is not to be redistributed. No
personally identifying information is available in this dataset.
More information about the Yahoo! Research Alliance Webscope
program is available at http://research.yahoo.com
=====================================================================

Full description:

This dataset represents a snapshot of the Yahoo! Music community's
preferences for various songs. The dataset contains over 717
million ratings of 136 thousand songs given by 1.8 million users
of Yahoo! Music services. The data was collected between 2002 and
2006. Each song in the data set is accompanied by artist, album,
and genre attributes. The users, songs, artists, and albums are
represented by randomly assigned numeric id's so that no
identifying information is revealed. The mapping from geners id's
to geners, as well as the genere hierarchy, is given.

Standard pre-processing for collaborative filtering and rating
prediction was applied to create this data set. The data has been
trimmed so that each user has rated at least 20 songs, and each
song has been rated by at least 20 users. The data has been
randomly partitioned into 10 equally sized sets of users to
enable cross-validation techniques. The available ratings for
each user have been randomly partitioned into training and test
sets to enable all-but-K testing protocols. The test set consist
of 10 ratings, and the training set consists of the remaining
ratings.

Tab is used as a delimiter for all data files.

This dataset consists of twenty two files:
1. ydata-ymusic-user-song-ratings-meta-v1_0/train-1.txt
2. ydata-ymusic-user-song-ratings-meta-v1_0/train-2.txt
3. ydata-ymusic-user-song-ratings-meta-v1_0/train-3.txt
4. ydata-ymusic-user-song-ratings-meta-v1_0/train-4.txt
5. ydata-ymusic-user-song-ratings-meta-v1_0/train-5.txt
6. ydata-ymusic-user-song-ratings-meta-v1_0/train-6.txt
7. ydata-ymusic-user-song-ratings-meta-v1_0/train-7.txt
8. ydata-ymusic-user-song-ratings-meta-v1_0/train-8.txt
9. ydata-ymusic-user-song-ratings-meta-v1_0/train-9.txt
10. ydata-ymusic-user-song-ratings-meta-v1_0/train-10.txt
11. ydata-ymusic-user-song-ratings-meta-v1_0/test-1.txt
12. ydata-ymusic-user-song-ratings-meta-v1_0/test-2.txt
13. ydata-ymusic-user-song-ratings-meta-v1_0/test-3.txt
14. ydata-ymusic-user-song-ratings-meta-v1_0/test-4.txt
15. ydata-ymusic-user-song-ratings-meta-v1_0/test-5.txt
16. ydata-ymusic-user-song-ratings-meta-v1_0/test-6.txt
17. ydata-ymusic-user-song-ratings-meta-v1_0/test-7.txt
18. ydata-ymusic-user-song-ratings-meta-v1_0/test-8.txt
19. ydata-ymusic-user-song-ratings-meta-v1_0/test-9.txt
20. ydata-ymusic-user-song-ratings-meta-v1_0/test-10.txt
21. ydata-ymusic-user-song-ratings-meta-v1_0/song-attributes.txt
22. ydata-ymusic-user-song-ratings-meta-v1_0/genre-hierarchy.txt

The content of the files are as follows:

=====================================================================

(1-10) "ydata-ymusic-user-song-ratings-meta-v1_0/train-n.txt":
Each training data file contains training data for 200,000 users.
Each user in each training data file has at least 10 observed
ratings. Each user has at most one observation for each song. The
users are ordered by randomly assigned user id. The observations
for each user are listed sequentially, and are ordered by
randomly assigned song id. The ratings values are on a scale from
1 to 5. The format of each row of each file is "user id<TAB>song
id<TAB>rating".

Snippet:
49      2169    5
49      2180    2
50      311     5
50      325     1

====================================================================

(11-20) "ydata-ymusic-user-song-ratings-meta-v1_0/test-n.txt":
For each user in each training file, 10 test songs are available
in the corresponding test file. The users are ordered by randomly
assigned user id. The test observations for each user are listed
sequentially, and are ordered by randomly assigned song id. The
format of each row of each file is "user id<TAB>song
id<TAB>rating".

Snippet:
49      1640    3
49      2095    3
49      2139    5
50      439     1
50      661     5

====================================================================

(21)"ydata-ymusic-user-song-ratings-meta-v1_0/song-attributes.txt":
Each line of this file lists the attributes for a particular
song. The available attributes are artist id, album id, and genre
id. The format of each row of each file is "song id<TAB>album
id<TAB>artist id<TAB>genre id".

Snippet:
143     124143  22021   0
144     107638  47567   106
145     70419   105252  106
146     20089   124124  134

====================================================================

(22) "ydata-ymusic-user-song-ratings-meta-v1_0/genre-hierarchy.txt":
his file describes the genre hierarchy used by Yahoo! Music. Each
line of this file lists the id number for a genre, the id number
of the parent of that genre, the level of the genre in the
hierarchy, and the name of the genre. Note that songs with no
genre assigned by Yahoo! Music have been assigned to the special
genre "Unknown". The format of each row of each file is "genre
id<TAB>parent genre id<TAB>level<TAB>genre name".

Snippet:
0       0       1       Unknown
1       1       1       Electronic/Dance
2       1       2       Ambient
3       2       3       Ambient Dub

