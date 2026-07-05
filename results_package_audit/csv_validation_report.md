# CSV Validation Report

## feature_statistics.csv
existence: YES
dimensions: (100, 7)
rows: 100
columns: 7
schema: ['filename', 'number_of_clips', 'feature_dimension', 'mean', 'std', 'min', 'max']
data types:
filename              object
number_of_clips        int64
feature_dimension      int64
mean                 float64
std                  float64
min                  float64
max                  float64
NaN count: 0
duplicate rows: 0
first 5 rows:
        filename  number_of_clips  feature_dimension      mean       std       min       max
0  MVI_20011.npy              163                768 -0.000169  0.414782 -1.925781  4.886719
1  MVI_20012.npy              231                768  0.000056  0.418516 -1.872070  4.968750
2  MVI_20032.npy              106                768 -0.000988  0.413531 -1.952148  5.183594
3  MVI_20033.npy              193                768 -0.000478  0.413017 -2.093750  5.089844
4  MVI_20034.npy              197                768 -0.000576  0.403954 -2.013672  5.152344
last 5 rows:
         filename  number_of_clips  feature_dimension      mean       std       min       max
95  MVI_63553.npy              348                768  0.000550  0.434534 -2.382812  4.609375
96  MVI_63554.npy              358                768  0.000786  0.440358 -2.072266  4.871094
97  MVI_63561.npy              318                768 -0.000131  0.410800 -1.835938  5.046875
98  MVI_63562.npy              293                768 -0.000906  0.395671 -1.889648  5.195312
99  MVI_63563.npy              344                768 -0.000880  0.394113 -2.064453  5.316406

## feature_distribution.csv
existence: YES
dimensions: (12, 2)
rows: 12
columns: 2
schema: ['metric', 'value']
data types:
metric     object
value     float64
NaN count: 0
duplicate rows: 0
first 5 rows:
             metric         value
0   feature_entropy  7.637759e-01
1  feature_sparsity  3.752077e-08
2      eigenvalue_1  6.067873e+00
3      eigenvalue_2  4.774627e+00
4      eigenvalue_3  3.708329e+00
last 5 rows:
           metric     value
7    eigenvalue_6  1.644894
8    eigenvalue_7  1.494906
9    eigenvalue_8  1.174919
10   eigenvalue_9  1.100499
11  eigenvalue_10  1.046665

## anomaly_scores.csv
existence: YES
dimensions: (100, 7)
rows: 100
columns: 7
schema: ['filename', 'mean_anomaly', 'max_anomaly', 'min_anomaly', 'variance', 'peak_count', 'anomaly_duration']
data types:
filename             object
mean_anomaly        float64
max_anomaly         float64
min_anomaly         float64
variance            float64
peak_count            int64
anomaly_duration      int64
NaN count: 0
duplicate rows: 0
first 5 rows:
        filename  mean_anomaly  max_anomaly  min_anomaly  variance  peak_count  anomaly_duration
0  MVI_20011.npy     14.099657    14.116663    14.086671  0.000033          12                29
1  MVI_20012.npy     14.095608    14.135648    14.075729  0.000107           6                28
2  MVI_20032.npy     14.096228    14.130140    14.069800  0.000147          11                16
3  MVI_20033.npy     14.110043    14.122913    14.092287  0.000032          23                31
4  MVI_20034.npy     14.101939    14.117204    14.085953  0.000038          16                34
last 5 rows:
         filename  mean_anomaly  max_anomaly  min_anomaly  variance  peak_count  anomaly_duration
95  MVI_63553.npy     14.130822    14.185835    14.105656  0.000158          28                56
96  MVI_63554.npy     14.129503    14.158635    14.101147  0.000128          38                59
97  MVI_63561.npy     14.109325    14.152793    14.090153  0.000109          27                54
98  MVI_63562.npy     14.100561    14.135465    14.083646  0.000061          15                38
99  MVI_63563.npy     14.099927    14.130812    14.082280  0.000063          26                57

## rewards.csv
existence: YES
dimensions: (5, 2)
rows: 5
columns: 2
schema: ['seed', 'reward']
data types:
seed        int64
reward    float64
NaN count: 0
duplicate rows: 0
first 5 rows:
   seed  reward
0    42 -0.1100
1   123 -2.2230
2   456  0.0020
3   789 -0.2270
4   999 -0.7605
last 5 rows:
   seed  reward
0    42 -0.1100
1   123 -2.2230
2   456  0.0020
3   789 -0.2270
4   999 -0.7605

## waiting.csv
existence: YES
dimensions: (5, 2)
rows: 5
columns: 2
schema: ['seed', 'waiting_time']
data types:
seed              int64
waiting_time    float64
NaN count: 0
duplicate rows: 0
first 5 rows:
   seed  waiting_time
0    42           0.0
1   123           0.0
2   456           0.0
3   789           0.0
4   999           0.0
last 5 rows:
   seed  waiting_time
0    42           0.0
1   123           0.0
2   456           0.0
3   789           0.0
4   999           0.0

## queue.csv
existence: YES
dimensions: (5, 2)
rows: 5
columns: 2
schema: ['seed', 'queue_length']
data types:
seed              int64
queue_length    float64
NaN count: 0
duplicate rows: 0
first 5 rows:
   seed  queue_length
0    42           0.0
1   123           0.0
2   456           0.0
3   789           0.0
4   999           0.0
last 5 rows:
   seed  queue_length
0    42           0.0
1   123           0.0
2   456           0.0
3   789           0.0
4   999           0.0

## throughput.csv
existence: YES
dimensions: (5, 2)
rows: 5
columns: 2
schema: ['seed', 'throughput']
data types:
seed          int64
throughput    int64
NaN count: 0
duplicate rows: 0
first 5 rows:
   seed  throughput
0    42         100
1   123         100
2   456         100
3   789         100
4   999         100
last 5 rows:
   seed  throughput
0    42         100
1   123         100
2   456         100
3   789         100
4   999         100

## ablation.csv
existence: YES
dimensions: (20, 3)
rows: 20
columns: 3
schema: ['mode', 'seed', 'reward']
data types:
mode       object
seed        int64
reward    float64
NaN count: 0
duplicate rows: 0
first 5 rows:
       mode  seed  reward
0  baseline    42 -0.9715
1  baseline   123 -0.9195
2  baseline   456 -0.9490
3  baseline   789 -0.3650
4  baseline   999 -1.1680
last 5 rows:
    mode  seed  reward
15  full    42 -0.3310
16  full   123 -1.1320
17  full   456 -0.7650
18  full   789 -0.7640
19  full   999 -3.9155

## statistical_analysis.csv
existence: YES
dimensions: (7, 10)
rows: 7
columns: 10
schema: ['comparison', 'metric', 'mean', 'median', 'std', 'ci_95', 'p_value', 'cohens_d', 'cliffs_delta', 'effect_size']
data types:
comparison       object
metric           object
mean            float64
median          float64
std             float64
ci_95           float64
p_value         float64
cohens_d        float64
cliffs_delta    float64
effect_size      object
NaN count: 28
duplicate rows: 0
first 5 rows:
            comparison       metric    mean  median       std     ci_95  p_value  cohens_d  cliffs_delta effect_size
0             baseline  descriptive -0.8746 -0.9490  0.301130  0.373902      NaN       NaN           NaN         NaN
1              feature  descriptive -1.0565 -0.8375  1.105026  1.372072      NaN       NaN           NaN         NaN
2              anomaly  descriptive -0.8390 -0.3165  0.785466  0.975285      NaN       NaN           NaN         NaN
3                 full  descriptive -1.3815 -0.7650  1.444674  1.793800      NaN       NaN           NaN         NaN
4  feature vs baseline  comparative     NaN     NaN       NaN       NaN      1.0 -0.224605          0.36       small
last 5 rows:
            comparison       metric    mean  median       std     ci_95  p_value  cohens_d  cliffs_delta effect_size
2              anomaly  descriptive -0.8390 -0.3165  0.785466  0.975285      NaN       NaN           NaN         NaN
3                 full  descriptive -1.3815 -0.7650  1.444674  1.793800      NaN       NaN           NaN         NaN
4  feature vs baseline  comparative     NaN     NaN       NaN       NaN   1.0000 -0.224605          0.36       small
5  anomaly vs baseline  comparative     NaN     NaN       NaN       NaN   0.8125  0.059849          0.20  negligible
6     full vs baseline  comparative     NaN     NaN       NaN       NaN   0.6250 -0.485771          0.12       small

