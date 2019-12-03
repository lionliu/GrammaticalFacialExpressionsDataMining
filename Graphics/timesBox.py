import matplotlib.pyplot as plt

mlp = [
    0.24186219999990044,
     0.3675352999999859,
     0.2712975000000597,
     0.26915439999993396,
     0.2170179999999391,
     0.43014240000002246,
     0.43922059999999874,
     0.2930390000000216,
     0.2706236000000217,
     0.21694490000004407,
     0.39970689999995557,
     0.3033702999999832,
     0.274655800000005,
     0.29102309999996123,
     0.35867749999999887,
     0.35417260000008355,
     0.2625034999999798,
     0.3417845999999827,
]

bag = [
     12.688171799999964,
     24.17234099999996,
     3.4096673999999894,
     15.869863899999928,
     6.621265200000153,
     24.439079300000003,
     4.013929799999914,
     15.474587400000019,
     31.738706000000093,
     24.581829599999992,
     21.01357810000013,
     8.574905200000103,
     14.295685999999932,
     27.33024679999994,
     3.9350306999999702,
     38.7410640999999,
     6.73759229999996,
     35.54161739999995,
              ]

tree = []
knn = []
forest = []
voting = []



times = [tree, knn, mlp,
    forest, bag, voting]
labels = ['tree', 'KNN', 'MLP', 'RF', 'Bagging', 'Voting']

plt.figure()
plt.title("Preprocessed dataset execution duration")
plt.xlabel('Classifiers')
plt.ylabel('Seconds')
box = plt.boxplot(times, labels=labels, patch_artist=True)
plt.show()
