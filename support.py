import numpy as np
import pandas as pd

def pointDistance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def pointAngle(x1, y1, x2, y2, x3, y3):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def getDFPointsAngles(dataset, targets):
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11 = ([] for _ in range(11))
    a1, a2, a3, a4, a5, a6, a7 = ([] for _ in range(7))
    for i in range(dataset.shape[0]):
        d1.append(pointDistance(
            dataset['2x'][i], dataset['2y'][i], dataset['17x'][i], dataset['17y'][i]))
        d2.append(pointDistance(
            dataset['10x'][i], dataset['10y'][i], dataset['27x'][i], dataset['27y'][i]))
        d3.append(pointDistance(
            dataset['2x'][i], dataset['2y'][i], dataset['89x'][i], dataset['89y'][i]))
        d4.append(pointDistance(
            dataset['10x'][i], dataset['10y'][i], dataset['89x'][i], dataset['89y'][i]))
        d5.append(pointDistance(
            dataset['48x'][i], dataset['48y'][i], dataset['54x'][i], dataset['54y'][i]))
        d6.append(pointDistance(
            dataset['39x'][i], dataset['39y'][i], dataset['89x'][i], dataset['89y'][i]))
        d7.append(pointDistance(
            dataset['44x'][i], dataset['44y'][i], dataset['89x'][i], dataset['89y'][i]))
        d8.append(pointDistance(
            dataset['51x'][i], dataset['51y'][i], dataset['57x'][i], dataset['57y'][i]))
        d9.append(pointDistance(
            dataset['17x'][i], dataset['17y'][i], dataset['27x'][i], dataset['27y'][i]))
        d10.append(pointDistance(
            dataset['39x'][i], dataset['39y'][i], dataset['57x'][i], dataset['57y'][i]))
        d11.append(pointDistance(
            dataset['44x'][i], dataset['44y'][i], dataset['57x'][i], dataset['57y'][i]))
        a1.append(pointAngle(dataset['48x'][i], dataset['48y'][i], dataset['89x']
                             [i], dataset['89y'][i], dataset['54x'][i], dataset['54y'][i]))
        a2.append(pointAngle(dataset['57x'][i], dataset['57y'][i], dataset['54x']
                             [i], dataset['54y'][i], dataset['51x'][i], dataset['51y'][i]))
        a3.append(pointAngle(dataset['57x'][i], dataset['57y'][i], dataset['48x']
                             [i], dataset['48y'][i], dataset['51x'][i], dataset['51y'][i]))
        a4.append(pointAngle(dataset['57x'][i], dataset['57y'][i], dataset['54x']
                             [i], dataset['54y'][i], dataset['89x'][i], dataset['89y'][i]))
        a5.append(pointAngle(dataset['57x'][i], dataset['57y'][i], dataset['48x']
                             [i], dataset['48y'][i], dataset['89x'][i], dataset['89y'][i]))
        a6.append(pointAngle(dataset['27x'][i], dataset['27y'][i], dataset['4x']
                             [i], dataset['4y'][i], dataset['12x'][i], dataset['12y'][i]))
        a7.append(pointAngle(dataset['17x'][i], dataset['17y'][i], dataset['12x']
                             [i], dataset['12y'][i], dataset['4x'][i], dataset['4y'][i]))
    d = {'d1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6, 'd7': d7, 'd8': d8, 'd9': d9,
         'd10': d10, 'd11': d11, 'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7}
    dataset2 = pd.DataFrame(data=d)
    dataset2 = pd.concat([dataset2, targets], axis=1)
    return dataset2
