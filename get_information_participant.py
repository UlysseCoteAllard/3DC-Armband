import pandas as pd
import numpy as np
import collections


def read_info_participants():
    data = pd.read_csv("information_participants.csv")

    data['Age'] = data['Age'].astype(dtype=np.float32)
    data['DistanceFirstArmband'] = data['DistanceFirstArmband'].astype(dtype=np.float32)
    data['TotalLengthForearm'] = data['TotalLengthForearm'].astype(dtype=np.float32)

    data['Sex'] = data['Sex'].astype(dtype='category')
    data['Handedness'] = data['Handedness'].astype(dtype='category')
    data['MyoUp'] = data['MyoUp'].astype(dtype='category')

    print(data)

    print("Distribution M/F: ", collections.Counter(np.array(data["Sex"])))
    print("Distribution Right/Left handed: ", collections.Counter(np.array(data["Handedness"])))

    print("Average Age: ", data['Age'].mean())
    print("STD Age: ", data['Age'].std())
    print("Youngest Participant: ", data['Age'].min())
    print("Oldest Participant: ", data['Age'].max())


if __name__ == '__main__':
    read_info_participants()
