#Majority Classifier
import numpy as np

# return array with features full of the most frequent class
def predictMaj(features):
    (values,counts) = np.unique(features,return_counts=True)
    values[np.argmax(counts)]  # prints the most frequent element
    return np.full(shape=len(features),fill_value=values[np.argmax(counts)],dtype=int)


def main():

      arr = np.array([0,0,0,0,0,0,0,0,0,0, 0, -2, 1, -2, 0, 4, 4, -6, -1])
      print predictMaj(arr)

if __name__ == '__main__':
    main()
