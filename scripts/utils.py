
import matplotlib.pyplot as plt

def add_stratifing_label(datas):
  stratifing_labels = []
  for data in datas:
      if data["extracted_part"]["text"][0]:
          extracted = 1
      else:
          extracted = 0
      stratifing_labels.append(data["label"] + "_" + str(extracted))
  return stratifing_labels

def plot_lurning_curve(train, val, label: str):

    plt.grid()
    plt.plot(train, color='#f4777f')
    plt.plot(val, color='#7eb19c')
    plt.title('Learning Curve')
    plt.ylabel(label)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='best')
    plt.show()