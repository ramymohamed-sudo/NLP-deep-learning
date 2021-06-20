


""" Shuffle the data """
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
val_samples = int(validation_split*data.shape[0])

x_train = data[:-val_samples]
y_train = labels[:-val_samples]
x_test = data[-val_samples:]
y_test = labels[-val_samples:]

