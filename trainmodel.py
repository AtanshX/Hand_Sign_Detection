import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure consistent lengths for all data samples
fixed_data = []
fixed_labels = []

for sample, label in zip(data_dict['data'], data_dict['labels']):
    if len(sample) == 42:  # Ensure all samples have the expected length
        fixed_data.append(sample)
        fixed_labels.append(label)

data = np.asarray(fixed_data)
labels = np.asarray(fixed_labels)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
