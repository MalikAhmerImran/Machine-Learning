import csv 
import random
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
model=Perceptron()

with open('/content/BankNoteAuthentication.csv') as f:
  data_set=csv.reader(f)
  next(data_set)
  data=[]
  for row in data_set:
    data.append({
      "evidence":[float(cell) for cell in row[:4]],
      "label":["Authentic" if row[4]=='0' else "Counterfit"]

  })

hold_out=int(0.50*len(data))
random.shuffle(data)
training=data[hold_out:]
testing=data[:hold_out]

x_training=[row["evidence"] for row in training]
y_training=[row["label"] for row in training]

x_testing=[row["evidence"] for row in testing]
y_testing=[row["label"] for row in testing]

model.fit(x_training,y_training)

predict=model.predict(x_testing)

correct=0
incorrect=0
total=0
for actual,predicted in zip(y_testing,predict):
 
  total+=1

  if actual[0]==predicted:
           correct+=1
    
  else:
      incorrect+=1


print(f" Correct:{correct}")
print(f" Incorrect:{incorrect}")
print(f" Accuracy:{100*correct/total:.2f}")

