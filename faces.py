import sys, random, math


def import_trainingdata(filename,faces):
    """Import image data into a dictionary"""
    with open(filename) as facefile:
        for line in facefile:
            if len(line) > 1 and not line.startswith('#'):
                if line.startswith('I'):
                    image_number = line.strip()
                    faces[image_number] = []
                else:
                    values = line.split()
                    values = [float(n)/32 for n in values]
                    faces[image_number].extend(values)
    #add bias 
    for item in faces:
        faces[item].append(1)

def import_facitdata(filename,facits):
    """Import facit data into a dictionary"""
    with open(filename) as facit:   
        for line in facit:
            if len(line) > 1 and not line.startswith('#'):
                data = line.split()
                facits[data[0]] = [0]*4
                facits[data[0]][int(data[1]) - 1] = 1


def output(sample,weight):
    """Calculate ouput of each each node. Use sigmoid fucntion as activation function"""
    sumup = [0]*4
    for j in range(4):
        for i in range(len(sample)):
            sumup[j] += sample[i] * weight[j][i] 
        sumup[j] = sigmoid(sumup[j])
    return sumup

def sigmoid(x):
    """Sigmoid function"""
    return 1/ (1+ math.exp(-x))
            
def training(trainingset):
    """Train network using back-progapation as a perception algorithm"""
    random.shuffle(trainingset)
    for image in trainingset:  
        tempresult = output(training_faces[image],weight)
        for i in range(4):
            error = (training_facits[image][i] - tempresult[i])
            for j in range(401):
                weight[i][j] += a*training_faces[image][j]*error
        
def test(testset):
    """Test how training scetion works by returning an average error count of all nodes"""
    result = []
    sumerror = 0
    for item in testset:
        result = output(training_faces[item],weight)
        abserror = 0
        for i in range(4):
            abserror += abs(training_facits[item][i] - result[i])
        sumerror += abserror
    return sumerror / len(testset)
                            
                                                            
training_faces = {}
training_facits = {}
classification_faces = {}
classification_facits = {}
test_results = {}
weight = []
a = 0.5
training_set = []
test_set = []
filename = []

for i in range(len(sys.argv)-1):
    filename.append(sys.argv[i+1])

import_trainingdata(filename[0],training_faces)
import_facitdata(filename[1],training_facits)
import_trainingdata(filename[2],classification_faces)
import_facitdata(filename[3],test_results)


        
# initialize weight with random number (0,1)
weight = [[random.random() for j in range(401)]for i in range(4)]

#seperate training set and test set
for i in range(len(training_faces.keys())):
    if i < len(training_faces.keys())*0.6:
        training_set.append(training_faces.keys()[i])
    else:
        test_set.append(training_faces.keys()[i])

#training circles
loop_count = 0                                                                                          
while test(test_set) > 0.6:
    loop_count += 1
    training(training_set)
    if loop_count > 500:
        break

#Classification  
right = 0
for item in classification_faces.keys():
    result = output(classification_faces[item],weight) 
    classification_facits[item] = result.index(max(result))+1
    if classification_facits[item] == test_results[item].index(1) +1 :
        right += 1
 
print right
    
#output classification answer file
fc = open('claasification-facits.txt','w')
fc.write('# AI assignment 2 neural network classification answers #\n')
ckeys = classification_facits.keys() 
ckeys.sort()      
for item in ckeys:
    fc.write(item+' '+str(classification_facits[item])+'\n')
    #print item + "   " + str(classification_facits[item])
fc.close()
