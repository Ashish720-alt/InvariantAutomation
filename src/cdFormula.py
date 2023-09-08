from pyexcel_ods import save_data
from collections import OrderedDict
import input 
import inspect
import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz  



def find_subclasses(module): 
    return [ (name,cls) for name, cls in inspect.getmembers(module) if inspect.isclass(cls) and name != '__class__' ]

def getc(A):
    if (len(A) == 0): 
        return -1
    return len(A[0])

def getTr_max(T):
    temp = 0
    for A in T:
        temp = max(temp,len(A.tlist))
    return temp

def maxcoeff(T):
    def maxcoeff_ptf(ptf):
        return abs(float(ptf.max()))
    

    maxcoeff = -math.inf
    for dt in T:
        for ptf in dt.tlist:
            temp = maxcoeff_ptf(ptf)
            maxcoeff = max(maxcoeff, temp)
    
    return maxcoeff


def interiorDerivative_T(T):
    def interiorDerivative_ptf (ptf):
        return float(np.trace(ptf) - 1)

    maxintder = -math.inf
    minintder = math.inf
    for dt in T:
        for ptf in dt.tlist:
            temp = interiorDerivative_ptf(ptf)
            maxintder = max(maxintder, temp)
            minintder = min(minintder, temp)
    
    return (minintder, maxintder)



def exteriorDerivative_T(T):
    def exteriorDerivative_ptf (ptf):
        def twoForm_norm( twoform):
            sum = 0
            for term1 in twoform:
                for term2 in twoform:
                    multivector1 = term1[1]
                    multivector2 = term2[1]
                    if ( multivector1[0] == multivector2[0] and multivector1[1] == multivector2[1] ):
                        det = 1
                    elif ( multivector1[0] == multivector2[1] and multivector1[1] == multivector2[0] ):
                        det = -1
                    else:
                        det = 0
                    sum = det * term1[0] * term2[0]

            return math.sqrt(sum)

        rv = []
        n = len(ptf) - 1 
        if (n == 1):
            return -1 
        for j in range(n): #Don't want final row and column entries
            for i in range(n):
                rv.append( ( ptf[i][j] , (i + 1, j + 1) ) )
        
        return twoForm_norm(rv)

    maxextder = -math.inf
    minextder = math.inf
    for dt in T:
        for ptf in dt.tlist:
            temp = exteriorDerivative_ptf(ptf)
            maxextder = max(maxextder, temp)
            minextder = min(minextder, temp)
    
    return (minextder, maxextder)

def maxabsconst (dnf):
    def maxabsconst_cc (cc):
        def maxabsconst_p(p):
            temp = p[:-2] + p[-1]
            return int(max( abs(max(temp)), abs(min(temp))))
        return max( [ maxabsconst_p(p) for p in cc  ] ) if len(cc) > 0 else 0
    return max( [maxabsconst_cc(cc) for cc in dnf] ) if len(dnf) > 0 else 0
    
            
def dim(dnf, n):
    def dim(cc, n):
        L = cc.tolist()
        d = n
        for p in L:
            if (p[-2] == 0):
                d = d - 1
        return d 
    return [dim(cc, n) for cc in dnf]      


def getlist(C, name):
    (minintder, maxintder) = interiorDerivative_T(C.T)
    (minextder, maxextder) = exteriorDerivative_T(C.T)
    maxcoeff_T = maxcoeff(C.T)
    A = max(dim(C.P, len(C.Var))) 
    return [name, len(C.Var), len(C.P), getc(C.P), maxabsconst (C.P),  len(C.Q), getc(C.Q), maxabsconst (C.Q), len(C.B), getc(C.B), maxabsconst (C.B), 
            len(C.T), getTr_max(C.T), maxcoeff_T, maxintder, minintder, minextder, maxextder, C.c, C.d  ] #V1
    # return [name, len(C.Var), getc(C.P), A , maxintder, minintder, minextder, maxextder, C.c, C.d  ] #V2
    


def getAllLists(module, rv):
    dirs = find_subclasses(module)
    for dir in dirs:
        files = find_subclasses(dir[1])
        for file in files:
            rv.append(getlist( file[1], dir[0] + '.' + file[0] ))
    return 
        
# interior and exterior derviatives are analogs of curl and gradient of 3D vector fields, curl f
# But exterior, interior and geometrical are linked
headers = ['Filename', 'n', 'P.d' , 'P.c' , 'P.maxabs', 'Q.d', 'Q.c', 'Q.maxabs', 'B.d', 'B.c', 'B.maxabs', 'T.d', 'T.r_max', 'T.coeffmax', 'T.interiorderivative_max', 'T.interiorderivative_min', 
                'T.exteriorderivative_max', 'T.exteriorderivative_min', 'I.c', 'I.d'] #V1

# headers = ['Filename', 'n', 'P.c' , 'P.dim', 'T.interiorderivative_max', 'T.interiorderivative_min', 'T.exteriorderivative_max', 
#            'T.exteriorderivative_min', 'I.c', 'I.d' ] #V2

entrycount = len(headers) 
printlist = [ headers]
getAllLists(input.Inputs, printlist)
dataset = np.array(printlist)
X = dataset[1:, 1:(entrycount - 2)].astype(float) 
y_d = dataset[1:, (entrycount - 1)].astype(float) 
y_c = dataset[1:, (entrycount - 2)].astype(float) 


# Decision Trees.

regressor_d = DecisionTreeRegressor(random_state = 0) 
regressor_c = DecisionTreeRegressor(random_state = 0) 
regressor_c.fit(X, y_c)
regressor_d.fit(X, y_d)

I_d_pred = ['I.d_predicted']
I_c_pred = ['I.c_predicted']
I_c_change = ['I.c_change']
I_d_change = ['I.d_change']

for (i,data) in enumerate(X):
    dpred = math.ceil(regressor_d.predict([data]))
    cpred = float(regressor_c.predict([data]))
    I_d_pred.append( dpred )
    I_c_pred.append( cpred )
    I_c_change.append(cpred - float(y_c[i]))
    I_d_change.append(dpred - float(y_d[i]))
    

finaldata = []
for i in range(len(printlist) - 1):
    finaldata.append(printlist[i][:-2] + [printlist[i][-2], I_c_pred[i], I_c_change[i], printlist[i][-1], I_d_pred[i], I_d_change[i]])
    
# print(printlist , '\n')
# print(I_d_pred, '\n', I_c_pred)

# print(finaldata)

dataexcel = OrderedDict()
dataexcel.update({"Sheet 1": finaldata})
save_data("cdData1.ods", dataexcel)
export_graphviz(regressor_d, out_file ='d_tree1.dot', feature_names =(finaldata[0])[1:-6]) 
export_graphviz(regressor_c, out_file ='c_tree1.dot', feature_names =(finaldata[0])[1:-6]) 






#Neural networks
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


# def custom_loss(y_true, y_pred):
#     # Define weights
#     false_positive_weight = 1.0
#     false_negative_weight = 10.0

#     # Calculate binary cross entropy
#     bce = tf.keras.losses.BinaryCrossentropy()

#     # Calculate loss
#     loss = bce(y_true, y_pred)

#     # Calculate weighted loss
#     weighted_loss = tf.where(tf.greater(y_true, y_pred), false_negative_weight * loss, false_positive_weight * loss)

#     return tf.reduce_mean(weighted_loss)

def custom_loss(y_true, y_pred):
    loss = tf.maximum(y_pred - y_true, 0) * 1 + tf.maximum(y_true - y_pred, 0) * 100
    return tf.reduce_mean(loss)


# split into input (X) and output (Y) variables
X = X.tolist()
# Y = [ [y_d[i], y_c[i]] for i in range(len(y_d))  ]
Y1 = [ [y_d[i]] for i in range(len(y_d))  ]
Y2 = [ [y_c[i]] for i in range(len(y_c))  ]

NN_I_d_pred = ['I.d_predicted']
NN_I_c_pred = ['I.c_predicted']
NN_I_c_change = ['I.c_change']
NN_I_d_change = ['I.d_change']


# create model for d
model1 = Sequential()
# Create Hidden Layers
model1.add(Dense(units=50, input_dim=len(X[0]), activation='relu'))
# model1.add(Dense(8, init='uniform', activation='relu'))
model1.add(Dense(units=1, activation='sigmoid')) 
# Compile model
model1.compile(loss=custom_loss, optimizer='adam') # Define custom loss function, previously loss='mean_squared_error'
# Fit the model
model1.fit(X, Y1, epochs=10000, batch_size=25)

outputs1 = model1.predict(X)
for (i,data) in enumerate(X):
    output1 = outputs1[i]
    dpred = float(output1[0])
    # cpred = float(output1[1])
    NN_I_d_pred.append( dpred )
    # NN_I_c_pred.append( cpred )
    # NN_I_c_change.append(cpred - float(y_c[i]))
    NN_I_d_change.append(dpred - float(y_d[i]))

# create new model for c
model2 = Sequential()
# Create Hidden Layers
model2.add(Dense(units=50, input_dim=len(X[0]), activation='relu'))
# model1.add(Dense(8, init='uniform', activation='relu'))
model2.add(Dense(units=1, activation='sigmoid'))
# Compile model
model2.compile(loss=custom_loss, optimizer='adam') # Define custom loss function, previously loss='mean_squared_error'
# Fit the model
model2.fit(X, Y2, epochs=10000, batch_size=25)

outputs2 = model1.predict(X)
for (i,data) in enumerate(X):
    output2 = outputs2[i]
    # dpred = float(output2[0])
    cpred = float(output2[0])
    # NN_I_d_pred.append( dpred )
    NN_I_c_pred.append( cpred )
    NN_I_c_change.append(cpred - float(y_c[i]))
    # NN_I_d_change.append(dpred - float(y_d[i]))


finaldata = []
for i in range(len(printlist) - 1):
    finaldata.append(printlist[i][:-2] + [printlist[i][-2], NN_I_c_pred[i], NN_I_c_change[i], printlist[i][-1], NN_I_d_pred[i], NN_I_d_change[i]])
    
# print(printlist , '\n')
# print(I_d_pred, '\n', I_c_pred)

# print(finaldata)

dataexcel = OrderedDict()
dataexcel.update({"Sheet 1": finaldata})
save_data("cdData1_NN.ods", dataexcel)

