import math as m
import enum

class ModelType(enum.Enum):
    SingleInputSingleOutput1 = 1
    SingleInputSingleOutput2 = 2
    TwoInputTwoOutput = 3

def generate_data(length:int,data_size:int,Control_complexity:bool,model_type: ModelType):
    if length <= 2:
        return
    Y = []
    U = []
    AppendInitialControl(U,data_size,Control_complexity)
    if data_size == 2:
        Y = [[],[]]
        U = [[],[]]
        if(model_type.value != 3):
            return
    Y = AppendInitialSystem(Y,U,model_type,data_size)

    for k in range(length - 2):
        #Generate U
        AppendControl(U,data_size,Control_complexity,k)
        #Generate Y
        AppendSystem(Y,U,model_type)


    print(Y)
    print(U)
    return [Y,U]


#
#       Control GENERATION
#
def AppendControl(u,data_size:int,complexity:bool,k:int):
    if  data_size == 1:
        if complexity:
            SingleInputComplexSin(u,k)
        else:
            SingleInputSimpleSin(u,k)
    elif data_size == 2:
        if complexity:
            TwoImputComplexSin(u,k)
        else:
            TwoImputSimpleSin(u,k)

def AppendInitialControl(u,data_size:int,complexity:bool):
    if  data_size == 1:
        if complexity:
            SingleInputComplexSin(u,-2)
            SingleInputComplexSin(u,-1)
        else:
            SingleInputSimpleSin(u,-2)
            SingleInputSimpleSin(u,-1)
    elif data_size == 2:
        if complexity:
            TwoImputComplexSin(u,-2)
            TwoImputComplexSin(u,-1)
        else:
            TwoImputSimpleSin(u,-2)
            TwoImputSimpleSin(u,-1)

def SingleInputSimpleSin(u,k):
    u.append(m.sin(m.pi * k /10))

def SingleInputComplexSin(u,k):
    u.append(3 * m.sin(m.pi * k /5) + 2 * m.cos(m.pi * k / 10))



def TwoImputSimpleSin(u,k):
    u[0].append(m.sin(m.pi * k /10))
    u[1].append(m.cos(m.pi * k /10))


def TwoImputComplexSin(u,k):
    u[0].append(2 * m.cos(m.pi * k /5) + m.sin(m.pi * k /10))
    u[1].append(2 * m.sin(m.pi * k /5) + m.cos(m.pi * k /10))

#
#       System GENERATION
#
def AppendSystem(Y,U,model:ModelType):
    if model.value == 1:
        SISO1(Y,U)
    if model.value == 2:
        SISO2(Y,U)
    if model.value == 3:
        TITO(Y,U)

def AppendInitialSystem(Y,U,model:ModelType,data_size:int):
    if data_size == 1:
        Y.append(0)
        Y.append(0)
    if data_size == 2:
        Y[0].append(0)
        Y[0].append(0)
        Y[1].append(0)
        Y[1].append(0)
    
    if model.value == 1:
        SISO1(Y,U)
        SISO1(Y,U)
    if model.value == 2:
        SISO2(Y,U)
        SISO2(Y,U)
    if model.value == 3:
        TITO(Y,U)
        TITO(Y,U)
    
    return Y[2:len(Y)]


def SISO1(Y,u):
    Y.append((Y[-1])/(pow(2,Y[-1]) + 1) + pow(u[-1],3)/(pow(Y[-1],2) + 1) - u[-1])

def SISO2(Y,u):
    Y.append((Y[-1] * Y[-2] * (Y[-1] + 2.5))  /  (1 + Y[-1] * Y[-1] *Y[-2] * Y[-2]) + u[-1])

def TITO(Y,U):
    y1 = Y[0]
    y2 = Y[1]
    Y[0].append( (y1[-1])          / (1 + pow(y2[-1],2)) + U[0][-1] )
    Y[1].append( (y1[-1] * y2[-1]) / (1 + pow(y2[-1],2)) + U[1][-1] )

data = generate_data(10,1,False,ModelType.SingleInputSingleOutput1)
print("")
print("")
print(data)