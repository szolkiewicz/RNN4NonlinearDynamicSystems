import math as m
import enum
import matplotlib.pyplot as plt
import pandas as pd

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
    u.append(2 * m.sin(m.pi * k /10))

def SingleInputComplexSin(u,k):
    u.append(1.75 * m.sin(m.pi * k /5) + 1.5 * m.cos(m.pi * k / 10))



def TwoImputSimpleSin(u,k):
    u[0].append(2 * m.sin(m.pi * k /10))
    u[1].append(2 * m.cos(m.pi * k /10))


def TwoImputComplexSin(u,k):
    u[0].append(1.75 * m.cos(m.pi * k /5) + 1.5 * m.sin(m.pi * k /10))
    u[1].append(1.75 * m.sin(m.pi * k /5) + 1.5 * m.cos(m.pi * k /10))

#
#       System GENERATION
#
def SISO1(Y,u):
    Y.append((Y[-1])/(pow(2,Y[-1]) + 1) + pow(u[-1],3)/(pow(Y[-1],2) + 1) - u[-1])

def SISO2(Y,u):
    Y.append((Y[-1] * Y[-2] * (Y[-1] + 2.5))  /  (1 + Y[-1] * Y[-1] *Y[-2] * Y[-2]) + u[-1])

def TITO(Y,U):
    y1 = Y[0]
    y2 = Y[1]
    Y[0].append( (y1[-1])          / (1 + pow(y2[-1],2)) + U[0][-1] )
    Y[1].append( (y1[-1] * y2[-1]) / (1 + pow(y2[-1],2)) + U[1][-1] )

class ModelType(enum.Enum):
    SingleInputSingleOutput1 = SISO1
    SingleInputSingleOutput2 = SISO2
    TwoInputTwoOutput = TITO

def AppendSystem(Y,U,model:ModelType):
    model(Y,U)

def AppendInitialSystem(Y,U,model:ModelType,data_size:int):
    if data_size == 1:
        Y.append(0)
        Y.append(0)
    if data_size == 2:
        Y[0].append(0)
        Y[0].append(0)
        Y[1].append(0)
        Y[1].append(0)
    
    model(Y,U)
    model(Y,U)

    if data_size == 2:
        Y = [Y[0][2:len(Y)],Y[1][2:len(Y)]]
        return
    
    Y = Y[2:len(Y)]


def generate_data(length:int,Control_complexity:bool,model_type: ModelType):
    """function generate data of given model_type, length and input_complexity

    Args:
        length (int): hryzon na jaki generujemy dane
        Control_complexity (bool): wybór wartości dotyczących sterowania na razie używamy sterowania typu sinusoidalnego, bez załkuceń o 2 złorzonościach 
        model_type (ModelType): wybór typu modelu jaki zostanie użyty do generowania danych 

    Returns:
        _type_: zip złorzony z par (Y_n,U_n)
    """
    if length <= 2:
        return
    if model_type == ModelType.TwoInputTwoOutput:
        data_size = 2
    else:
        data_size = 1
    Y = []
    U = []
    if data_size == 2:
        Y = [[],[]]
        U = [[],[]]
    AppendInitialControl(U,data_size,Control_complexity)
    AppendInitialSystem(Y,U,model_type,data_size)

    for k in range(length - 2):
        #Generate U
        AppendControl(U,data_size,Control_complexity,k)
        #Generate Y
        AppendSystem(Y,U,model_type)

    if data_size == 2:
        zip(zip(Y[0], U[0]),zip(Y[1], U[1]))
    return zip(Y,U)


data = generate_data(1000,True,ModelType.SingleInputSingleOutput1)
print("")
print("Wyświetlamy wygenerowane dane:")
print("niebiseki - model   output")
print("czerw     - control output")
print("")
unzippedData = []
unzippedControl = []
for elem in tuple(data): 
    unzippedData.append(elem[0])
    unzippedControl.append(elem[1])
# plt.plot(unzippedData,c='b')
# plt.plot(unzippedControl, c='r')
# plt.show()

# na szybko dla 1 zm stanu i 1 sterowania, ps czym jest ta abominacja, co tak sie rozpakowuje strasznie
# Tworzenie DataFrame z danymi
df = pd.DataFrame({'x1': unzippedData, 'u1': unzippedControl})

# Zapis do pliku CSV
df.to_csv('Data/generated_data.csv', index=False)
