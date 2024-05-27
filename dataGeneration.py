import math as m
import enum
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

#
#       Control GENERATION
#

Control_Types = ["Sin","SinComplex","UnitJump","RandomUnitJumps","DiracDelta","Const"]

def AppendControl(u,complexity:str,k:int,const_value:int = 0,dist_val: int = 0):
    if complexity == "Sin":
        InputSimpleSin(u,k)
    elif complexity == "SinComplex":
        InputComplexSin(u,k)
    elif complexity == "UnitJump":
        InputConst(u,const_value)
    elif complexity == "DiracDelta":
        InputConst(u,0)
    elif complexity == "Const" or complexity == "RandomUnitJumps":
        InputConst(u,const_value)
    Distraction(u,dist_val)

def AppendInitialControl(u,complexity:bool,const_value:int = 0,dist_val: int = 0):
    if complexity == "Sin":
        InputSimpleSin(u,-2)
        Distraction(u,dist_amplitude=dist_val)
        InputSimpleSin(u,-1)
    elif complexity == "SinComplex":
        InputComplexSin(u,-2)
        Distraction(u,dist_amplitude=dist_val)
        InputComplexSin(u,-1)
    elif complexity == "UnitJump":
        InputConst(u,0)
        Distraction(u,dist_amplitude=dist_val)
        InputConst(u,0)
    elif complexity == "DiracDelta":
        InputConst(u,0)
        Distraction(u,dist_amplitude=dist_val)
        InputConst(u,const_value)
    elif complexity == "Const" or complexity == "RandomUnitJumps":
        InputConst(u,const_value)
        Distraction(u,dist_amplitude=dist_val)
        InputConst(u,const_value)

    Distraction(u,dist_amplitude=dist_val)

def InputSimpleSin(u,k):
    u.append(2 * m.sin(m.pi * k /10))

def InputComplexSin(u,k):
    u.append(1.75 * m.sin(m.pi * k /5) + 1.5 * m.cos(m.pi * k / 10))

def InputConst(u,const:int):
    u.append(const)

def Distraction(u,dist_amplitude:int):
    u[-1] += random.uniform(-1,1) * dist_amplitude

def GenerateSignalNoise(len,len2, amplitude):
    Z = ((np.random.rand(len,len2)*2)-1)*amplitude
    return list(np.array(Z))


#
#       System GENERATION
#
def SISO1(Y,u):
    Y.append((Y[-1])/(pow(2,Y[-1]) + 1) + pow(u[-1],2) - u[-2])

def SISO2(Y,u):
    Y.append((Y[-1] * Y[-2] * (Y[-1] + 2.5))  /  (1 + Y[-1] * Y[-1] *Y[-2] * Y[-2]) + u[-1])

class ModelType(enum.Enum):
    InputOutput1 = SISO1
    InputOutput2 = SISO2

def AppendSystem(Y,U,model:ModelType):
    model(Y,U)

def AppendInitialSystem(Y,U,model:ModelType,y_base:int):
    Y.append(y_base)
    Y.append(y_base)
    
    
    #model(Y,U)
    #model(Y,U)

    #if data_size == 2:
    #    Y = [Y[0][2:len(Y)],Y[1][2:len(Y)]]
    #    return
    
    #Y = Y[2:len(Y)]


def generate_data(n:int,Control_type:str,model_type: ModelType,const :int = 0,disruption_amplitude: int = 0, output_noise: int = 0, output_noise_scale: int = 0,jump_value:int = 1,y_base_value: int = 0):
    """function generate data of given model_type, length and input_complexity

    Args:
        length (int): hryzon na jaki generujemy dane
        Control_type (string): wpisujemy jeden z wybranych z tabeli: ["Sin","SinComplex","UnitJump","DiracDelta","Const"]. UWAGA jak chcesz sam szum to wybierasz  Const i nie zmieniasz const_value
        model_type (ModelType): wybór typu modelu jaki zostanie użyty do generowania danych 
        const_value (int): Zmieniasz gdy chcesz mieć jakąś stałą wartość podawaną na wejście (nie zmieniasz gdy podajesz sam szum)
        disruption_amplitude (int): ustawia ampltude zakłuceń jakie pojawią się w sterwaniu  
        iump_value (int): określa wysokość zmiany delty diracka albo skoku jednostkowego (uwaga mi raz wykoleiło system chyba 2 z rowerka (trzeba uważać ))
        output_noise_scale (double) - dodaje szum o wartosci procentowej do aktualnego wyjścia jf wyjście 

    Returns:
        TUPLA{zip złorzony z par (Y_n,U_n)<<"to co było" , scale_value(FLOAT) <<"wartość przez jaką przeskalowane jest wyjście i sterowanie"}
    """
    if Control_type not in Control_Types:
        return [[],[]]

    length = n * 50

    Y = []
    U = []
    const_value = const
        
    max_finded_falue = 0.0
    swap = random.randrange(15,35 + 50 * (n-1))
    AppendInitialControl(U,Control_type,const_value=const_value,dist_val=disruption_amplitude)
    AppendInitialSystem(Y,U,model_type,y_base_value)
    for k in range(length - 2):
        if(k == swap and Control_type == "RandomUnitJumps" ):
            const_value = jump_value
        #Generate U
        AppendControl(U,Control_type,k,const_value=const_value,dist_val=disruption_amplitude)
        max_finded_falue = max(max_finded_falue,abs(U[-1]))
        #Generate Y
        AppendSystem(Y,U,model_type)
        Distraction(Y,abs(Y[-1]*output_noise_scale))
        max_finded_falue = max(max_finded_falue,abs(Y[-1]))
    return (zip(Y,U),max_finded_falue)



def GenerateData(filename:str = "temp", ntimer:int = 1,):
    n = 2 * ntimer
    maxes = []
    #Generowanie ntimer * 100 próbek wyjścia sinusa
    data, max = generate_data(n,"Sin",ModelType.InputOutput2,const=0,disruption_amplitude=0,output_noise=0,output_noise_scale=0)
    SinusData = tuple(data)
    maxes.append(max)
    #Generowanie ntimer * 100 próbek wyjścia complex_sinusa
    data, max = generate_data(n,"SinComplex",ModelType.InputOutput2,const=0,disruption_amplitude=0,output_noise=0,output_noise_scale=0)
    ComplexSinData = tuple(data)
    maxes.append(max)
    #Generowanie ntimer * 100
    JumpsData = []
    prev_y = 0
    prev_u = random.uniform(-4, 4)
    new_u = random.uniform(-4, 4)
    for i in range(0,ntimer):
        while (new_u == prev_u):
            new_u = random.uniform(-4, 4)
        data,max = generate_data(2,"RandomUnitJumps",ModelType.InputOutput2,const=prev_u,disruption_amplitude=0,output_noise=0,output_noise_scale=0,jump_value=new_u,y_base_value=prev_y)
        prev_u = new_u
        Data = tuple(data)
        prev_y = Data[-1][0]
        maxes.append(max)
        JumpsData += Data

    AllData = SinusData + ComplexSinData + tuple(JumpsData)
    print("")
    print("Wyświetlamy wygenerowane dane:")
    print("niebiseki - model   output")
    print("czerw     - control output")
    print("")
    unzippedData = []
    unzippedControl = []
    #print(JumpsData)
    for elem in AllData: 
        print(elem)
        unzippedData.append(elem[0])
        unzippedControl.append(elem[1])

    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

    ax.plot(unzippedData,c='b',label="Output")
    ax.plot(unzippedControl, c='r',label="Input")
    ax.legend() 
    plt.grid(True)
    plt.show()

    return


GenerateData(ntimer=3)
#["Sin","SinComplex","UnitJump","RandomUnitJumps","DiracDelta","Const"]
# data,maxValue = generate_data(2,"RandomUnitJumps",ModelType.InputOutput2,const=3,disruption_amplitude=0,output_noise=0,output_noise_scale=0,jump_value=-1)
# Data = tuple(data)
# data2,maxValue = generate_data(2,"RandomUnitJumps",ModelType.InputOutput2,const=-1,disruption_amplitude=0,output_noise=0,output_noise_scale=0,jump_value=2,y_base_value=Data[-1][0])
# Data2 = tuple(data2)
# data3,maxValue = generate_data(2,"RandomUnitJumps",ModelType.InputOutput2,const=2,disruption_amplitude=0,output_noise=0,output_noise_scale=0,jump_value=-3,y_base_value=Data2[-1][0])
# print("")
# print("Wyświetlamy wygenerowane dane:")
# print("niebiseki - model   output")
# print("czerw     - control output")
# print("")
# print(maxValue)
# unzippedData = []
# unzippedControl = []
# Data = Data + Data2 + tuple(data3)
# for elem in tuple(data): 
#     # print(elem)
#     unzippedData.append(elem[0])
#     unzippedControl.append(elem[1])

# fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

# ax.plot(unzippedData,c='b',label="Output")
# ax.plot(unzippedControl, c='r',label="Input")
# ax.legend() 
# plt.grid(True)
# plt.show()

# fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')


# na szybko dla 1 zm stanu i 1 sterowania, ps czym jest ta abominacja, co tak sie rozpakowuje strasznie
# Tworzenie DataFrame z danymi
#df = pd.DataFrame({'y1': unzippedData, 'u1': unzippedControl})

# Zapis do pliku CSV
#df.to_csv('Data/generated_data.csv', index=False)
