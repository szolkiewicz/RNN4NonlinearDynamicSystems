import math as m
import enum
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

#
#       Control GENERATION
#

Control_Types = ["Sin","SinComplex","UnitJump","DiracDelta","Const"]

def AppendControl(u,data_size:int,complexity:str,k:int,jump_value:int,const_value:int = 0,dist_val: int = 0):
    if  data_size == 1:
        if complexity == "Sin":
            SingleInputSimpleSin(u,k)
        elif complexity == "SinComplex":
            SingleInputComplexSin(u,k)
        elif complexity == "UnitJump":
            SingleInputConst(u,jump_value)
        elif complexity == "DiracDelta":
            SingleInputConst(u,0)
        elif complexity == "Const":
            SingleInputConst(u,const_value)
        SingleDistraction(u,dist_val)
    elif data_size == 2:
        if complexity == "Sin":
            TwoImputSimpleSin(u,k)
        elif complexity == "SinComplex":
            TwoImputComplexSin(u,k)
        elif complexity == "UnitJump":
            TwoInputConst(u,jump_value)
        elif complexity == "DiracDelta":
            TwoInputConst(u,0)
        elif complexity == "Const":
            TwoInputConst(u,const_value)
        TwoDistraction(u,dist_val)

def AppendInitialControl(u,data_size:int,complexity:bool,jump_value:int,const_value:int = 0,dist_val: int = 0):
    if  data_size == 1:
        if complexity == "Sin":
            SingleInputSimpleSin(u,-2)
            SingleDistraction(u,dist_amplitude=dist_val)
            SingleInputSimpleSin(u,-1)
        elif complexity == "SinComplex":
            SingleInputComplexSin(u,-2)
            SingleDistraction(u,dist_amplitude=dist_val)
            SingleInputComplexSin(u,-1)
        elif complexity == "UnitJump":
            SingleInputConst(u,0)
            SingleDistraction(u,dist_amplitude=dist_val)
            SingleInputConst(u,0)
        elif complexity == "DiracDelta":
            SingleInputConst(u,0)
            SingleDistraction(u,dist_amplitude=dist_val)
            SingleInputConst(u,jump_value)
        elif complexity == "Const":
            SingleInputConst(u,const_value)
            SingleDistraction(u,dist_amplitude=dist_val)
            SingleInputConst(u,const_value)
        SingleDistraction(u,dist_amplitude=dist_val)
    elif data_size == 2:
        if complexity == "Sin":
            TwoImputSimpleSin(u,-1)
            TwoDistraction(u,dist_val)
            TwoImputSimpleSin(u,-2)
        elif complexity == "SinComplex":
            TwoImputComplexSin(u,-1)
            TwoDistraction(u,dist_val)
            TwoImputComplexSin(u,-2)
        elif complexity == "UnitJump":
            TwoInputConst(u,0)
            TwoDistraction(u,dist_val)
            TwoInputConst(u,0)
        elif complexity == "DiracDelta":
            TwoInputConst(u,0)
            TwoDistraction(u,dist_val)
            TwoInputConst(u,jump_value)
        elif complexity == "Const":
            TwoInputConst(u,const_value)
            TwoDistraction(u,dist_val)
            TwoInputConst(u,const_value)
        TwoDistraction(u,dist_val)

def SingleInputSimpleSin(u,k):
    u.append(2 * m.sin(m.pi * k /10))

def SingleInputComplexSin(u,k):
    u.append(1.75 * m.sin(m.pi * k /5) + 1.5 * m.cos(m.pi * k / 10))

def SingleInputConst(u,const:int):
    u.append(const)

def SingleDistraction(u,dist_amplitude:int):
    u[-1] += random.normalvariate(0,1) * dist_amplitude

def GenerateSignalNoise(len,len2, amplitude):
    Z = ((np.random.rand(len,len2)*2)-1)*amplitude
    return list(np.array(Z))

def TwoImputSimpleSin(u,k):
    u[0].append(2 * m.sin(m.pi * k /10))
    u[1].append(2 * m.cos(m.pi * k /10))


def TwoImputComplexSin(u,k):
    u[0].append(1.75 * m.cos(m.pi * k /5) + 1.5 * m.sin(m.pi * k /10))
    u[1].append(1.75 * m.sin(m.pi * k /5) + 1.5 * m.cos(m.pi * k /10))

def TwoInputConst(u,const:int):
    u[0].append(const)
    u[1].append(const)

def TwoDistraction(u,dist_amplitude:int):
    u[0][-1] += random.normalvariate(0,1) * dist_amplitude
    u[1][-1] += random.normalvariate(0,1) * dist_amplitude


#
#       System GENERATION
#
def SISO1(Y,u):
    Y.append((Y[-1])/(pow(2,Y[-1]) + 1) + pow(u[-1],2) - u[-2])

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
    
    #model(Y,U)
    #model(Y,U)

    #if data_size == 2:
    #    Y = [Y[0][2:len(Y)],Y[1][2:len(Y)]]
    #    return
    
    #Y = Y[2:len(Y)]


def generate_data(length:int,Control_type:str,model_type: ModelType,const_value :int = 0,disruption_amplitude: int = 0, output_noise: int = 0,jump_value:int = 1, filename_to_save:str = "generated_data"):
    """function generate data of given model_type, length and input_complexity

    Args:
        length (int): hryzon na jaki generujemy dane
        Control_type (string): wpisujemy jeden z wybranych z tabeli: ["Sin","SinComplex","UnitJump","DiracDelta","Const"]. UWAGA jak chcesz sam szum to wybierasz  Const i nie zmieniasz const_value
        model_type (ModelType): wybór typu modelu jaki zostanie użyty do generowania danych 
        const_value (int): Zmieniasz gdy chcesz mieć jakąś stałą wartość podawaną na wejście (nie zmieniasz gdy podajesz sam szum)
        disruption_amplitude (int): ustawia ampltude zakłuceń jakie pojawią się w sterwaniu  
        iump_value (int): określa wysokość zmiany delty diracka albo skoku jednostkowego (uwaga mi raz wykoleiło system chyba 2 z rowerka (trzeba uważać ))
        filename_to_save (str): nazwa pliku pod jaką zostaną zapisane dane 

    UWAGI:
        Fajny ten SISO1 bo jest nie stabilny dla const wejścia ;-; trzeba będzie go zmienić raczej używamy SISO2. 

    Returns:
        _type_: zip złorzony z par (Y_n,U_n)
    """
    #
    # Sprawdzania poprawności typu sterowania
    #
    if Control_type not in Control_Types:
        return [[],[]]

    if length <= 2:
        return
    #
    # ustawianie parametru rozmiaru danych
    #
    if model_type == ModelType.TwoInputTwoOutput: 
        data_size = 2
    else:
        data_size = 1
    Y = []
    U = []
    if data_size == 2:
        Y = [[],[]]
        U = [[],[]]
        
        
    AppendInitialControl(U,data_size,Control_type,const_value=const_value,jump_value=jump_value,dist_val=disruption_amplitude)
    AppendInitialSystem(Y,U,model_type,data_size)
    print("preU:")
    print(len(U))
    print("preY:")
    print(len(Y))
    for k in range(length - 2):
        #Generate U
        AppendControl(U,data_size,Control_type,k,const_value=const_value,jump_value=jump_value,dist_val=disruption_amplitude)
        #Generate Y
        AppendSystem(Y,U,model_type)
    print("postU:")
    print(len(U))
    print("postY:")
    print(len(Y))
    Z = GenerateSignalNoise(1,len(Y),output_noise)[0]
    if data_size == 2:
        Z = GenerateSignalNoise(len(Y),len(Y[0]),output_noise)
    
    if data_size == 2:
        Y = [list(l) for l in Y]
        df = pd.DataFrame({'Y1': Y[0], 'U1': U[0],'Y2': Y[1], 'U2': U[1]})
        df.to_csv('Data/'+filename_to_save+'.csv', index=False)
        return zip(zip(Y[0], U[0]),zip(Y[1], U[1]))
    print(Y)
    df = pd.DataFrame({'Y': Y, 'U': U})
    df.to_csv('Data/'+filename_to_save+'.csv', index=False)
    return zip(Y,U)


data = generate_data(1000,"Const",ModelType.SingleInputSingleOutput1,disruption_amplitude=0.5,output_noise=0.1,filename_to_save="learn_noise_siso2")
print("")
print("Wyświetlamy wygenerowane dane:")
print("niebiseki - model   output")
print("czerw     - control output")
print("")
unzippedData = []
unzippedControl = []
for elem in tuple(data): 
    # print(elem)
    unzippedData.append(elem[0])
    unzippedControl.append(elem[1])

# SingleInputSingleOutput1

# fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

# ax.plot(unzippedData,c='b',label="Output")
# ax.plot(unzippedControl, c='r',label="Input")
# ax.legend() 
# plt.grid(True)
# plt.show()

# na szybko dla 1 zm stanu i 1 sterowania, ps czym jest ta abominacja, co tak sie rozpakowuje strasznie
# Tworzenie DataFrame z danymi
#df = pd.DataFrame({'y1': unzippedData, 'u1': unzippedControl})

# Zapis do pliku CSV
#df.to_csv('Data/generated_data.csv', index=False)
