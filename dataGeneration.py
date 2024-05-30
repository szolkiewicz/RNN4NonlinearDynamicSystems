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
        InputSimpleSin(u,k,const=const_value)
    elif complexity == "SinComplex":
        InputComplexSin(u,k,const_value)
    elif complexity == "UnitJump":
        InputConst(u,const_value)
    elif complexity == "DiracDelta":
        InputConst(u,0)
    elif complexity == "Const" or complexity == "RandomUnitJumps":
        InputConst(u,const_value)
    Distraction(u,dist_val)

def AppendInitialControl(u,complexity:bool,const_value:int = 0,dist_val: int = 0):
    if complexity == "Sin":
        InputSimpleSin(u,-2,const_value)
        Distraction(u,dist_amplitude=dist_val)
        InputSimpleSin(u,-1,const_value)
    elif complexity == "SinComplex":
        InputComplexSin(u,-2,const_value)
        Distraction(u,dist_amplitude=dist_val)
        InputComplexSin(u,-1,const_value)
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

def InputSimpleSin(u,k,const):
    u.append(2 * m.sin(m.pi * k /10) + const)

def InputComplexSin(u,k,const):
    u.append(1.75 * m.sin(m.pi * k /5) + 1.5 * m.cos(m.pi * k / 10) + const)

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


def generate_data(n:int,Control_type:str,model_type: ModelType,const :int = 0,disruption_amplitude: int = 0, output_noise_scale: int = 0,jump_value:int = 0,y_base_value: int = 0):
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
    swap = random.randrange(15,length - 15)
    AppendInitialControl(U,Control_type,const_value=const_value,dist_val=disruption_amplitude)
    AppendInitialSystem(Y,U,model_type,y_base_value)

    for k in range(length - 2):
        if(k == swap):
            const_value = jump_value
        #Generate U
        AppendControl(U,Control_type,k,const_value=const_value,dist_val=disruption_amplitude)
        max_finded_falue = max(max_finded_falue,abs(U[-1]))
        #Generate Y
        AppendSystem(Y,U,model_type)
        Distraction(Y,abs(Y[-1]*output_noise_scale))
        max_finded_falue = max(max_finded_falue,abs(Y[-1]))
    return (zip(Y,U),max_finded_falue)



def GenerateData(filename:str = "",plot:bool = True,iteration_length:int = 1,sin_on:bool = False, complex_sin_on:bool = False, jumps_on:bool = False,sin_jumps_on:bool = False,complex_sin_jumps_on:bool = False, data100_on1jump:int = 1,output_noice_percentage = 0.02):
    maxes = []
    prev_y = 0
    AllData = []
    #Generowanie próbek wyjścia sinusa
    if sin_on:
        data, max = generate_data(iteration_length,"Sin",ModelType.InputOutput2,const=0,disruption_amplitude=0,output_noise_scale=output_noice_percentage,y_base_value=prev_y)
        AllData += tuple(data)
        prev_y = AllData[-1][0]
        maxes.append(max)
    #Generowanie próbek wyjścia complex_sinusa
    if complex_sin_on:
        data, max = generate_data(iteration_length,"SinComplex",ModelType.InputOutput2,const=0,disruption_amplitude=0,output_noise_scale=output_noice_percentage,y_base_value=prev_y)
        AllData += tuple(data)
        prev_y = AllData[-1][0]
        maxes.append(max)
    #Generowanie próbek z serowaniem skokowym
    if jumps_on:
        if len(AllData) != 0:
            prev_u = AllData[-1][1]
        else:

            prev_u = random.uniform(-3, 3)
        new_u = random.uniform(-3, 3)
        for _ in range(0,iteration_length):
            while (abs(new_u - prev_u) < 0.5):
                new_u = random.uniform(-3, 3)
            data,max = generate_data(data100_on1jump,"RandomUnitJumps",ModelType.InputOutput2,const=prev_u,disruption_amplitude=0,output_noise_scale=output_noice_percentage,jump_value=new_u,y_base_value=prev_y)
            prev_u = new_u
            AllData += tuple(data)
            prev_y = AllData[-1][0]
    #Generowanie próbek z serowaniem skokowym + sin
    if sin_jumps_on:
        if len(AllData) != 0:
            prev_u = AllData[-1][1]

        else:
            prev_u = random.uniform(-3, 3)
        new_u = random.uniform(-3, 3)
        for _ in range(0,iteration_length):
            while (abs(new_u - prev_u) < 0.5):
                new_u = random.uniform(-3, 3)
            data,max = generate_data(data100_on1jump*2,"Sin",ModelType.InputOutput2,const=prev_u,disruption_amplitude=0,output_noise_scale=output_noice_percentage,jump_value=new_u,y_base_value=prev_y)
            prev_u = new_u
            AllData += tuple(data)
            prev_y = AllData[-1][0]
        #Generowanie próbek z serowaniem skokowym + complexSin
    if complex_sin_jumps_on:
        if len(AllData) != 0:
            prev_u = AllData[-1][1]
        else:
            prev_u = random.uniform(-3, 3)
        new_u = random.uniform(-3, 3)
        for _ in range(0,iteration_length):
            while (abs(new_u - prev_u) < 0.5):
                new_u = random.uniform(-3, 3)
            data,max = generate_data(data100_on1jump*2,"SinComplex",ModelType.InputOutput2,const=prev_u,disruption_amplitude=0,output_noise_scale=output_noice_percentage,jump_value=new_u,y_base_value=prev_y)
            prev_u = new_u
            AllData += tuple(data)
            prev_y = AllData[-1][0]
    #print(AllData)
    Y = []
    U = []
    for a in AllData:
        Y.append(a[0])
        U.append(a[1])
    if filename != "":
        df = pd.DataFrame({'Y': Y, 'U': U})
        df.to_csv('Data/'+filename+'.csv', index=False)
    if plot:
        print("")
        print("Wyświetlamy wygenerowane dane:")
        print("niebiseki - model   output")
        print("czerw     - control output")
        print("")
        unzippedData = []
        unzippedControl = []
        #print(JumpsData)
        for elem in AllData: 
            unzippedData.append(elem[0])
            unzippedControl.append(elem[1])

        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

        ax.plot(unzippedData,c='b',label="Output")
        ax.plot(unzippedControl, c='r',label="Input")
        ax.legend() 
        plt.grid(True)
        plt.show()

    return

#UWAGA zostanie wygenerowanych 50 razy tyle danych
ILE_DANYCH = 4 # XD 
# GenerateData(iteration_length=ILE_DANYCH,sin_on=True,filename="sin_input_data")
# GenerateData(iteration_length=ILE_DANYCH,complex_sin_on=True,filename="complex_sin_input_data")
# GenerateData(iteration_length=ILE_DANYCH,jumps_on=True,filename="jumps_input_data")
GenerateData(iteration_length=ILE_DANYCH,sin_jumps_on=True,filename="a")
# GenerateData(iteration_length=ILE_DANYCH,complex_sin_jumps_on=True,filename="complexANDjumps_sin_input_data")

#GenerateData(iteration_length=ILE_DANYCH,sin_on=False,complex_sin_on=False,jumps_on=False,sin_jumps_on=False,complex_sin_jumps_on=False,plot=False,filename="")
# ntimer zmieniamy tylko dla sygnałów bez skoków
# iteration_length dla sygnałów ze skokami 




# UWAGA UWAGA 
#
#  jeżeli przybyłeś tu w poszukiwaniu odpowieni na to jak generować dane i jak to działa.
#
# 1)nie pytaj czemu to tak chujowo działa. Przeszło to przez parę koncepcji(części z nich już nawet nie pamiętam) jest 22:47 i mam już dość kombinowania.
#
# 2) To generuje dane tylko w 1 sposób. nie przeplata różnego rodzaju sterowań, jak wygenereujesz więcej sterowań na raz to je sklei w data-ludzką-stonoge. Jak chcesz mega plik z danymi podajesz |iteration_length = n| i dostajesz dane o długości 50*n(opróbek) UWAGA dla każdy typ sterowania na true (dwa typy sterwoania n == 2, dostajesz 100 próbek jednego i 100 próbek drugiego)
#
# data100_on1jump - pozwala ci dać rzadziej skoki w danych tj dla 1 - sok na każde 50 danych uwaga jak to zwiększysz rośnie ilość danych generowana "data100_on1jump" razy
#
#
# chcesz inne zaszumienie?  >>> zmieniasz "output_noice_percentage" << UWAGA TO JE PROCENT AKUTALNEGO WYJŚCIA
#
#
# lepiej kożystać z plików z danymi 
#
#
# jak chcesz nowe dane starego typu just odpal ten program. Polecam jedynie manipulować wsp. iteration_length i traktować go jako oczekiwana długość danych podzielona przez 50 50
#
#
#
#
