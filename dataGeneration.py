import math as m


def generate_data(length, ):
    if length <= 0:
        return
    data = []

        #Generate U
        #Generate Y

    return data


#
#       Control GENERATION
#

def SingleInputSimpleSin(u,k):
    u.append(m.sin(m.pi * k /10))

def SingleInputComplexSin(u,k):
    u.append(3 * m.sin(m.pi * k /5) + 2 * m.cos(m.pi * k / 10))



def TwoImputSimpleSin(u,k):
    u[0].append(m.sin(m.pi * k /10))
    u[1].append(m.cos(m.pi * k /10))

#
#       System GENERATION
#

def SISO1(Y,u):
    Y.append((Y[-1])/(pow(2,Y[-1]) + 1) + pow(u[-1],3)/(pow(Y[-1],2) + 1) - u)

def SISO2(Y,u):
    Y.append((Y[-1] * Y[-2] * (Y[-1] + 2.5))  /  (1 + Y[-1] * Y[-1] *Y[-2] * Y[-2]) + u[-1])

def TwoITwoO(Y,U):
    y1 = Y[0]
    y2 = Y[1]
    Y[0].append( (y1[-1])          / (1 + pow(y2[-1],2)) + U[0][-1] )
    Y[1].append( (y1[-1] * y2[-1]) / (1 + pow(y2[-1],2)) + U[1][-1] )
