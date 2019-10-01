# Anthony Yoshimura
# 06/06/19

from numpy import pi, log, e, cos, sin, arccos, arcsin, sqrt, degrees, radians
from numpy import floor, ceil, sum
from numpy import array, transpose, linspace, zeros, arange
from periodic import table as p_dict
from numpy.linalg import norm
import matplotlib.pyplot as plt

r"""                                   
Moller Cross Section
                                                  p4 \  / p3
                                                      \/                       
       p4 \                / p3                       /\                       
           \              /                          /  \                      
            \            /                          /    \                     
             \  q --->  /                          /      \                    
             /\/\/\/\/\/\                         /        \                   
            /            \                       /\/\/\/\/\/\                  
           /              \                     /   q --->   \                 
       p1 /                \ p2                /              \    
                                           p1 /                \ p2

Want likelihood of transfering certain momentum p3 to valence electron given
beam energy Eb, all in eV.
"""

def getProbOfP3(p1_ar, p2_ar, p3_ar, p4_ar=None):
    """
    returns Moller differential cross section in LAB frame for
        electron-electron scattering into differential momentum element
        centered on p3
        * beam electron parralel to z-direction
        * delta function (which is required for cross section to have units of
          area) is implied by only selecting physically allowed p3 and p4 in
          excited.getDifCrossDict()
    pn_ar: initial material 4-momentum in eV (array of 4 floats)
    """
    if p4_ar is None:
        p4_ar = p1_ar + p2_ar - p3_ar

    E1, p1x, p1y, p1z = p1_ar
    E2, p2x, p2y, p2z = p2_ar
    E3, p3x, p3y, p3z = p3_ar
    E4, p4x, p4y, p4z = p4_ar

    p12 = E1*E2 - p1z*p2z
    p13 = E1*E3 - p1z*p3z
    p14 = E1*E4 - p1z*p4z
    
    t = (E1 - E3)**2 - p3x**2 - p3y**2 - (p1z - p3z)**2
    u = (E1 - E4)**2 - p4x**2 - p4y**2 - (p1z - p4z)**2
    
    first = 1/t**2*(p12**2 + p14**2 - 2*m**2*p13 + 2*m**4)
    second = 1/u**2*(p12**2 + p13**2 - 2*m**2*p14 + 2*m**4)
    third = 2/u/t*(p12**2 - 2*m**2*p12)

    p1tz = norm(p1_ar[1:] - p2_ar[1:])
    prefactor = alpha**2/abs(E1*p2z - E2*p1z)/abs(p1tz*m)

    return 4*prefactor*(first + second + third) # x2, P(theta) = P(pi-theta)


#--------------------------------- CM FRAME -----------------------------------

def getProbOfP3p(pp, p3pz):
#def getProbOfP3p(p3z, Eb):
    """
    returns Moller differential cross section in CM frame for electron-electron
        scattering into differential momentum volume element centered on pp3.
    p3z: final momentum in LAB frame in eV (array of 3 floats)
    Eb: beam energy in LAB frame in eV (float)
    """
#    pp, p3pz = getPpAndP3pz(p3z, Eb)

    prefactor = alpha**2/4/pp**4/(m**2 + pp**2)/(pp**2 - p3pz**2)**2
    first = pp**2*(3*pp**2 + p3pz**2)**2
    second = m**2*(m**2 + 4*pp**2)*(pp**2 + 3*p3pz**2)

    return 2*prefactor*(first + second) # doubled since P(theta) = P(pi-theta)


def getBetaAndGamma(Eb):
    """
    returns CM velocity in LAB frame and corresponding Lorentz factor
        * should only do this once. Eventually, all other functions should
            accept beta and gamma instead of Eb
    Eb: beam energy in LAB frame in eV (float)
    """
    v1 = (Eb*(Eb + 2*m))**.5/(Eb + m)
    beta = (1 - (1 - v1**2)**.5)/v1
    gamma= (1 - beta**2)**-.5

    return beta, gamma#, v1


def getPpAndP3pz(p3z, Eb):
    """
    returns magnitude and z-component of final momentum p3' in CM frame
    p3z: final momentum in LAB frame in eV (array of 3 floats)
    Eb: beam energy in LAB frame in eV (float)
    """
    beta, gamma = getBetaAndGamma(Eb)
    pp = gamma*m*beta

    return pp, p3z/gamma - beta*(pp**2 + m**2)**.5

#-------------------------------- TEST PLOTS ----------------------------------

# def plotOldVNew(low = 1e4, high = 3e5):
#     p3z = 100 # eV
#     p = linspace(low, high, 200)
#     Eb = (p**2 + m**2)**.5 - m
#     pp, p3pz = getPpAndP3pz(p3z, Eb)
#
#     new = getProbOfP3(p, [0,0,0],[0,0,p3z])
#     old = getProbOfP3p(pp, p3pz)
#
#     ax.set_xlabel('LAB momentum (eV)')
#     ax.set_ylabel('cross section (s$^{-2}$)')
#     ax.legend()
#     plt.tight_layout()
#     plt.show()


# def plotTheta(theta1 = 1e-6, theta2 = 1e-4):
#     theta = linspace(theta1, theta2 , 200)
#     pp = getPp(1e5)
#     p3pz = pp*cos(theta)
#     E = (pp**2 + m**2)**.5
#     v = pp/E
#     fig, ax = plt.subplots()
#
#     myTheta = pp**2*getThetaOfP3p(pp, theta)
#     myPp = pp**2*getProbOfP3p(pp, p3pz)
#     wikiTheta = getWikiProbOfP3p(pp, theta)
#     wikiPp= getWikiPpOfP3p(pp, p3pz)
#
#     ax.plot(theta, myTheta/wikiTheta, label = 'theta')
#     ax.plot(theta, myPp/wikiPp, label = 'pp')
#
#     ax.set_xlabel('scattering angle')
#     ax.set_ylabel('cross section (s$^{-2}$)')
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

def plotPp(pp1 = 1e4, pp2 = 3e5):
    theta = 1e-5
    pp = linspace(pp1, pp2 , 200)
    p3pz = pp*cos(theta)
    E = (pp**2 + m**2)**.5
    v = pp/E
    fig, ax = plt.subplots()

    myTheta = pp**2*getThetaOfP3p(pp, theta)
    myPp = pp**2*getProbOfP3p(pp, p3pz)
    wikiTheta = getWikiThetaOfP3p(pp, theta)
    wikiPp = getWikiPpOfP3p(pp, p3pz)

    ax.plot(pp, myTheta/wikiTheta, label = 'theta')
    ax.plot(pp, myPp/wikiPp, label = 'pp')

    ax.set_xlabel('CM momentum (eV)')
    ax.set_ylabel('cross section (s$^{-2}$)')
    ax.legend()
    plt.tight_layout()
    plt.show()

# def plotWiki():
#     theta = linspace(1e-7, 1e-6 ,200)
#     pp = getPp(1e5)
#     fig, ax = plt.subplots()
#     ax.plot(theta, getWikiProbOfP3p(pp, theta))
#     plt.show()

# def plotMine():
#     p = 8e4
#     p2 = [0,0,0]
#     p3 = [0,0,0]
#     fig, ax = plt.subplots()
#     ax.plot(theta, getProbOfP3(pp, p3pz))
#     plt.show()

# def oldPlotMine():
#     theta = linspace(pi - 1e-6, pi - 1e-7 ,200)
#     pp = getPp(1e5)
#     p3pz = pp*cos(theta)
#     fig, ax = plt.subplots()
#     ax.plot(theta, getProbOfP3p(pp, p3pz))
#     plt.show()

#---------------------------- PHYSICAL CONSTANTS ------------------------------

m = 5.109989461e5  # mass of electron (eV)
M = 9.3827231e8  # mass of proton (eV)
#m  = 9.10938356e-31  # mass of electron (kg)
#M = 1.6726219e-27  # mass of electron (kg)
c  = 299792458  # speed of light (m/s)
a0 = 5.29177e-11  # Bohr radius (m)
ev = 1.60217662e-19  # electronVolt (J)
e0 = 8.85418782e-12  # permittivity of free space (SI)
kb = 8.6173303e-5  # Boltzmann constant (eV/K)
bm = 9.37e-11  # minimum impact parameter in MoS2 (m)
alpha = 1/137.035999084  # fine structure constant (unitless)
hbar = 6.582119569e-16  # planck constant (eV s)
#hbar = 1  # planck constant (unitless)

#------------------------- SPECIES-SPECIFIC CONSTANTS -------------------------

#Zb = 5
#Zn = 7
ZS  = 16
ZSe = 34
ZTe = 52
Mb = 10.811 * M 
Mn = 14.007 * M 
MS = 32.06 * M 
MSe = 78.97 * M 
MTe = 127.6 * M 
#Tdb = 19.36 * e
#Tdn = 23.06 * e
TdS = 7.08
TdSe = 6.58
TdTe = 6.16

#----------------------------- UNIT CONVERSIONS -------------------------------

scale1 = 1e28 * c**2                  # s^2 to 100 fm^2 (barn)
const = pi*hbar**2*alpha**2 * scale1  # prefactor to cross sections

#------------------------------ TEST FUNCTIONS --------------------------------

def getThetaOfP3p(pp, theta):
    """
    returns Moller differential cross section in CM frame for electron-electron
        scattering into differential momentum volume element centered on pp3.
    pp: electron beam momentum in CM frame in eV (float)
    p3pz: z-component momentum of outgoing electron in CM frame in eV (float)
        * assume incident electron has momentum in z direction
    """
    prefactor = alpha**2/4/pp**6/(m**2 + pp**2)
    first = pp**4
    second = (3*m**4 + 12*m**2*pp**2 + 8*pp**4)/sin(theta)**2
    third = 4*(m**2 + 2*pp**2)**2/sin(theta)**4

    return prefactor*(first - second + third)

def getWikiThetaOfP3p(pp, theta):
    """
    https://en.wikipedia.org/wiki/Møller_scattering
    """
    EcmSq = 4*(pp**2 + m**2)
    prefactor = alpha**2/EcmSq/pp**4/sin(theta)**4
    first = 4*(m**2 + 2*pp**2)**2
    second = (4*pp**4 - 3*(m**2 + 2*pp**2)**2)*sin(theta)**2
    third = pp**4*sin(theta)**4

    return prefactor*(first + second + third)

def getWikiPpOfP3p(pp, p3pz):
    """
    https://en.wikipedia.org/wiki/Møller_scattering
    """
    EcmSq = 4*(pp**2 + m**2)
    prefactor = alpha**2/EcmSq/pp**4/(1-(p3pz/pp)**2)**2
    first = 4*(m**2 + 2*pp**2)**2
    second = (4*pp**4 - 3*(m**2 + 2*pp**2)**2)*(1 - (p3pz/pp)**2)
    third = pp**4*(1 - (p3pz/pp)**2)**2

    return prefactor*(first + second + third)

def getWikiCos(cos, p = 4e12):
    """
    https://en.wikipedia.org/wiki/Møller_scattering
    """
    sin2 = 1 - cos**2
    sin4 = sin2**2
    Ecm2 = 2*(m**2 + p**2)

    pre = alpha**2/Ecm2/p**4/sin4
    first = 4*(m**2 + 2*p**2)**2
    second = (4*p**4 - 3*(m**2 + 2*p**2)**2)*sin2
    third = p**4*sin4

    return 4*pre*(first + second + third)*invEVSqtoPb

def plotWikiCos(pp = 4e12):
    cos_ar = linspace(-.96,.96,500)
    sig_ar = getWikiCos(cos_ar)
    fig, ax = plt.subplots()
    ax.plot(cos_ar, sig_ar)
    ax.set_xlabel(r'cos($\theta$)', fontsize = 14)
    ax.set_ylabel('cross section (pb)', fontsize = 14)
    ax.set_yscale("log", nonposy='clip')
#    ax.set_ylim(0, 3.5)
    ax.set_xlim(-1, 1)
    ax.grid()
    plt.show()
    
#----------------------------- UNIT CONVERSIONS -------------------------------

scale1 = 1e28 * c**2                  # s^2 to 100 fm^2 (barn)
const = pi*hbar**2 * alpha**2 * scale1  # prefactor to cross sections
invÅtoEV = 1e10 * c * hbar
invÅtomeV = 1e10 * c * hbar * 1000
invEVSqtoÅSq = hbar**2 * c**2 * 1e20
invEVSqtoBarn = hbar**2 * c**2 * 1e28
invEVSqtoPb = hbar**2 * c**2 * 1e40

#---------------------------------- SCRATCH -----------------------------------

#    revTheta = linspace(pi - theta1, pi - theta2, 200)
#    wikiDataRev = getWikiProbOfP3p(pp, revTheta)
#    ax.plot(theta, wikiDataRev, label = 'wiki')

#def getP3pz(p3, Eb):
#    """
#    returns z-component of final momentum p3' in CM frame given the final
#        momentum p3 in LAB frame
#    p3: final momentum in LAB frame in eV (array of 3 floats)
#    Eb: beam energy in LAB frame in eV (float)
#    """
#    p3x, p3y, p3z = p3
#    E3 = sqrt(dot(p3, p3) + me**2)

#    beta, gammaSquared = getBetaAndGammaSquared(Eb)
#
#    return gammaSquared*(p3z - beta*E3)**2

# Er = 2.17896e-18  # J
# ke = 2.30707751e-28  #r^2/J
#def getProbOfP3p(pp, p3pz):
#    """
#    returns Moller differential cross section in CM frame for electron-electron
#        scattering into differential momentum volume element centered on pp3.
#    pp: electron beam momentum in CM frame in eV (float)
#    p3pz: z-component momentum of outgoing electron in CM frame in eV (float)
#        * assume incident electron has momentum in z direction
#    """
#    # UNDER CONSTRUCTION:
#        # currently, Eb is the CM energy of the beam electron
#        # beam energy should be the LAB kinetic energy of the beam,
#            # i.e., without mass energy
#        # then add mass via Eb = Eb + me, or something like that
#    
#    # dot products of 4-momenta: pnm = me**2 + p**2 - pn \dot pm
#    p12 = m**2 + 2*pp**2
#    p13 = m**2 + pp*(p3pz - pp)
#    p14 = m**2 + pp*(p3pz + pp)
#    E1p = pp**2 + m**2
#
#    # differences of 4-momenta:
#    s = 4*E1p**2
#    t = 2*pp*(p3pz - pp)
#    u = -2*pp*(p3pz + pp)
#
#    # moller cross section
#    dsigma = alpha**2/16/pp**2/E1p**2*(\
#                 2/t**2*(p12**2 + p14**2 - 2*m*p13 + 2*m**4)\
#               + 2/u**2*(p12**2 + p13**2 - 2*m*p14 + 2*m**4)\
#               + 4/t/u*p12*(p12 - 2*m**2))
#                     
#    return dsigma
#
#def getDietProbOfP3p(E, theta):
#    """
#    https://www.youtube.com/watch?v=ROK9nXe9ENg&t=122s
#    """
#    first = alpha**2/4/E**2
#    second = (2*E**2 - m**2)**2/(E**2 - m**2)**2
#    third = 2/sin(theta)**2 - (E**2 - m**2)/(2*E**2 - m**2)
#
#    return first * second * third**2
#
#def getEricProbOfP3p(pp, theta):
#    """
#    ecotner.bol.ucla.edu/Classes/PHYS230/Phys%20230B%20(QFT)%20Homework%204.pdf
#    """
#    ESq= pp**2 + m**2
#    s = 4*ESq
#    t = 2*m**2 - 2*(ESq - pp**2*cos(theta))
#    u = 2*m**2 - 2*(ESq + pp**2*cos(theta))
#    prefactor = alpha**2/s
#    first = (s - u)**2/t**2
#    second = (s - t)**2/u**2
#    third = 1/u/t*(s**2 + 5*(u**2 + t**2))
#
#    return prefactor*(12 + first + second - third)
#
#def getPrasProbOfP3p(E, theta):
#    """
#    https://www.youtube.com/watch?v=dkrmJO3GNPw&t=2560s
#    """
#    prefactor = alpha**2*(2*E**2 - m**2)/4/E**2/(E**2 - m**2)**2
#    first = 4/sin(theta)**4
#    second = 3/sin(theta)**2
#    third = (E**2 - m**2)**2 / (2*E**2 - m**2)**2 * (1 + 4/sin(theta)**2)
#  
#    return prefactor*(first - second + third)
#
#def getRoqProbOfP3p(vp, theta):
#    """
#    10.1007/bf00377049
#    """
#    v = 2*vp/(1 + vp**2)
#    gamma = (1 - v**2)**(-1/2)
#    prefactor = 2*(gamma + 1)/m**2/gamma**2/v**4
#    first = 4/sin(theta)**4
#    second = 3/sin(theta)**2
#    third = (gamma - 1)**2/4/gamma**2*(1 + 4/sin(theta)**2)
#
#    return prefactor*(first - second + third)
#
#def getChamProbOfP3p(vp, theta):
#    """
#    10.1007/bf00377049
#    """
#    v = 2*vp/(1 + vp**2)
#    gamma = (1 - v**2)**(-1/2)
#    x = cos(theta)
#    prefactor = 4*pi/(m*v**2)**2*(gamma + 1)/gamma**2
#    first = 4/(1 - x**2)**2
#    second = 3/(1 - x**2)
#    third = (gamma - 1)**2/4/gamma**2*(1 + 4/(1 - x**2))
#
#    return prefactor*(first - second + third)
#
#    ax.plot(theta, prasData, label = 'pras')
#    ax.plot(theta, ericData, label = 'eric')
#    ax.plot(theta, dietData, label = 'diet')
#    ax.plot(theta, roqData, label = 'roq')
#    ax.plot(theta, chamData, label = 'cham')

#def getPp(Eb):
#    """
#    returns the magnitude of the beam electron's 3-momentum in CM frame
#    Eb: beam energy in LAB frame in eV (float)
#    """
#    beta, gamma = getBetaAndGamma(Eb)
#
#    return gamma*m*beta
#
#
#    pp: electron beam momentum in CM frame in eV (float)
#    p3pz: z-component momentum of outgoing electron in CM frame in eV (float)
#        * assume incident electron has momentum in z direction
#    prefactor = alpha**2/abs(E1*p2z - E2*p1z)/abs(E3*p4z - E4*p3z)*2
#    Eb = E1 - m
#    beta, gamma = getBetaAndGamma(Eb)
#    p1pz = gamma*m*beta
#    E1p = gamma*m
#    prefactor = alpha**2/p1pz**2/4/E1p**2
