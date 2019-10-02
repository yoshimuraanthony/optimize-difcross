# Excite.py
# Anthony Yoshimura
# 09/21/19

from numpy import pi, log, e, cos, sin, arccos, arcsin, sqrt, degrees, radians
from numpy import floor, ceil, cross
from numpy import array, transpose, linspace, zeros, arange, dot, insert
from periodic import table as p_dict
from numpy.linalg import norm

from moller import getProbOfP3, getBetaAndGamma

# catch warnings (e.g. RuntimeWarning) as an exception
import warnings
warnings.filterwarnings('error') 

from time import time

r"""
Excitation                             |  \                 /             
                                       |   \_             _/    
        p4 \                / p3 ------|-> k'\__       __/                     
            \              /          e|        \_____/                        
             \            /           n|                    
              \  q --->  /            e|                    
              /\/\/\/\/\/\            r|        _____                          
             /            \           g|     __/     \__                       
            /              \          y|   _/           \_                     
        p1 /                \ p2 <-----|-k/               \                    
                                       |_/_________________\_____   
                                               momentum
                                                                 
For excitation, the p3 state can be approximated as a nonrelativistic plane
wave. Most functions should accept p3 and Eb as inputs

Should use px and py LAB frame values, since they are much greater than pz
also, cos(theta) ~ 1 - theta**2/2, while sin(theta)

Plane wave coefficients are generated from WaveTrans.f90 on vasp calculations 

p(E=600 eV) = 24770 eV (24763 eV nonrelativistially)
p(E=80 keV) = 296917 eV (285937 eV nonrelativistially)

label 3D reciprocal lattice coordinates as k, and 4-momenta as p

does not work if spin breaks degeneracy

assume first k-point is gamma

all valence bands must be filled in ground state
"""

def getTotCross(
        cross_dict
        ):
    """
    returns total cross section for all valence to conduction band excitations
    """
    totCross = 0
    for i2 in cross_dict:
        for i3 in cross_dict[i2]:
            for vb in cross_dict[i2][i3]:
                for cb in cross_dict[i2][i3][vb]:
                    totCross += cross_dict[i2][i3][vb][cb]

    return totCross


def getCrossDict(
        difCross_dict,
        G_dict,
        vb_list = 'all',
        cb_list = 'all',
        Eb = 8e4,  # eV
        GCOEFF = 'GCOEFF.txt',
        OUTCAR = 'OUTCAR',
        progress = 'progress.out',
        ):
    """
    returns dictionary of excitation cross sections for each given valence and
        conduction band pair for each k-point pair
        * if probabilities are much less than one, we can add them.
    Eb: beam energy in LAB frame in eV (pos float)
    GCOEFF: GCOEFF file containing plane wave coefficients (str)
    OUTCAR: OUTCAR file from vasp run (str)
    """
    # UNDER CONSTRUCTION: p3 only overlaps with states on the same k-point
    startTime = ()
    nbands, nelect, wt_list, area = getProperties(OUTCAR)
    normalization = 1/sum(wt_list)**2 
    occ = int(nelect/2)
    if vb_list == 'all':
        vb_list = list(range(occ))
    if cb_list == 'all':
        cb_list = list(range(occ, nbands))

    with open(progress, 'a') as f:
        f.write('summing cross sections for all ground state excitations\n')
        print('summing cross sections for all ground state excitations')
    
        cross_dict = {}
        startTime = time()
        for i2, wt2 in enumerate(wt_list):
            loopTime = time()
            f.write('\tsumming from kpt %s\n' %i2)
            print('\tsumming from kpt %s' %i2)
    
            cross_dict[i2] = {}
    
            for i3, wt3 in enumerate(wt_list):
                cross_dict[i2][i3] = {}
        
                for vb in vb_list:
                    cross_dict[i2][i3][vb] = {}

                    for cb in cb_list:
                        cross_i2i3vbcb = sum(
                            G_dict[i2][vb][k2]
                            * difCross_dict[i2][i3][k2][k3][0]
                            * G_dict[i3][cb][k3]
                            for k2 in difCross_dict[i2][i3]
                            for k3 in difCross_dict[i2][i3][k2]
                        ) # eV^{-2}

                        # cross section in unit cell area, x4 for spins
                        cross_dict[i2][i3][vb][cb] = (
                             cross_i2i3vbcb * 4
                             * invEVSqtoÅSq * normalization
                             * wt2 * wt3 / area
                        )
            # f.write('\tloop time = %s\n' %(time() - loopTime))
            print('\tloop time = %s' %(time() - loopTime))
    
        # f.write('total getCrossDict time = %s\n\n' %(time() - startTime))
        print('total getCrossDict time = %s\n' %(time() - startTime))

    return cross_dict

def getGDict(
        infile = 'GCOEFF.txt',
        progress = 'progress.out',
        **kwargs,
        ):
    """
    returns dictioanry containing modulus squared of plane wave coefficients
        * only works for gamma point calculations
        G_dict[(kpt, band)] = wave_dict
            wace_dict[(kx, ky, kz)] = |GCoeff|^2
    infile: GCOEFF.txt file containing plane wave coefficients (str)
    outfile: file to which progress is written (str)
    writeMode: 'w' = overwrite, 'a' = append ('w' or 'a')
    """
    startTime = time()
    with open(infile) as f:
        with open(progress, 'a') as out:
            f.readline()
            nkpts = int(f.readline())
            nbands = int(f.readline())
            
            for n in range(6):
                f.readline()
    
            kptBand_dict = {}
            for i in range(nkpts):
                out.write('obtaining coefficients for kpt %s of %s\n'
                          %(i + 1, nkpts))
                print('obtaining coefficients for kpt %s of %s'
                          %(i + 1, nkpts))
                kptBandi_dict = {}
                f.readline()
                for j in range(nbands):
                    nwaves = int(f.readline().split()[1])
                    energy = float(f.readline().split()[1])
                    wave_dict = {}
                    for k in range(nwaves):
                        G_list = f.readline().split()
                        kx, ky, kz = [int(val) for val in G_list[:3]]
                        reG = float(G_list[4])
                        imG = float(G_list[6])
                        GSq = reG**2 + imG**2
                        wave_dict[(kx, ky, kz)] = GSq
    
                        kptBandi_dict[j] = wave_dict  # indices start from 0
                kptBand_dict[i] = kptBandi_dict 
    
            # out.write('total getGDict time = %s\n\n' %(time() - startTime))
            print('total getGDict time = %s\n' %(time() - startTime))
    
        return kptBand_dict


def getDifCrossDict(
        Eb = 8e4,
        infile = 'GCOEFF.txt',
        printInfo = True,
        outfile = 'difCross.out',
        progress = 'progress.out',
        k2x = 0,  # testExcite.py
        k2z = 0,
        **kwargs,
        ):
    """
    returns dictionary of differential cross sections for all plane wave pairs
        in WAVECAR that satisfy conservation of momentum
        * p2 can be any wave in WAVECAR
        * one p3 per (kx, ky) given in WAVECAR
        * store all p3 - p2 that are physically allowed
        * assumes lattice vector a3 is perpendicular to a1 and a2
    Eb: beam energy in LAB frame in eV (pos float)
    infile: GCOEFF.txt file containing plane wave coefficients (str)
    """
    startTime = time()

    # beam momentum
    E1 = Eb + m
    gamma = E1/m
    p1 = (E1**2 - m**2)**.5
    p1_ar = array([E1, 0, 0, p1])

    p2_dict, p3_dict = readWavecar(infile)

    # check point
    with open(outfile, 'w') as f:
        f.write('Eb = %s\n\n' %Eb)
        print('Eb = %s\n' %Eb)
        for i in p3_dict:
            p3_i_dict = p3_dict[i]
            nwaves = sum([len(p3_i_dict[pair][0]) for pair in p3_i_dict]) 
            f.write('obtained %s waves at k-point %s\n' %(nwaves, i))
            f.write('number of (kx, ky) pairs in p3_i_dict: %s\n'
                    %len(p3_i_dict))
            print('obtained %s waves at k-point %s' %(nwaves, i))
            print('number of (kx, ky) pairs in p3_i_dict: %s'
                    %len(p3_i_dict))

        crossStartTime = time()
        readTime = crossStartTime - startTime
        # f.write('read time = %s\n\n' %readTime)
        f.write('selecting all physically allowed p3 for each p2\n')
        print('read time = %s\n' %readTime)
        f.write('\ncalculating differential cross sections\n')
        print('calculating differential cross sections')

    # track number of times each p3z and k3z are scattered into.
    # find dif cross section for each physically allowed momentum transfer
    difCross_dict = computeDifCrossDict(gamma, outfile, p1, p1_ar,
                                        p2_dict, p3_dict, progress)

    # check point
    with open(outfile, 'a') as f:
        endTime = time()
        crossTime = endTime - crossStartTime

        # f.write('transition calculations time = %s\n' %transTime)
        # f.write('\nread time = %s\n' %readTime)
        # f.write('transition calculations time = %s\n' %transTime)
        # f.write('cross calculation time = %s\n' %crossTime)
        # f.write('total getDifCross time = %s\n\n' %(endTime - startTime))
        print('cross calculation time = %s' %crossTime)
        print('\nread time = %s' %readTime)
        print('cross calculation time = %s' %crossTime)
        print('total getDifCross time = %s\n' %(endTime - startTime))

    return difCross_dict


def readWavecar(infile):
    # for each k3x, k3y pair, store one dictionary for each k3z
    #     need to prepare dictionary structure beforehand since k's are read
    #     off from one long unordered list
    p2_dict = {} # p2_dict[nkpts][(kx, ky, kz)][4]
    p3_dict = {} # p3_dict[nkpts][(kx, ky)] = ({k3z: [4]}, p3x, p3y)
    for i in range(200):
        p3_dict[i] = {}
        for j in range(-60, 61):  # 20 Å with 600 eV maxed at 33 waves
            for k in range(-60, 61):  # j and k are recip. lat. coords
                p3_dict[i][(j, k)] = [{}]  # [{k3z: p3_ar}, p3x, p3y]
                # so that p3x, p3y are stored once

    # record all momenta listed in WAVECAR
    print('storing all plane waves')
    with open(infile) as f:

        spin = int(f.readline().strip())
        nkpts = int(f.readline().strip())

        for _ in range(4):
            f.readline()

        rCell = []
        for _ in range(3):
            rCell.append([float(val) for val in f.readline().split()])
        rCell = array(rCell) * invÅtoEV  # multiply here instead of in loop
        bz = norm(rCell[2])
        f.readline()

        crystalP_ar = zeros(3)  # assume first k-point is gamma
        for i in range(nkpts):
            p2_i_dict = {}
            nwaves = int(f.readline().split()[-1])
            print('\tstoring %s waves at k-point %s: %s'
                  % (nwaves, i, crystalP_ar / invÅtoEV))
            f.readline()

            # store each p3 in each k3z_dict
            for j in range(nwaves):
                # MUST ACCTOUNT for crystal momentum!
                k_ar = array([int(val) for val in f.readline().split()[:3]])
                p_ar = dot(k_ar, rCell) + crystalP_ar
                px, py, pz = p_ar
                E = (sum([comp ** 2 for comp in p_ar]) + m ** 2) ** .5
                p_ar = insert(p_ar, 0, E)  # make p_ar a 4-vector

                # store material momentum
                kx, ky, kz = k_ar
                p2_i_dict[(kx, ky, kz)] = p_ar
                k3z_list = p3_dict[i][(kx, ky)]
                k3z_list[0][kz] = (E, pz)
                if len(k3z_list) == 1:
                    p3_dict[i][(kx, ky)] += [px, py]

            for line in f:
                if '.' in line[:5]:
                    kpt = array([float(val) for val in line.split()])
                    crystalP_ar = dot(kpt, rCell)
                    break

            p2_dict[i] = p2_i_dict

    # Delete empty dictionaries
    for i in list(p3_dict):
        for k in list(p3_dict[i]):
            if len(p3_dict[i][k]) == 1:
                del p3_dict[i][k]
        if i not in range(nkpts):
            del p3_dict[i]

    return p2_dict, p3_dict


def computeDifCrossDict(gamma, outfile, p1, p1_ar, p2_dict, p3_dict,
                        progress):
    bestK3z_list = []
    bestP3z_list = []  # count number of p3z's for debugging
    # get all physically allowed transitions between pairs of plane waves
    with open(progress, 'w') as f:
        # trans_dict[kpt2][kpt3][(k2x, k2y, k2z)][(k3x, k3y, k3z)] = ([4], [4], [4])
        difCross_dict = {}  # dict with dicts for each k2 containing allowed k3s

        for i2 in traceLoopTime(p2_dict, 'selecting transitions and calculating differential cross section for k-point %s', f):
            difCross_dict[i2] = {}

            for i3 in p3_dict:
                difCross_dict[i2][i3] = getDifCrossDictAtKpoints(gamma, p1,
                                                                 p1_ar,
                                                                 p2_dict[i2],
                                                                 p3_dict[i3],
                                                                 bestK3z_list,
                                                                 bestP3z_list)

    logBestK3zCounts(bestK3z_list, bestP3z_list, outfile)
    return difCross_dict


def getDifCrossDictAtKpoints(gamma, p1, p1_ar, p2_i2_dict, p3_i3_dict,
                             bestK3z_list, bestP3z_list):
    difCross_i2i3_dict = {}
    for k2, p2_ar in p2_i2_dict.items():
        difCross_i2i3_dict[k2] = {}
        (k2x, k2y, k2z) = k2
        E2, p2x, p2y, p2z = p2_ar

        for k3x, k3y in p3_i3_dict:

            # ignore zero-scattering scenario
            if (k2x, k2y) == (k3x, k3y):
                continue

            # calculate trueP3z that conserves of momentum
            k3z_dict, p3x, p3y = p3_i3_dict[(k3x, k3y)]
            gammap = gamma + 1
            trueP3z = (p1 + p2z - (
                (p1 + p2z) ** 2
                - 2 * gammap * (
                    p1 * p2z - p2x * p3x - p2y * p3y
                    + 0.5 * gammap * (p3x ** 2 + p3y ** 2)
                )
            ) ** .5) / gammap

            # find closest k3z for given k3x, k3y pair
            minP3zDiff = invÅtomeV
            for k3z in k3z_dict:
                _, p3z = k3z_dict[k3z]
                p3zDiff = abs(p3z - trueP3z)

                if p3zDiff < minP3zDiff:
                    minP3zDiff = p3zDiff
                    bestK3z = k3z
                    bestP3z = p3z

            # store closest k3z in k3_dict
            bestE3, _ = k3z_dict[bestK3z]
            bestK3_key = (k3x, k3y, bestK3z)
            bestP3_ar = array([bestE3, p3x, p3y, bestP3z])

            difCross = getProbOfP3(p1_ar, p2_ar, bestP3_ar)
            difCross_i2i3_dict[k2][bestK3_key] = (difCross, bestP3_ar)
            bestK3z_list.append(bestK3z)  # for debugging
            bestP3z_list.append(bestP3_ar[-1])
    return difCross_i2i3_dict


def traceLoopTime(iterable, msg, outfile=None):
    for x in iterable:
        loopTime = time()
        print('\t' + msg % x)
        if outfile is not None:
            print('\t' + msg % x, file=outfile)

        yield x
        print('\tloop time = %s' % (time() - loopTime))


def logBestK3zCounts(bestK3z_list, bestP3z_list, outfile):
    # count how many times each p3z was scattered into
    with open(outfile, 'a') as f:
        bestK3z_dict = {}
        for bestK3z, bestP3z in zip(bestK3z_list, bestP3z_list):
            if (bestK3z, bestP3z) in bestK3z_dict:
                bestK3z_dict[(bestK3z, bestP3z)] += 1
            else:
                bestK3z_dict[(bestK3z, bestP3z)] = 0
        possible_list = [(k3z, p3z, bestK3z_dict[(k3z, p3z)])
                         for k3z, p3z in bestK3z_dict]
        possible_list.sort()
        f.write("all bestK3z's:\n")
        f.write('\tk3z\tp3z\t\tcount\n')
        for k3z, p3z, count in possible_list:
            f.write('\t%s,\t%.4g eV\t%s\n' % (k3z, p3z, count))


def getProperties(infile = 'OUTCAR'):
    """
    returns number of bands, number of electrons and k-point weights from
        VASP calculation
    infile: OUTCAR file (str)
    """
    with open(infile) as f:
        for line in f:

            if 'Following reciprocal' in line:
                f.readline()
                wt_list = []
                for line in f:
                    if len(line) > 10:
                        wt = float(line.split()[-1])
                        wt_list.append(wt)
                    else:
                        break

            if 'NBANDS' in line:
                nbands = int(line.split()[-1])

            if 'NELECT' in line:
                nelect = int(round(float((line.split()[2]))))

            if 'direct lattice vectors' in line:
                vec1 = [float(val[:-1]) for val in f.readline().split()[:3]]
                vec2 = [float(val[:-1]) for val in f.readline().split()[:3]]
                area = abs(cross(vec1, vec2)[-1])  # Å^2
                break

    return nbands, nelect, wt_list, area

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
invÅtoEV = 1e10*c*hbar
invÅtomeV = 1e10*c*hbar*1000
invEVSqtoÅSq = hbar**2*c**2*1e20

#--------------------------------- SCRATCH ------------------------------------

#            if abs((p3p - pp)/pp) < 3e-3 and abs(p3z) > 1e-5:
#                difCross_dict[wave_key] = getProbOfP3p(pp, p3pz)

#        numpasses = 0
#        numfails = 0
#                try:
#                    numpasses +=1
#                except UnboundLocalError:
#                    numfails += 1
#                    print('E4_1 = %s\tE4_2 = %s' %(E4_1, E4_2))
#                    break
#        print('number of fails at (%s, %s) = %s' %(k2x, k2y, numfails))
#        print('number of passes at (%s, %s) = %s' %(k2x, k2y, numpasses))
            
    # find best kz for each (kx, ky)
#    for kx, ky in fullWave_dict:
#        wave_dict = fullWave_dict[(kx, ky)]
#        minDif = pp
#        for wave_key in wave_dict:
#            p3p_ar = wave_dict[wave_key]
#            p3p = norm(p3p_ar)
#            dif = abs(p3p - pp)
#            if dif < minDif:
#                minDif = dif
#                bestWave_key = wave_key
#                bestP3p_ar = p3p_ar
#
#        p3pz = bestP3p_ar[2]
#        difCross_dit[bestWave_key] = getProbOfP3p(pp, p3pz) 
#        
#    return difCross_dict

# testing getDifCrossDict with test.py
#------------------ speed up for test.py -----------------------
# def getDifCrossDict(Eb = 8e4, infile = 'GCOEFF.txt', k2z = 0, k2x = 0, printInfo = True):
#    print('considering %s (kx, ky) coordinates' %(len(kxky_dict)))

#    for i in range(1):
#        for j in range(1):
#            k2y = 0
#------------------ speed up for test.py -----------------------

#            k3xk3y_list = [(0, n) for n in range(29)]
#            for k3x, k3y in [(0,0),(0,1),(0,2),(0,3)]:
#            for k3x, k3y in k3xk3y_list:
#    for k2x, k2y in kxky_dict:
#        n += 1
#        print('selecting transitions from (%s, %s).\t%s of %s'
#              %(k2x, k2y, n, numkxky))
#        k2z_dict = kxky_dict[(k2x, k2y)]
#        for k2z in k2z_dict:
#
#            # each k2 has a set of physically allowed k3's
#            k2z_dict = kxky_dict[(k2x, k2y)]
#            p2_ar = k2z_dict[k2z]
#            E2, p2x, p2y, p2z = p2_ar
#        
#            # find physically allowed k3 for k2
#            k3_dict = {}
#            bestK3z_list = []
#            bestP3z_list = []
#            p3z = (p1 - (p1**2 - 2*(gamma + 1)*p3x**2)**.5)/2
            # ignore k3z that severly break conservation of momentum
#            maxE2 = encut + m
#            maxP2 = (maxE2**2 - m**2)**.5
#            p3x = (p_ar[0]**2 + p_ar[1]**2)**.5
#            p3z = (p1 + p2z - ((p1 - p2z)**2
#                               - 4*(p3x*(p2x + p3x) + p3y*(p2y + p3y))**.5)/2
#            theta = arctan(maxP2/p1)
#
#            minP3 = p3z*cos(theta) - p3x*sin(theta)
#            maxP3 = p3z*cos(theta) + p3x*sin(theta)
#            minK3 = minP3/bz/invÅ
#            maxK3 = maxP3/bz/invÅ
#
#            if kz >= minKz - 1 and kz <= maxKz + 1:
#                k3xk3y_dict[(kx, ky)][kz] = p_ar
#
#            print('selecting transitions from k2 = (%s, %s, %s)\t%s of %s'
#                  %(k2x, k2y, k2z, n, numk2))
#        beta2 = p2z/E2
#        gamma2 = (1 - beta2**2)**-.5
#        scale = gamma(p1 - beta*E1)/p1
#            minEDiff = E1
#                statP3_ar = k3z_dict[statK3z]
#                statE3, p3x, p3y, statP3z = statP3_ar
#                scaledP3z = statP3z*scale
#                p3z = gamma2*(Stat3z - beta*statE3)
#                E3 = gamma2*(Stat3z - beta*statE3)
#                k3z = p3z/bz/invÅtoEV
        
#                p4_ar = p1_ar + p2_ar - p3_ar
        
#                # check how well p4 satisfies conservation of 4-momentum
#                E4_1, p4x, p4y, p4z = p4_ar
#                E4_2 = (p4x**2 + p4y**2 + p4z**2 + m**2)**.5
#        
#                EDiff = abs(E4_1 - E4_2)
#            for statK3z in k3z_dict:
#                if EDiff < minEDiff:
#                    minEDiff = EDiff
#                    bestK3z = k3z
#                    bestK3_key = (k3x, k3y, k3z)
#                    bestP3_ar = p3_ar
#                    bestP4_ar = p4_ar
#    encut: maximum plane wave energy in eV considered (pos float)

#            print('p1_ar = %s\np2_ar = %s\np3_ar = %s\np4_ar = %s\n' %(p1_ar, p2_ar, p3_ar, p4_ar))
#            trueP3z = (p1 + p2z - ((p1 - p2z)**2
#                               - 4*(p3x*(p2x + p3x) + p3y*(p2y + p3y)))**.5)/2
#------------------ old ----------------------------
#            minEDiff = E1
#            for k3z in k3z_dict:
#                p3_ar = k2xk2yk2z_dict[(k3x,k3y,k3z)]
#                p4_ar = p1_ar + p2_ar - p3_ar
#                # check how well p4 satisfies conservation of 4-momentum
#                E4_1, p4x, p4y, p4z = p4_ar
#                E4_2 = (p4x**2 + p4y**2 + p4z**2 + m**2)**.5
#        
#                EDiff = abs(E4_1 - E4_2)
#                if EDiff < minEDiff:
#                    minEDiff = EDiff
#                    bestK3z = k3z
#                    bestK3_key = (k3x, k3y, k3z)
#                    bestP3_ar = p3_ar
#                    bestP4_ar = p4_ar
#            # store closest k3z in k3_dict
#            k3_dict[bestK3_key] = (p2_ar, bestP3_ar, bestP4_ar)
#            bestK3z_list.append(bestK3z) # for debugging
#            bestP3z_list.append(bestP3_ar[-1])

#----------------------------------------------
    
#------------------------------------
#    for i in range(1):
#        for j in range(1):
#            k2y = 0
#-------------------------------------
#        f.write('number of waves in k3xk3y_dict: %s\n' %totNumWaves)
#------------------------------------
#            for j in range(1):
#                k2y = 0
#-------------------------------------
#        numk2 = len(kpt2_dict[i2])
#            n = 0
#                n += 1
#                if n%1000 == 0:
#                    f.write('\tselecting transition %s of %s\n' %(n, numk2))
#                    print('\tselecting transition %s of %s' %(n, numk2))
#-------------------------------------
#            for j in range(1):
#                k2y = 0
#-------------------------------------
