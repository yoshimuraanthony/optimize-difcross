# Anthony Yoshimura
# 09/18/19

from numpy import array, floor
import matplotlib.pyplot as plt 
#from excite import getDifCrossDict, getGDict, getCrossDict, getTotCross
from excite import getGDict, getCrossDict, getTotCross
from excite import getProperties
from getKwargsDict import getKwargsDict
from time import time

def plotCrossVsEnergy(
        # inputs
        ediff_list = 'file',
        cross_list = 'file', 
        infile = 'cross.out',
        Eb = 8e4,

        # saving
        save = False,
        outfile = 'crossVsEnergy.png',
        **kwargs,
        ):
    """
    plots cross section in Å^2 for each band-to-band transition
    """
    if type(ediff_list) == str:
        ediff_list = []
        cross_list = []
        with open(infile) as f:
            f.readline()
            for line in f:
                ediff, cross = [float(val) for val in line.split()[:2]]
                ediff_list.append(ediff)
                cross_list.append(cross)

    fig, ax = plt.subplots()
    ax.plot(ediff_list, cross_list, 'o', markersize = 1.5, color = 'red')
#    , label = 'Eb = %s keV' %(Eb/1000))
    ax.set_xlabel('energy (eV)', fontsize = 14)
    ax.set_ylabel('cross section (Å$^2$)', fontsize = 14)
    ax.grid()
    ax.text(.98, .98, 'Eb = %s keV' %(Eb/1000), ha = 'right', va = 'top',
             transform = ax.transAxes, fontsize = 14)
    plt.tight_layout()

    if save:
        plt.savefig(outfile)
    plt.show()


def writeData(
        infile = 'ex.in',
        OUTCAR = 'OUTCAR',
        outfile = 'cross.out',
        progress = 'progress.out',
        ):
    """
    writes all output files
    """
    startTime = time()

    try:
        kwargs_dict = getKwargsDict(infile)
    except FileNotFoundError:
        kwargs_dict = {} 

    for key in kwargs_dict:
        val = kwargs_dict[key]
        print('%s = %s' %(key, val))

    #difCross_dict = getDifCrossDict(**kwargs_dict)
    k_dict, G_dict = getGDict(**kwargs_dict)

    cross_dict = getCrossDict(k_dict, G_dict, **kwargs_dict)
    ediff_dict = getEdiffDict(OUTCAR)

    with open(outfile, 'w') as f:
        f.write('energy\tcross section\ttransition\n')
        for i2 in ediff_dict:
            ediff_i2_dict = ediff_dict[i2]
            cross_i2_dict = cross_dict[i2]
            for i3 in ediff_i2_dict:
                ediff_i2i3_dict = ediff_i2_dict[i3]
                cross_i2i3_dict = cross_i2_dict[i3]
                for vb in ediff_i2i3_dict:
                    ediff_i2i3vb_dict = ediff_i2i3_dict[vb]
                    cross_i2i3vb_dict = cross_i2i3_dict[vb]
                    for cb in ediff_i2i3vb_dict:
                        ediff = ediff_i2i3vb_dict[cb]
                        cross = cross_i2i3vb_dict[cb]
                        f.write('%.5g\t%.5g\t%s %s %s %s\n'
                                %(ediff, cross, i2, i3, vb, cb))

    totCross = getTotCross(cross_dict)

    with open(progress, 'a') as f:
        f.write('\ntotal excitation cross section = %s\n' %totCross)
        # f.write('total writeData time = %s\n' %(time() - startTime))
        print('\ntotal excitation cross section = %s' %totCross)
        print('total writeData time = %s' %(time() - startTime))
    

def getEdiffDict(
        OUTCAR = 'OUTCAR',
        **kwargs,
        ):
    """
    returns energy differences for all possible excitations
    """
    energy_tab = getEnergyTab(OUTCAR)
    nbands, nelect, wt_list, area = getProperties(OUTCAR)
    occ = int(nelect/2)

    ediff_dict = {}
    for i2, energy2_list in enumerate(energy_tab):
        ediff_i2_dict = {}
        for i3, energy3_list in enumerate(energy_tab):
            ediff_i2i3_dict = {}
            for vb, energy2 in enumerate(energy2_list[:occ]):
                ediff_i2i3vb_dict = {}
                for cb, energy3 in enumerate(energy3_list[occ:]):
                    ediff_i2i3vb_dict[cb + occ] = energy3 - energy2
                ediff_i2i3_dict[vb] = ediff_i2i3vb_dict
            ediff_i2_dict[i3] = ediff_i2i3_dict
        ediff_dict[i2] = ediff_i2_dict

    return ediff_dict
    

def getEnergyTab(infile = 'OUTCAR'):
    """ returns table of energies at each k-point listed """
    energy_tab = []
    with open(infile) as f:
        for line in f:
            if 'NKPTS' in line:
                line_list = line.split()
                nkpts = int(line_list[3])
                nbands = int(line_list[-1])
                print('nkpts = %s, nbands = %s' %(nkpts, nbands))
    
            if 'k-point   ' in line:
                for i in range(nkpts):
                    f.readline()
                    energy_list = []
                    for j in range(nbands):
                        energy = float(f.readline().split()[1])
                        energy_list.append(energy)
                    f.readline()
                    f.readline()
                    energy_tab.append(energy_list)
                break

    return array(energy_tab)

#---------------------------- PHYSICAL CONSTANTS ------------------------------

m = 5.109989461e5  # mass of electron (eV)
M = 9.3827231e8  # mass of proton (eV)
c  = 299792458  # speed of light (m/s)
a0 = 5.29177e-11  # Bohr radius (m)
ev = 1.60217662e-19  # electronVolt (J)
e0 = 8.85418782e-12  # permittivity of free space (SI)
kb = 8.6173303e-5  # Boltzmann constant (eV/K)
bm = 9.37e-11  # minimum impact parameter in MoS2 (m)
alpha = 1/137.035999084  # fine structure constant (unitless)
hbar = 6.582119569e-16  # planck constant (eV s)

#--------------------------- CALLING FROM TERMINAL ----------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('INPUT', nargs='?', default='ex.in', help='keyword args input file')
    p.add_argument('--output', '-o', default='cross.out', help='output file')
    p.add_argument('--progress', '-P', default='progress.out', help='progress file')
    args = p.parse_args()

    writeData(
        infile=args.INPUT,
        outfile=args.output,
        progress=args.progress,
    )

if __name__ == '__main__':
    main()

#---------------------------------- SCRATCH -----------------------------------
#
# def getExCrossFromFile(
#         infile = 'crossAndEdiff.out',
#         OUTCAR = 'OUTCAR',
#         ):
#     """
#     returns sums over all valence to conduction band cross secitons
#     """
#     trans_dict = {}
#     with open(infile) as f:
#         f.readline()
#         for line in f:
#             line_list = line.split()
#             cross = float(line_list[1])
#             kpt = int(line_list[2])
#             band1 = int(line_list[4])
#             band2 = int(line_list[6])
#             trans_dict[(kpt, band1, band2)] = cross
#
#     nbands, nelect, wt_list = getNbandsNelectAndKptWts(OUTCAR)
#     occ = int(nelect/2)
#     vb_list = list(range(occ))
#     cb_list = list(range(occ, nbands))
#
#     totCross = 0
#     for i, wt in enumerate(wt_list):
#         for vb in vb_list:
#             for cb in cb_list:
#                 cross = trans_dict[(i, vb, cb)]
#                 totCross += cross
#
#     return totCross  # already multiplied by 4


