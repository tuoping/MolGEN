import os
import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations,permutations
from copy import deepcopy
import random

# Generates the adjacency matrix based on UFF bond radii
def Table_generator(Elements,Geometry):
    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # NOTE: Units of angstroms 
    # NOTE: These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
    # Modified values: Si (ORIGINAL: 1.117), Al (ORIGINAL: 1.244), H (ORIGINAL: 0.354)
    Radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    Max_Bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':1,    'Ar':1,\
                   'K':None, 'Ca':None, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':None, 'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
                     
    # Scale factor is used for determining the bonding threshold. 1.2 is a heuristic that give some lattitude in defining bonds since the UFF radii correspond to equilibrium lengths. 
    scale_factor = 1.2

    # Generate distance matrix holding atom-atom separations (only save upper right)
    Dist_Mat = np.triu(cdist(Geometry,Geometry))
    
    # Find plausible connections
    x_ind,y_ind = np.where( (Dist_Mat > 0.0) & (Dist_Mat < max([ Radii[i]**2.0 for i in Radii.keys() ])) )

    # Initialize Adjacency Matrix
    Adj_mat = np.zeros([len(Geometry),len(Geometry)])

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        
        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*scale_factor:            
            Adj_mat[i,y_ind[count]]=1
    
    # Hermitize Adj_mat
    Adj_mat=Adj_mat + Adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = { i:0 for i in Radii.keys() }
    conditions = { "H":1, "C":4, "F":1, "Cl":1, "Br":1, "I":1, "O":2, "N":4, "B":4 }
    for count_i,i in enumerate(Adj_mat):

        if Max_Bonds[Elements[count_i]] is not None and sum(i) > Max_Bonds[Elements[count_i]]:
            problem_dict[Elements[count_i]] += 1
            cons = sorted([ (Dist_Mat[count_i,count_j],count_j) if count_j > count_i else (Dist_Mat[count_j,count_i],count_j) for count_j,j in enumerate(i) if j == 1 ])[::-1]
            while sum(Adj_mat[count_i]) > Max_Bonds[Elements[count_i]]:
                sep,idx = cons.pop(0)
                Adj_mat[count_i,idx] = 0
                Adj_mat[idx,count_i] = 0

    return Adj_mat


# Returns a list with the number of electrons on each atom and a list with the number missing/surplus electrons on the atom
# 
# Inputs:  elements:  a list of element labels indexed to the adj_mat 
#          adj_mat:   np.array of atomic connections
#          bonding_pref: optional list of (index, bond_number) tuples that sets the target bond number of the indexed atoms
#          q_tot:     total charge on the molecule
#          fixed_bonds: optional list of (index_1,index_2,bond_number) tuples that creates fixed bonds between the index_1
#                       and index_2 atoms. No further bonds will be added or subtracted between these atoms.
#
# Optional inputs for ion and radical cases:
#          fc_0:      a list of formal charges on each atom
#          keep_lone: a list of atom index for which contains a radical 
#
# Returns: lone_electrons:
#          bonding_electrons:
#          core_electrons:
#          bond_mat:  an NxN matrix holding the bond orders between all atoms in the adj_mat
#          bonding_pref (optinal): optional list of (index, bond_number) tuples that sets the target bond number of the indexed atoms  
#
def find_lewis(elements,adj_mat_0,bonding_pref=[],q_tot=0,fixed_bonds=[],fc_0=None,keep_lone=[],return_pref=False,verbose=False,b_mat_only=False,return_FC=False,octet_opt=True,check_lewis_flag=False):
    
    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(find_lewis, "sat_dict"):

        find_lewis.lone_e = {'h':0, 'he':2,\
                             'li':0, 'be':2,                                                                                                                'b':0,     'c':0,     'n':2,     'o':4,     'f':6,    'ne':8,\
                             'na':0, 'mg':2,                                                                                                               'al':0,    'si':0,     'p':2,     's':4,    'cl':6,    'ar':8,\
                             'k':0, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':None, 'ge':0,    'as':3,    'se':4,    'br':6,    'kr':None,\
                             'rb':0, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':None, 'sn':None, 'sb':None, 'te':None,  'i':6,    'xe':None,\
                             'cs':0, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':None, 'pb':None, 'bi':None, 'po':None, 'at':None, 'rn':None }

        # Initialize periodic table
        find_lewis.periodic = {  "h": 1,  "he": 2,\
                                 "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                                 "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                  "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}
        
        # Electronegativity ordering (for determining lewis structure)
        find_lewis.en = { "h" :2.3,  "he":4.16,\
                          "li":0.91, "be":1.58,                                                                                                               "b" :2.05, "c" :2.54, "n" :3.07, "o" :3.61, "f" :4.19, "ne":4.79,\
                          "na":0.87, "mg":1.29,                                                                                                               "al":1.61, "si":1.91, "p" :2.25, "s" :2.59, "cl":2.87, "ar":3.24,\
                          "k" :0.73, "ca":1.03, "sc":1.19, "ti":1.38, "v": 1.53, "cr":1.65, "mn":1.75, "fe":1.80, "co":1.84, "ni":1.88, "cu":1.85, "zn":1.59, "ga":1.76, "ge":1.99, "as":2.21, "se":2.42, "br":2.69, "kr":2.97,\
                          "rb":0.71, "sr":0.96, "y" :1.12, "zr":1.32, "nb":1.41, "mo":1.47, "tc":1.51, "ru":1.54, "rh":1.56, "pd":1.58, "ag":1.87, "cd":1.52, "in":1.66, "sn":1.82, "sb":1.98, "te":2.16, "i" :2.36, "xe":2.58,\
                          "cs":0.66, "ba":0.88, "la":1.09, "hf":1.16, "ta":1.34, "w" :1.47, "re":1.60, "os":1.65, "ir":1.68, "pt":1.72, "au":1.92, "hg":1.76, "tl":1.79, "pb":1.85, "bi":2.01, "po":2.19, "at":2.39, "rn":2.60} 

        # Polarizability ordering (for determining lewis structure)
        find_lewis.pol ={ "h" :4.5,  "he":1.38,\
                          "li":164.0, "be":377,                                                                                                               "b" :20.5, "c" :11.3, "n" :7.4, "o" :5.3,  "f" :3.74, "ne":2.66,\
                          "na":163.0, "mg":71.2,                                                                                                              "al":57.8, "si":37.3, "p" :25.0,"s" :19.4, "cl":14.6, "ar":11.1,\
                          "k" :290.0, "ca":161.0, "sc":97.0, "ti":100.0, "v": 87.0, "cr":83.0, "mn":68.0, "fe":62.0, "co":55, "ni":49, "cu":47.0, "zn":38.7,  "ga":50.0, "ge":40.0, "as":30.0,"se":29.0, "br":21.0, "kr":16.8,\
                          "rb":320.0, "sr":197.0, "y" :162,  "zr":112.0, "nb":98.0, "mo":87.0, "tc":79.0, "ru":72.0, "rh":66, "pd":26.1, "ag":55, "cd":46.0,  "in":65.0, "sn":53.0, "sb":43.0,"te":28.0, "i" :32.9, "xe":27.3,}

        # Bond energy dictionary {}-{}-{} refers to atom1, atom2 additional bonds number (1 refers to double bonds)
        # If energy for multiple bonds is missing, it means it's unusual to form multiple bonds, such value will be -10000.0, if energy for single bonds if missing, directly take multiple bonds energy as the difference 
        # From https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Bond_Energies
        #find_lewis.be = { "6-6-1": 267, "6-6-2":492, "6-7-1":310, "6-7-2":586, "6-8-1":387, "6-8-2":714, "7-8-1":406, "7-7-1":258, "7-7-2":781, "8-8-1":349, "8-16-1":523, "16-16-1":152}
        # Or from https://www2.chemistry.msu.edu/faculty/reusch/OrgPage/bndenrgy.htm ("6-15-1" is missing)
        # Remove 6-16-1:73
        find_lewis.be = { "6-6-1": 63, "6-6-2":117, "6-7-1":74, "6-7-2":140, "6-8-1":92.5, "6-8-2":172.5, "7-7-1":70.6, "7-7-2":187.6, "7-8-1":88, "8-8-1":84, "8-15-1":20, "8-16-1":6, "15-15-1":84,"15-15-2": 117, "15-16-1":70}
        
        # Initialize periodic table
        find_lewis.atomic_to_element = { find_lewis.periodic[i]:i for i in find_lewis.periodic.keys() }

    # Consistency check on fc_0 argument, if supplied
    if fc_0 is not None:
        if len(fc_0) != len(elements):
            print("ERROR in find_lewis: the fc_0 and elements lists must have the same dimensions.")
            quit()
        if int(sum(fc_0)) != int(q_tot):
            print("ERROR in find_lewis: the sum of formal charges does not equal q_tot.")
            quit()

    # Initalize elementa and atomic_number lists for use by the function
    atomic_number = [ find_lewis.periodic[i.lower()] for i in elements ]
    adj_mat = deepcopy(adj_mat_0)

    # Initially assign all valence electrons as lone electrons
    lone_electrons    = np.zeros(len(elements),dtype="int")    
    bonding_electrons = np.zeros(len(elements),dtype="int")    
    core_electrons    = np.zeros(len(elements),dtype="int")
    valence           = np.zeros(len(elements),dtype="int")
    bonding_target    = np.zeros(len(elements),dtype="int")
    valence_list      = np.zeros(len(elements),dtype="int")    
    
    for count_i,i in enumerate(elements):

        # Grab the total number of (expected) electrons from the atomic number
        N_tot = atomic_number[count_i]   

        # Determine the number of core/valence electrons based on row in the periodic table
        if N_tot > 54:
            print("ERROR in find_lewis: the algorithm isn't compatible with atomic numbers greater than 54 owing to a lack of rules for treating lanthanides. Exiting...")
            quit()
        elif N_tot > 36:
            N_tot -= 36
            core_electrons[count_i] = 36
            valence[count_i]        = 18
        elif N_tot > 18:
            N_tot -= 18
            core_electrons[count_i] = 18
            valence[count_i]        = 18
        elif N_tot > 10:
            N_tot -= 10
            core_electrons[count_i] = 10
            valence[count_i]        = 8
        elif N_tot > 2:
            N_tot -= 2
            core_electrons[count_i] = 2
            valence[count_i]        = 8
        lone_electrons[count_i] = N_tot
        valence_list[count_i] = N_tot

        # Assign target number of bonds for this atom
        if count_i in [ j[0] for j in bonding_pref ]:
            bonding_target[count_i] = next( j[1] for j in bonding_pref if j[0] == count_i )
        else:
            bonding_target[count_i] = N_tot - find_lewis.lone_e[elements[count_i].lower()]       

    # Loop over the adjmat and assign initial bonded electrons assuming single bonds (and adjust lone electrons accordingly)
    for count_i,i in enumerate(adj_mat_0):
        bonding_electrons[count_i] += sum(i)
        lone_electrons[count_i] -= sum(i)

    # Apply keep_lone: add one electron to such index    
    for count_i in keep_lone:
        lone_electrons[count_i] += 1
        
    # Eliminate all radicals by forming higher order bonds
    change_list = range(len(lone_electrons))
    bonds_made = []    
    loop_list   = [ (atomic_number[i],i) for i in range(len(lone_electrons)) ]
    loop_list   = [ i[1] for i in sorted(loop_list) ]

    # Check for special chemical groups
    for i in range(len(elements)):
        # Handle nitro groups
        if is_nitro(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ]
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],1)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            lone_electrons[O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[1],i] += 1

        # Handle sulfoxide groups
        if is_sulfoxide(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the thioketone atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind[0])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[O_ind[0],i] += 1

        # Handle sulfonyl groups
        if is_sulfonyl(i,adj_mat_0,elements) is True:
            
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the sulfoxide atoms from the bonding_pref list
            bonding_pref += [(i,6)]
            bonding_pref += [(O_ind[0],2)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 2
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            bonds_made += [(i,O_ind[0])]
            bonds_made += [(i,O_ind[1])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[0],i] += 1
            adj_mat[O_ind[1],i] += 1            
        
        # Handle phosphate groups 
        if is_phosphate(i,adj_mat_0,elements) is True:
            O_ind      = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in ["o","O"] ] # Index of single bonded O-P oxygens
            O_ind_term = [ j for j in O_ind if sum(adj_mat_0[j]) == 1 ] # Index of double bonded O-P oxygens
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the phosphate atoms from the bonding_pref list
            bonding_pref += [(i,5)]
            bonding_pref += [(O_ind_term[0],2)]  # during testing it ended up being important to only add a bonding_pref tuple for one of the terminal oxygens
            bonding_electrons[O_ind_term[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind_term[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind_term[0])]
            adj_mat[i,O_ind_term[0]] += 1
            adj_mat[O_ind_term[0],i] += 1

        # Handle cyano groups
        if is_cyano(i,adj_mat_0,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat_0[count_j]) == 2 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,3)]
            bonding_pref += [(C_ind[0],4)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2

        # Handle isocyano groups
        if is_isocyano(i,adj_mat,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(C_ind[0],3)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2

    # Apply fixed_bonds argument
    off_limits=[]
    for i in fixed_bonds:

        # Initalize intermediate variables
        a = i[0]
        b = i[1]
        N = i[2]
        N_current = len([ j for j in bonds_made if (a,b) == j or (b,a) == j ]) + 1
        # Check that a bond exists between these atoms in the adjacency matrix
        if adj_mat_0[a,b] != 1:
            print("ERROR in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but the adjacency matrix doesn't reflect a bond. Exiting...")
            quit()

        # Check that less than or an equal number of bonds exist between these atoms than is requested
        if N_current > N:
            print("ERROR in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but {} bonds already exist between these atoms. There may be a conflict".format(N_current))
            print("                      between the special groups handling and the requested lewis_structure.")
            quit()

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[a] < (N - N_current):
            print("Warning in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(elements[a],lone_electrons[a]))

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[b] < (N - N_current):
            print("Warning in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(elements[b],lone_electrons[b]))
        

        # Make the bonds between the atoms
        for j in range(N-N_current):
            bonding_electrons[a] += 1
            bonding_electrons[b] += 1
            lone_electrons[a]    -= 1
            lone_electrons[b]    -= 1
            bonds_made += [ (a,b) ]

        # Append bond to off_limits group so that further bond additions/breaks do not occur.
        off_limits += [(a,b),(b,a)]

    # Turn the off_limits list into a set for rapid lookup
    off_limits = set(off_limits)
    
    # Adjust formal charges (if supplied)
    if fc_0 is not None:
        for count_i,i in enumerate(fc_0):
            if i > 0:
                #if lone_electrons[count_i] < i:
                    #print "ERROR in find_lewis: atom ({}, index {}) doesn't have enough lone electrons ({}) to be removed to satisfy the specified formal charge ({}).".format(elements[count_i],count_i,lone_electrons[count_i],i)
                    #quit()
                lone_electrons[count_i] = lone_electrons[count_i] - i
            if i < 0:
                lone_electrons[count_i] = lone_electrons[count_i] + int(abs(i))
        q_tot=0
    
    # diagnostic print            
    if verbose is True:
        print("Starting electronic structure:")
        print("\n{:40s} {:20} {:20} {:20} {:20} {}".format("elements","lone_electrons","bonding_electrons","core_electrons","formal_charge","bonded_atoms"))
        for count_i,i in enumerate(elements):
            print("{:40s} {:<20d} {:<20d} {:<20d} {:<20d} {}".format(elements[count_i],lone_electrons[count_i],bonding_electrons[count_i],core_electrons[count_i],\
                                                                     valence_list[count_i] - bonding_electrons[count_i] - lone_electrons[count_i],\
                                                                     ",".join([ "{}".format(count_j) for count_j,j in enumerate(adj_mat[count_i]) if j == 1 ])))

    # Initialize objects for use in the algorithm
    lewis_total = 1000
    lewis_lone_electrons = []
    lewis_bonding_electrons = []
    lewis_core_electrons = []
    lewis_valence = []
    lewis_bonding_target = []
    lewis_bonds_made = []
    lewis_adj_mat = []
    lewis_identical_mat = []
    
    # Determine the atoms with lone pairs that are unsatisfied as candidates for electron removal/addition to satisfy the total charge condition  
    happy = [ i[0] for i in bonding_pref if i[1] <= bonding_electrons[i[0]]]
    bonding_pref_ind = [i[0] for i in bonding_pref]
        
    # Determine is electrons need to be removed or added
    if q_tot > 0:
        adjust = -1
        octet_violate_e = []
        for count_j,j in enumerate(elements):
            if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] > 8:
                    octet_violate_e += [count_j]
            elif j.lower() in ["br","i"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] > 18:
                    octet_violate_e += [count_j]
        
        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy and count_i not in octet_violate_e]
    
    elif q_tot < 0:
        adjust = 1
        octet_violate_e = []
        for count_j,j in enumerate(elements):
            if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] < 8:
                    octet_violate_e += [count_j]
                    
            elif j.lower() in ["br","i"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] < 18:
                    octet_violate_e += [count_j]

        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy and count_i not in octet_violate_e]
        
    else:
        adjust = 1
        octet_violate_e = []
        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy ]
    
    # The outer loop checks each bonding structure produced by the inner loop for consistency with
    # the user specified "pref_bonding" and pref_argument with bonding electrons are
    for dummy_counter in range(lewis_total):
        lewis_loop_list = loop_list
        random.shuffle(lewis_loop_list)
        outer_counter     = 0
        inner_max_cycles  = 1000
        outer_max_cycles  = 1000
        bond_sat = False
        
        lewis_lone_electrons.append(deepcopy(lone_electrons))
        lewis_bonding_electrons.append(deepcopy(bonding_electrons))
        lewis_core_electrons.append(deepcopy(core_electrons))
        lewis_valence.append(deepcopy(valence))
        lewis_bonding_target.append(deepcopy(bonding_target))
        lewis_bonds_made.append(deepcopy(bonds_made))
        lewis_adj_mat.append(deepcopy(adj_mat))
        lewis_counter = len(lewis_lone_electrons) - 1
        
        # Adjust the number of electrons by removing or adding to the available lone pairs
        # The algorithm simply adds/removes from the first N lone pairs that are discovered
        random.shuffle(octet_violate_e)
        random.shuffle(normal_adjust)
        adjust_ind=octet_violate_e+normal_adjust
    
        if len(adjust_ind) >= abs(q_tot): 
            for i in range(abs(q_tot)):
                lewis_lone_electrons[-1][adjust_ind[i]] += adjust
                lewis_bonding_target[-1][adjust_ind[i]] += adjust 
        else:
            for i in range(abs(q_tot)):
                lewis_lone_electrons[-1][0] += adjust
                lewis_bonding_target[-1][0] += adjust

        # Search for an optimal lewis structure
        while bond_sat is False:
        
            # Initialize necessary objects
            change_list   = range(len(lewis_lone_electrons[lewis_counter]))
            inner_counter = 0
            bond_sat = True                
            # Inner loop forms bonds to remove radicals or underbonded atoms until no further
            # changes in the bonding pattern are observed.
            while len(change_list) > 0:
                change_list = []
                for i in lewis_loop_list:

                    # List of atoms that already have a satisfactory binding configuration.
                    happy = [ j[0] for j in bonding_pref if j[1] <= lewis_bonding_electrons[lewis_counter][j[0]]]            
                    
                    # If the current atom already has its target configuration then no further action is taken
                    if i in happy: continue

                    # If there are no lone electrons or too more bond formed then skip
                    if lewis_lone_electrons[lewis_counter][i] == 0: continue
                    
                    # Take action if this atom has a radical or an unsatifisied bonding condition
                    if lewis_lone_electrons[lewis_counter][i] % 2 != 0 or lewis_bonding_electrons[lewis_counter][i] != lewis_bonding_target[lewis_counter][i]:
                        # Try to form a bond with a neighboring radical (valence +1/-1 check ensures that no improper 5-bonded atoms are formed)
                        lewis_bonded_radicals = [ (-find_lewis.en[elements[count_j].lower()],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] % 2 != 0 \
                                                  and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j]\
                                                  and lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 and count_j not in happy ]

                        lewis_bonded_lonepairs= [ (-find_lewis.en[elements[count_j].lower()],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] > 0 \
                                                  and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j] and lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 \
                                                  and count_j not in happy ]

                        # Sort by atomic number (cheap way of sorting carbon before other atoms, should probably switch over to electronegativities) 
                        lewis_bonded_radicals  = [ j[1] for j in sorted(lewis_bonded_radicals) ]
                        lewis_bonded_lonepairs = [ j[1] for j in sorted(lewis_bonded_lonepairs) ]

                        # Correcting radicals is attempted first
                        if len(lewis_bonded_radicals) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_radicals[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_radicals[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_radicals[0]][i] += 1 
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_radicals[0]] -= 1
                            change_list += [i,lewis_bonded_radicals[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_radicals[0])]
                                                        
                        # Else try to form a bond with a neighboring atom with spare lone electrons (valence check ensures that no improper 5-bonded atoms are formed)
                        elif len(lewis_bonded_lonepairs) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_lonepairs[0]][i] += 1
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_lonepairs[0]] -= 1
                            change_list += [i,lewis_bonded_lonepairs[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_lonepairs[0])]
                            #lewis_bonds_en[lewis_counter] += 1.0/find_lewis.en[elements[i].lower()]/find_lewis.en[elements[lewis_bonded_lonepairs[0]].lower()]
                
                # Increment the counter and break if the maximum number of attempts have been made
                inner_counter += 1
                if inner_counter >= inner_max_cycles:
                    print("WARNING: maximum attempts to establish a reasonable lewis-structure exceeded ({}).".format(inner_max_cycles))
            
            # Check if the user specified preferred bond order has been achieved.
            if bonding_pref is not None:
                unhappy = [ i[0] for i in bonding_pref if i[1] != lewis_bonding_electrons[lewis_counter][i[0]]]            
                if len(unhappy) > 0:

                    # Break the first bond involving one of the atoms bonded to the under/over coordinated atoms
                    ind = set([unhappy[0]] + [ count_i for count_i,i in enumerate(adj_mat_0[unhappy[0]]) if i == 1 and (count_i,unhappy[0]) not in off_limits ])
                    
                    # Check if a rearrangment is possible, break if none are available
                    try:
                        break_bond = next( i for i in lewis_bonds_made[lewis_counter] if i[0] in ind or i[1] in ind )
                    except:
                        print("WARNING: no further bond rearrangments are possible and bonding_pref is still not satisfied.")
                        break
                    
                    # Perform bond rearrangment
                    lewis_bonding_electrons[lewis_counter][break_bond[0]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[0]] += 1
                    lewis_adj_mat[lewis_counter][break_bond[0]][break_bond[1]] -= 1
                    lewis_adj_mat[lewis_counter][break_bond[1]][break_bond[0]] -= 1
                    lewis_bonding_electrons[lewis_counter][break_bond[1]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[1]] += 1

                    # Remove the bond from the list and reorder lewis_loop_list so that the indices involved in the bond are put last                
                    lewis_bonds_made[lewis_counter].remove(break_bond)
                    lewis_loop_list.remove(break_bond[0])
                    lewis_loop_list.remove(break_bond[1])
                    lewis_loop_list += [break_bond[0],break_bond[1]]
                                        
                    # Update the bond_sat flag
                    bond_sat = False
                    
                # Increment the counter and break if the maximum number of attempts have been made
                outer_counter += 1
                    
                # Periodically reorder the list to avoid some cyclical walks
                if outer_counter % 100 == 0:
                    lewis_loop_list = reorder_list(lewis_loop_list,atomic_number)

                # Print diagnostic upon failure
                if outer_counter >= outer_max_cycles:
                    print("WARNING: maximum attempts to establish a lewis-structure consistent")
                    print("         with the user supplied bonding preference has been exceeded ({}).".format(outer_max_cycles))
                    break
        
        # Re-apply keep_lone: remove one electron from such index    
        for count_i in keep_lone:
            lewis_lone_electrons[lewis_counter][count_i] -= 1

        # Special cases, share pair of electrons
        total_electron=np.array(lewis_lone_electrons[lewis_counter])+np.array(lewis_bonding_electrons[lewis_counter])*2
        
        # count for atom which doesn't satisfy
        # Notice: need systematical check for this part !!!
        unsatisfy = [count_t for count_t,te in enumerate(total_electron) if te > 2 and te < 8 and te % 2 ==0]
        for uns in unsatisfy:
            full_connect=[count_i for count_i,i in enumerate(adj_mat_0[uns]) if i == 1 and total_electron[count_i] == 8 and lewis_lone_electrons[lewis_counter][count_i] >= 2]
            if len(full_connect) > 0:

                lewis_lone_electrons[lewis_counter][full_connect[0]]-=2
                lewis_bonding_electrons[lewis_counter][uns]+=1
                lewis_bonding_electrons[lewis_counter][full_connect[0]]+=1
                lewis_adj_mat[lewis_counter][uns][full_connect[0]]+=1
                lewis_adj_mat[lewis_counter][full_connect[0]][uns]+=1 

        # Delete last entry in the lewis np.arrays if the electronic structure is not unique: introduce identical_mat includes both info of bond_mats and formal_charges
        identical_mat=np.vstack([lewis_adj_mat[-1], np.array([ valence_list[k] - lewis_bonding_electrons[-1][k] - lewis_lone_electrons[-1][k] for k in range(len(elements)) ]) ])
        lewis_identical_mat.append(identical_mat)
        
        if array_unique(lewis_identical_mat[-1],lewis_identical_mat[:-1]) is False :
            lewis_lone_electrons    = lewis_lone_electrons[:-1]
            lewis_bonding_electrons = lewis_bonding_electrons[:-1]
            lewis_core_electrons    = lewis_core_electrons[:-1]
            lewis_valence           = lewis_valence[:-1]
            lewis_bonding_target    = lewis_bonding_target[:-1]
            lewis_bonds_made        = lewis_bonds_made[:-1]
            lewis_adj_mat           = lewis_adj_mat[:-1]
            lewis_identical_mat     = lewis_identical_mat[:-1]
            
    # Find the total number of lone electrons in each structure
    lone_electrons_sums = []
    for i in range(len(lewis_lone_electrons)):
        lone_electrons_sums.append(sum(lewis_lone_electrons[i]))
        
    # Find octet violations in each structure
    octet_violations = []
    for i in range(len(lewis_lone_electrons)):
        ov = 0
        if octet_opt is True:
            for count_j,j in enumerate(elements):
                if j.lower() in ["c","n","o","f","si","p","s","cl","br","i"] and count_j not in bonding_pref_ind:
                    if lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j] != 8 and lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j] != 18:
                        ov += 1
        octet_violations.append(ov)

    ## Calculate bonding energy
    lewis_bonds_energy = []
    for bonds_made in lewis_bonds_made:
        for lb,bond_made in enumerate(bonds_made): bonds_made[lb]=tuple(sorted(bond_made))
        count_bonds_made = ["{}-{}-{}".format(min(atomic_number[bm[0]],atomic_number[bm[1]]),max(atomic_number[bm[0]],atomic_number[bm[1]]),bonds_made.count(bm) ) for bm in set(bonds_made)]
        lewis_bonds_energy += [sum([find_lewis.be[cbm] if cbm in find_lewis.be.keys() else -10000.0 for cbm in count_bonds_made  ]) ]
    # normalize the effect
    lewis_bonds_energy = [-be/max(1,max(lewis_bonds_energy)) for be in lewis_bonds_energy]

    ## Find the total formal charge for each structure
    formal_charges_sums = []
    for i in range(len(lewis_lone_electrons)):
        fc = 0
        for j in range(len(elements)):
            fc += valence_list[j] - lewis_bonding_electrons[i][j] - lewis_lone_electrons[i][j]
        formal_charges_sums.append(fc)
    
    ## Find formal charge eletronegativity contribution
    lewis_formal_charge = [ [ valence_list[i] - lewis_bonding_electrons[_][i] - lewis_lone_electrons[_][i] for i in range(len(elements)) ] for _ in range(len(lewis_lone_electrons)) ]
    lewis_keep_lone     = [ [ count_i for count_i,i in enumerate(lone) if i % 2 != 0] for lone in lewis_lone_electrons]
    lewis_fc_en = []  # Electronegativity for stabling charge/radical
    lewis_fc_pol = [] # Polarizability for stabling charge/radical
    lewis_fc_hc  = [] # Hyper-conjugation contribution
    for i in range(len(lewis_lone_electrons)):
        formal_charge = lewis_formal_charge[i]
        radical_atom = lewis_keep_lone[i]
        fc_ind = [(count_j,j) for count_j,j in enumerate(formal_charge) if j != 0]
        for R_ind in radical_atom:  # assign +0.5 for radical
            fc_ind += [(R_ind,0.5)]
        
        # initialize en,pol and hc
        fc_en,fc_pol,fc_hc = 0,0,0
        
        # Loop over formal charges and radicals
        for count_fc in fc_ind:
            ind = count_fc[0]    
            charge = count_fc[1]
            # Count the self contribution: (-) on the most electronegative atom and (+) on the least electronegative atom
            fc_en += 10 * charge * find_lewis.en[elements[ind].lower()]
             
            # Find the nearest and next-nearest atoms for each formal_charge/radical contained atom
            gs = graph_seps(adj_mat_0)
            nearest_atoms = [count_k for count_k,k in enumerate(lewis_adj_mat[i][ind]) if k >= 1] 
            NN_atoms = list(set([ count_j for count_j,j in enumerate(gs[ind]) if j == 2 ]))
            
            # only count when en > en(C)
            fc_en += charge*(sum([find_lewis.en[elements[count_k].lower()] for count_k in nearest_atoms if find_lewis.en[elements[count_k].lower()] > 2.54] )+\
                             sum([find_lewis.en[elements[count_k].lower()] for count_k in NN_atoms if find_lewis.en[elements[count_k].lower()] > 2.54] ) * 0.1 )

            if charge < 0: # Polarizability only affects negative charge
                fc_pol += charge*sum([find_lewis.pol[elements[count_k].lower()] for count_k in nearest_atoms ])

            # find hyper-conjugation strcuture
            nearby_carbon = [nind for nind in nearest_atoms if elements[nind].lower()=='c']
            for carbon_ind in nearby_carbon:
                carbon_nearby=[nind for nind in NN_atoms if lewis_adj_mat[i][carbon_ind][nind] >= 1 and elements[nind].lower() in ['c','h']]
                if len(carbon_nearby) == 3: fc_hc -= charge*(len([nind for nind in carbon_nearby if elements[nind].lower() == 'c'])*2 + len([nind for nind in carbon_nearby if elements[nind].lower() == 'h']))

        lewis_fc_en.append(fc_en)        
        lewis_fc_pol.append(fc_pol)        
        lewis_fc_hc.append(fc_hc)        

    # normalize the effect
    lewis_fc_en = [lfc/max(1,max(abs(np.array(lewis_fc_en)))) for lfc in lewis_fc_en]
    lewis_fc_pol= [lfp/max(1,max(abs(np.array(lewis_fc_pol)))) for lfp in lewis_fc_pol]
    
    # Add the total number of radicals to the total formal charge to determine the criteria.
    # The radical count is scaled by 0.01 and the lone pair count is scaled by 0.001. This results
    # in the structure with the lowest formal charge always being returned, and the radical count 
    # only being considered if structures with equivalent formal charges are found, and likewise with
    # the lone pair count. The structure(s) with the lowest score will be returned.
    lewis_criteria = []
    for i in range(len(lewis_lone_electrons)):
        #lewis_criteria.append( 10.0*octet_violations[i] + abs(formal_charges_sums[i]) + 0.1*sum([ 1 for j in lewis_lone_electrons[i] if j % 2 != 0 ]) + 0.001*lewis_bonds_energy[i]  + 0.00001*lewis_fc_en[i] + 0.000001*lewis_fc_pol[i] + 0.0000001*lewis_fc_hc[i]) 
        lewis_criteria.append( 10.0*octet_violations[i] + abs(formal_charges_sums[i]) + 0.1*sum([ 1 for j in lewis_lone_electrons[i] if j % 2 != 0 ]) + 0.01*lewis_fc_en[i] + 0.005*lewis_fc_pol[i] +\
                               0.0001*lewis_fc_hc[i] + 0.0001*lewis_bonds_energy[i]) 
    best_lewis = [i[0] for i in sorted(enumerate(lewis_criteria), key=lambda x:x[1])]  # sort from least to most and return a list containing the origial list's indices in the correct order
    best_lewis = [ i for i in best_lewis if lewis_criteria[i] == lewis_criteria[best_lewis[0]] ]    

    # Finally check formal charge to keep those with 
    lewis_re_fc     = [ lewis_formal_charge[_]+lewis_keep_lone[_] for _ in best_lewis]
    appear_times    = [ lewis_re_fc.count(i) for i in lewis_re_fc]
    best_lewis      = [best_lewis[i] for i in range(len(lewis_re_fc)) if appear_times[i] == max(appear_times) ] 
    
    # Apply keep_lone information, remove the electron to form lone electron
    for i in best_lewis:
        for j in keep_lone:
            lewis_lone_electrons[i][j] -= 1

    # Print diagnostics
    if verbose is True:
        for i in best_lewis:
            print("Bonding Matrix  {}".format(i))
            print("Formal_charge:  {}".format(formal_charges_sums[i]))
            print("Lewis_criteria: {}\n".format(lewis_criteria[i]))
            print("{:<40s} {:<40s} {:<15s} {:<15s}".format("Elements","Bond_Mat","Lone_Electrons","FC"))
            for j in range(len(elements)):
                print("{:<40s} {}    {} {}".format(elements[j]," ".join([ str(k) for k in lewis_adj_mat[i][j] ]),lewis_lone_electrons[i][j],valence_list[j] - lewis_bonding_electrons[i][j] - lewis_lone_electrons[i][j]))
            print (" ")

    # If only the bonding matrix is requested, then only that is returned
    if b_mat_only is True:
        if return_FC is False:  
            return [ lewis_adj_mat[_] for _ in best_lewis ]
        else:
            return [ lewis_adj_mat[_] for _ in best_lewis ], [ lewis_formal_charge[_] for _ in best_lewis ]

    # return like check_lewis function
    if check_lewis_flag is True:
        if return_pref is True:
            return lewis_lone_electrons[best_lewis[0]], lewis_bonding_electrons[best_lewis[0]], lewis_core_electrons[best_lewis[0]],bonding_pref
        else:
            return lewis_lone_electrons[best_lewis[0]], lewis_bonding_electrons[best_lewis[0]], lewis_core_electrons[best_lewis[0]]
    
    # Optional bonding pref return to handle cases with special groups
    if return_pref is True:
        if return_FC is False:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],bonding_pref
        else:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],[ lewis_formal_charge[_] for _ in best_lewis ],bonding_pref 

    else:
        if return_FC is False:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ]
        else:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],[ lewis_formal_charge[_] for _ in best_lewis ]

# Returns an NxN matrix holding the bond orders between all atoms in the molecular structure.
# 
# Inputs:  elements:  a list of element labels indexed to the adj_mat
#          adj_mat:   a list of bonds indexed to the elements list
#
# Returns: 
#          bond_mat:  an NxN matrix holding the bond orders between all atoms in the adj_mat
#
def get_bonds(elements,adj_mat,verbose=False):

    # Initialize the saturation dictionary the first time this function is called
    if not hasattr(get_bonds, "sat_dict"):
        get_bonds.sat_dict = {  'H':1, 'He':1,\
                               'Li':1, 'Be':2,                                                                                                                'B':3,     'C':4,     'N':3,     'O':2,     'F':1,    'Ne':1,\
                               'Na':1, 'Mg':2,                                                                                                               'Al':3,    'Si':4,     'P':3,     'S':2,    'Cl':1,    'Ar':1,\
                                'K':1, 'Ca':2, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':3,    'Ge':3,    'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                               'Rb':1, 'Sr':2,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                               'Cs':1, 'Ba':2, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }


        # Initialize periodic table
        get_bonds.periodic = { "h": 1,  "he": 2,\
                              "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                              "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                               "k":19,  "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                              "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                              "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

        get_bonds.lone_e = {   'h':0, 'he':2,\
                              'li':0, 'be':2,                                                                                                                'b':0,     'c':0,     'n':2,     'o':4,     'f':6,    'ne':8,\
                              'na':0, 'mg':2,                                                                                                               'al':0,    'si':0,     'p':2,     's':4,    'cl':6,    'ar':8,\
                               'k':0, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':10,   'ge':0,    'as':3,    'se':4,    'br':6,    'kr':None,\
                              'rb':0, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':None, 'sn':None, 'sb':None, 'te':None,  'i':6,    'xe':None,\
                              'cs':0, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':None, 'pb':None, 'bi':None, 'po':None, 'at':None, 'rn':None  }

        # Initialize periodic table
        get_bonds.atomic_to_element = { get_bonds.periodic[i]:i for i in get_bonds.periodic.keys() }

    # Initalize atomic_number lists for use by the function
    atomic_number = [ get_bonds.periodic[i.lower()] for i in elements ]
    
    # Generate the bonding preference to optimally satisfy the valency conditions of each atom 
    bonding_pref = [ (count_i,get_bonds.sat_dict[i]) for count_i,i in enumerate(elements) ]

    # Convert to atomic numbers is working in elements mode
    atomtypes = [ "["+str(get_bonds.periodic[i.lower()])+"]" for i in elements ]

    # Initially assign all valence electrons as lone electrons
    lone_electrons    = np.zeros(len(atomtypes),dtype="int")    
    bonding_electrons = np.zeros(len(atomtypes),dtype="int")    
    core_electrons    = np.zeros(len(atomtypes),dtype="int")
    valence           = np.zeros(len(atomtypes),dtype="int")
    bonding_target    = np.zeros(len(atomtypes),dtype="int")
    for count_i,i in enumerate(atomtypes):

        # Grab the total number of (expected) electrons from the atomic number
        N_tot = int(i.split('[')[1].split(']')[0])

        # Determine the number of core/valence electrons based on row in the periodic table
        if N_tot > 54:
            print( "ERROR in get_bonds: the algorithm isn't compatible with atomic numbers greater than 54 owing to a lack of rules for treating lanthanides. Exiting...")
            quit()
        elif N_tot > 36:
            N_tot -= 36
            core_electrons[count_i] = 36
            valence[count_i]        = 18
        elif N_tot > 18:
            N_tot -= 18
            core_electrons[count_i] = 18
            valence[count_i]        = 18
        elif N_tot > 10:
            N_tot -= 10
            core_electrons[count_i] = 10
            valence[count_i]        = 8
        elif N_tot > 2:
            N_tot -= 2
            core_electrons[count_i] = 2
            valence[count_i]        = 8
        lone_electrons[count_i] = N_tot
        bonding_target[count_i] = N_tot - get_bonds.lone_e[get_bonds.atomic_to_element[int(i.split('[')[1].split(']')[0])]]    

    # Loop over the adjmat and assign initial bonded electrons assuming single bonds (and adjust lone electrons accordingly)
    for count_i,i in enumerate(adj_mat):
        bonding_electrons[count_i] += sum(i)
        lone_electrons[count_i] -= sum(i)

    # Eliminate all radicals by forming higher order bonds
    change_list = range(len(lone_electrons))
    outer_counter     = 0
    inner_max_cycles  = 1000
    outer_max_cycles  = 1000
    loop_list   = [ (atomic_number[i],i) for i in range(len(lone_electrons)) ]
    loop_list   = [ i[1] for i in sorted(loop_list) ]
    bonds_made = []
    bond_sat = False

    # Check for special chemical groups
    for i in range(len(atomtypes)):

        # Handle nitro groups
        if is_nitro(i,adj_mat,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ]
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],1)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            lone_electrons[O_ind[0]] += 1
            bonds_made += [(i,O_ind[1])]

        # Handle sulfoxide groups
        if is_sulfoxide(i,adj_mat,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the sulfoxide atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind[0])]

        # Handle sulfonyl groups
        if is_sulfonyl(i,adj_mat,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the sulfonyl atoms from the bonding_pref list
            bonding_pref += [(i,6)]
            bonding_pref += [(O_ind[0],2)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 2
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            bonds_made += [(i,O_ind[0])]
            bonds_made += [(i,O_ind[1])]

        # Handle phosphate groups 
        if is_phosphate(i,adj_mat,elements) is True:
            O_ind      = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] ] # Index of single bonded O-P oxygens
            O_ind_term = [ j for j in O_ind if sum(adj_mat[j]) == 1 ] # Index of double bonded O-P oxygens
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the phosphate atoms from the bonding_pref list
            bonding_pref += [(i,5)]
            bonding_pref += [(O_ind_term[0],2)] # during testing it ended up being important to only add a bonding_pref tuple for one of the terminal oxygens
            bonding_electrons[O_ind_term[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind_term[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind_term[0])]

    # diagnostic print            
    if verbose is True:
        print( "Starting electronic structure:")
        print( "\n{:40s} {:60} {:20} {:20} {:20} {}".format("elements","atomtypes","lone_electrons","bonding_electrons","core_electrons","bonded_atoms"))
        for count_i,i in enumerate(atomtypes):
            print( "{:40s} {:60} {:<20d} {:<20d} {:<20d} {}".format(elements[count_i],i,lone_electrons[count_i],bonding_electrons[count_i],core_electrons[count_i],",".join([ "{}".format(count_j) for count_j,j in enumerate(adj_mat[count_i]) if j == 1 ])))

    # The outer loop checks each bonding structure produced by the inner loop for consistency with
    # the user specified "pref_bonding" and pref_argument with bonding electrons are
    while bond_sat is False:

        # Initialize necessary objects
        change_list   = range(len(lone_electrons))
        inner_counter = 0
        bond_sat = True
        random.shuffle(loop_list)
        
        # Inner loop forms bonds to remove radicals or underbonded atoms until no further
        # changes in the bonding pattern are observed.
        while len(change_list) > 0:
            change_list = []
            for i in loop_list:

                # List of atoms that already have a satisfactory binding configuration.
                happy = [ j[0] for j in bonding_pref if j[1] == bonding_electrons[j[0]]]            
                
                # If the current atom already has its target configuration then no further action is taken
                if i in happy: continue

                # If there are no lone electrons then skip
                if lone_electrons[i] == 0: continue

                # Take action if this atom has a radical or an unsatifisied bonding condition
                if lone_electrons[i] % 2 != 0 or bonding_electrons[i] != bonding_target[i]:

                    # Try to form a bond with a neighboring radical (valence +1/-1 check ensures that no improper 5-bonded atoms are formed)
                    bonded_radicals = [ (atomic_number[count_j],count_j) for count_j,j in enumerate(adj_mat[i]) if j == 1 and lone_electrons[count_j] % 2 != 0 \
                                        and 2*(bonding_electrons[count_j]+1)+(lone_electrons[count_j]-1) <= valence[count_j] and lone_electrons[count_j]-1 >= 0 and count_j not in happy ]
                    bonded_lonepairs = [ (atomic_number[count_j],count_j) for count_j,j in enumerate(adj_mat[i]) if j == 1 and lone_electrons[count_j] > 0 \
                                        and 2*(bonding_electrons[count_j]+1)+(lone_electrons[count_j]-1) <= valence[count_j] and lone_electrons[count_j]-1 >= 0 and count_j not in happy ]

                    # Sort by atomic number (cheap way of sorting carbon before other atoms, should probably switch over to electronegativities) 
                    bonded_radicals = [ j[1] for j in  sorted(bonded_radicals) ]
                    bonded_lonepairs = [ j[1] for j in  sorted(bonded_lonepairs) ]

                    # Correcting radicals is attempted first
                    if len(bonded_radicals) > 0:
                        bonding_electrons[i] += 1
                        bonding_electrons[bonded_radicals[0]] += 1
                        lone_electrons[i] -= 1
                        lone_electrons[bonded_radicals[0]] -= 1
                        change_list += [i,bonded_radicals[0]]
                        bonds_made += [(i,bonded_radicals[0])]

                    # Else try to form a bond with a neighboring atom with spare lone electrons (valence check ensures that no improper 5-bonded atoms are formed)
                    elif len(bonded_lonepairs) > 0:
                        bonding_electrons[i] += 1
                        bonding_electrons[bonded_lonepairs[0]] += 1
                        lone_electrons[i] -= 1
                        lone_electrons[bonded_lonepairs[0]] -= 1
                        change_list += [i,bonded_lonepairs[0]]
                        bonds_made += [(i,bonded_lonepairs[0])]
            
            # Increment the counter and break if the maximum number of attempts have been made
            inner_counter += 1
            if inner_counter >= inner_max_cycles:
                print( "WARNING: maximum attempts to establish a reasonable lewis-structure exceeded ({}).".format(inner_max_cycles))

        # Check if the user specified preferred bond order has been achieved.
        if bonding_pref is not None:
            unhappy = [ i[0] for i in bonding_pref if i[1] != bonding_electrons[i[0]]]
            if len(unhappy) > 0:

                # Break the first bond involving one of the atoms bonded to the under/over coordinated atoms
                # NOTE: Added check since nitro-containing groups can lead to situations with no bonds being formed                
                ind = set([unhappy[0]] + [ count_i for count_i,i in enumerate(adj_mat[unhappy[0]]) if i == 1 ])

                # Check if a rearrangment is possible, break if none are available
                try:
                    break_bond = next( i for i in bonds_made if i[0] in ind or i[1] in ind )
                except:
                    print( "WARNING: no further bond rearrangments are possible and bonding_pref is still not satisfied.")
                    break

                # Perform bond rearrangment
                bonding_electrons[break_bond[0]] -= 1
                lone_electrons[break_bond[0]] += 1
                bonding_electrons[break_bond[1]] -= 1
                lone_electrons[break_bond[1]] += 1

                # Remove the bond from the list and reorder loop_list so that the indices involved in the bond are put last                
                bonds_made.remove(break_bond)
                loop_list.remove(break_bond[0])
                loop_list.remove(break_bond[1])
                loop_list += [break_bond[0],break_bond[1]]

                # Update the bond_sat flag
                bond_sat = False

            # Increment the counter and break if the maximum number of attempts have been made
            outer_counter += 1

            # Periodically reorder the list to avoid some cyclical walks
            if outer_counter % 100 == 0:
                loop_list = reorder_list(loop_list,atomic_number)

            # Print diagnostic upon failure
            if outer_counter >= outer_max_cycles:
                print( "WARNING: maximum attempts to establish a lewis-structure consistent")
                print( "         with the user supplied bonding preference has been exceeded ({}).".format(outer_max_cycles))
                break

    # diagnostic print            
    if verbose is True:
        print( "\nFinal electronic structure:")
        print( "\n{:40s} {:60} {:20} {:20} {:20} {}".format("elements","atomtypes","lone_electrons","bonding_electrons","core_electrons","bonded_atoms"))
        for count_i,i in enumerate(atomtypes):
            print( "{:40s} {:60} {:<20d} {:<20d} {:<20d} {}".format(elements[count_i],i,lone_electrons[count_i],bonding_electrons[count_i],core_electrons[count_i],",".join([ "{}".format(count_j) for count_j,j in enumerate(adj_mat[count_i]) if j == 1 ])))

    # Create the bond matrix
    bond_mat = adj_mat.copy()
    for i in bonds_made:
        bond_mat[i[0],i[1]] += 1
        bond_mat[i[1],i[0]] += 1

    return bond_mat

# Return bool depending on if the atom is a nitro nitrogen atom
def is_nitro(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    if len(O_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfoxide sulfur atom
def is_sulfoxide(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] ] 
    if len(O_ind) == 1 and len(C_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_sulfonyl(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] ] 
    if len(O_ind) == 2 and len(C_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a phosphate phosphorus atom
def is_phosphate(i,adj_mat,elements):

    status = False
    if elements[i] not in ["P","p"]:
        return False
    O_ind      = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] ] 
    O_ind_term = [ j for j in O_ind if sum(adj_mat[j]) == 1 ]
    if len(O_ind) == 4 and sum(adj_mat[i]) == 4 and len(O_ind_term) > 0:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_cyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 2 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_isocyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 1 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

# Description: Checks if an np.array "a" is unique compared with a list of np.arrays "a_list"
#              at the first match False is returned.
def array_unique(a,a_list):
    for i in a_list:
        if np.array_equal(a,i):
            return False
    return True

# Description: returns a canonicallized TAFFI bond. TAFFI bonds are written so that the lesser *atom_type* between 1 and 2 is first. 
# 
# inputs:      types: a list of taffi atom types defining the bond
#              ind:   a list of indices corresponding to the bond
#
# returns:     a canonically ordered bond (and list of indices if ind was supplied)
def canon_bond(types,ind=None):

    # consistency checks
    if len(types) != 2: 
        print( "ERROR in canon_bond: the supplied dihedral doesn't have two elements. Exiting...")
        quit()
    if ind != None and len(ind) != 2: 
        print( "ERROR in canon_bond: the iterable supplied to ind doesn't have two elements. Exiting...")
        quit()
        
    # bond types are written so that the lesser *atom_type* between 1 and 2 is first.
    if types[0] <= types[1]:
        if ind == None:
            return types
        else:
            return types,ind
    else:
        if ind == None:
            return types[::-1]
        else:
            return types[::-1],ind[::-1]

# Description: returns a canonicallized TAFFI angle. TAFFI angles are written so that the lesser *atom_type* between 1 and 3 is first. 
# 
# inputs:      types: a list of taffi atom types defining the angle
#              ind:   a list of indices corresponding to the angle
#
# returns:     a canonically ordered angle (and list of indices if ind was supplied)
def canon_angle(types,ind=None):

    # consistency checks
    if len(types) != 3: 
        print( "ERROR in canon_angle: the supplied dihedral doesn't have three elements. Exiting...")
        quit()
    if ind != None and len(ind) != 3: 
        print( "ERROR in canon_angle: the iterable supplied to ind doesn't have three elements. Exiting...")
        quit()
        
    # angle types are written so that the lesser *atom_type* between 1 and 3 is first.
    if types[0] <= types[2]:
        if ind == None:
            return types
        else:
            return types,ind
    else:
        if ind == None:
            return types[::-1]
        else:
            return types[::-1],ind[::-1]

# Description: returns a canonicallized TAFFI dihedral. TAFFI dihedrals are written so that the lesser *atom_type* between 1 and 4 is first. 
#              In the event that 1 and 4 are of the same type, then the lesser of 2 and 3 goes first. 
#
# inputs:      types: a list of taffi atom types defining the dihedral
#              ind:   a list of indices corresponding to the dihedral
#
# returns:     a canonically ordered dihedral (and list of indices if ind was supplied)
def canon_dihedral(types_0,ind=None):
    
    # consistency checks
    if len(types_0) < 4: 
        print( "ERROR in canon_dihedral: the supplied dihedral has less than four elements. Exiting...")
        quit()
    if ind != None and len(ind) != 4: 
        print( "ERROR in canon_dihedral: the iterable supplied to ind doesn't have four elements. Exiting...")
        quit()

    # Grab the types and style component (the fifth element if available)
    types = list(types_0[:4])
    if len(types_0) > 4:
        style = [types_0[4]]
    else:
        style = []

    # dihedral types are written so that the lesser *atom_type* between 1 and 4 is first.
    # In the event that 1 and 4 are of the same type, then the lesser of 2 and 3 goes first
    if types[0] == types[3]:
        if types[1] <= types[2]:
            if ind == None:
                return tuple(types+style)
            else:
                return tuple(types+style),ind
        else:
            if ind == None:
                return tuple(types[::-1]+style)
            else:
                return tuple(types[::-1]+style),ind[::-1]
    elif types[0] < types[3]:
        if ind == None:
            return tuple(types+style)
        else:
            return tuple(types+style),ind
    else:
        if ind == None:
            return tuple(types[::-1]+style)
        else:
            return tuple(types[::-1]+style),ind[::-1]

# Description: returns a canonicallized TAFFI improper. TAFFI impropers are written so that 
#              the three peripheral *atom_types* are written in increasing order.
#
# inputs:      types: a list of taffi atom types defining the improper
#              ind:   a list of indices corresponding to the improper
#
# returns:     a canonically ordered improper (and list of indices if ind was supplied)
def canon_improper(types,ind=None):

    # consistency checks
    if len(types) != 4: 
        print( "ERROR in canon_improper: the supplied improper doesn't have four elements. Exiting...")
        quit()
    if ind != None and len(ind) != 4: 
        print( "ERROR in canon_improper: the iterable supplied to ind doesn't have four elements. Exiting...")
        quit()
        
    # improper types are written so that the lesser *atom_type* between 1 and 4 is first.
    # In the event that 1 and 4 are of the same type, then the lesser of 2 and 3 goes first
    if ind == None:
        return tuple([types[0]]+sorted(types[1:]))
    else:
        tmp_types,tmp_ind = zip(*sorted(zip(types[1:],ind[1:])))
        return tuple([types[0]]+list(tmp_types[:])),tuple([ind[0]]+list(tmp_ind[:]))


# Helper function to check_lewis and get_bonds that rolls the loop_list carbon elements
def reorder_list(loop_list,atomic_number):
    c_types = [ count_i for count_i,i in enumerate(loop_list) if atomic_number[i] == 6 ]
    others  = [ count_i for count_i,i in enumerate(loop_list) if atomic_number[i] != 6 ]
    if len(c_types) > 1:
        c_types = c_types + [c_types.pop(0)]
    return [ loop_list[i] for i in c_types+others ]


# Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix 
def graph_seps(adj_mat_0):

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)
    
    # Initialize an np.array to hold the graphical separations with -1 for all unassigned elements and 0 for the diagonal.
    seps = np.ones([len(adj_mat),len(adj_mat)])*-1
    np.fill_diagonal(seps,0)

    # Perform searches out to len(adj_mat) bonds (maximum distance for a graph with len(adj_mat) nodes
    for i in np.arange(len(adj_mat)):        

        # All perform assignments to unassigned elements (seps==-1) 
        # and all perform an assignment if the value in the adj_mat is > 0        
        seps[np.where((seps==-1)&(adj_mat>0))] = i+1

        # Since we only care about the leading edge of the search and not the actual number of paths at higher orders, we can 
        # set the larger than 1 values to 1. This ensures numerical stability for larger adjacency matrices.
        adj_mat[np.where(adj_mat>1)] = 1
        
        # Break once all of the elements have been assigned
        if -1 not in seps:
            break

        # Take the inner product of the A**(i+1) with A**(1)
        adj_mat = np.dot(adj_mat,adj_mat_0)

    return seps


# Description:
# Rotate Point by an angle, theta, about the vector with an orientation of v1 passing through v2. 
# Performs counter-clockwise rotations (i.e., if the direction vector were pointing
# at the spectator, the rotations would appear counter-clockwise)
# For example, a 90 degree rotation of a 0,0,1 about the canonical 
# y-axis results in 1,0,0.
#
# Point: 1x3 np.array, coordinates to be rotated
# v1: 1x3 np.array, point the rotation passes through
# v2: 1x3 np.array, rotation direction vector
# theta: scalar, magnitude of the rotation (defined by default in degrees)
def axis_rot(Point,v1,v2,theta,mode='angle'):

    # Temporary variable for performing the transformation
    rotated=np.array([Point[0],Point[1],Point[2]])

    # If mode is set to 'angle' then theta needs to be converted to radians to be compatible with the
    # definition of the rotation vectors
    if mode == 'angle':
        theta = theta*np.pi/180.0

    # Rotation carried out using formulae defined here (11/22/13) http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/)
    # Adapted for the assumption that v1 is the direction vector and v2 is a point that v1 passes through
    a = v2[0]
    b = v2[1]
    c = v2[2]
    u = v1[0]
    v = v1[1]
    w = v1[2]
    L = u**2 + v**2 + w**2

    # Rotate Point
    x=rotated[0]
    y=rotated[1]
    z=rotated[2]

    # x-transformation
    rotated[0] = ( a * ( v**2 + w**2 ) - u*(b*v + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*x*np.cos(theta) + L**(0.5)*( -c*v + b*w - w*y + v*z )*np.sin(theta)

    # y-transformation
    rotated[1] = ( b * ( u**2 + w**2 ) - v*(a*u + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*y*np.cos(theta) + L**(0.5)*(  c*u - a*w + w*x - u*z )*np.sin(theta)

    # z-transformation
    rotated[2] = ( c * ( u**2 + v**2 ) - w*(a*u + b*v - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*z*np.cos(theta) + L**(0.5)*( -b*u + a*v - v*x + u*y )*np.sin(theta)

    rotated = rotated/L
    return rotated
