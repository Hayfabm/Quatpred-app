import math
import re
import string
from collections import Counter
from typing import List, Tuple

import pandas as pd

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

# Amino Acid composition 

def calculate_aa_composition(protein_sequence):
	"""
	Calculate the composition of Amino acids for a given protein sequence.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing the composition of 20 amino acids.
	"""

	length_sequence=len(protein_sequence)
	result={}
	for i in AALetter:
		result[i] = round(float(protein_sequence.count(i)) / length_sequence * 100, 3)
	return result


# Dipeptide composition

def calculate_dipeptide_composition(protein_sequence):
	"""
	Calculate the composition of dipeptidefor a given protein sequence.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing the composition of 400 dipeptides.
	"""
	length_sequence=len(protein_sequence)
	result={}
	for i in AALetter:
		for j in AALetter:
			dipeptide=i+j
			result[dipeptide]=round(float(protein_sequence.count(dipeptide)) / (length_sequence - 1) * 100, 2)
	return result

def DPC_processing(sequences):
	code=[]
	for i in sequences:
		DPC= calculate_dipeptide_composition(i)
		dp=(list(DPC.values()))
		code.append(dp)

	return code

# tripeptide composition 

def getkmers():
	"""
	Get the amino acid list of 3-mers.
	:return: result is a list form containing 8000 tri-peptides.
	"""
	kmers = list()
	for i in AALetter:
		for j in AALetter:
			for k in AALetter:
				kmers.append(i+j+k)
	return kmers




def get_spectrum_dict(protein_sequence):
	"""
	Calculate the spectrum of 3-mers for a given protein.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing the composition values of 8000
	"""

	result = {}
	kmers = getkmers()
	for i in kmers:
		result[i]=len(re.findall(i, protein_sequence))
	return result



def calculate_aa_tripeptide_composition(protein_sequence):
	"""
	Calculate the composition of AADs, dipeptide and 3-mers for a given protein sequence.
	:param protein_sequence: protein is a pure protein sequence.
	:return: result is a dict form containing all composition values of AADs, dipeptide and 3-mers (8420).
	"""
	result={}
	result.update(calculate_aa_composition(protein_sequence))
	result.update(calculate_dipeptide_composition(protein_sequence))
	result.update(get_spectrum_dict(protein_sequence))

	return result


# Pseudoamino acid composition 

AALetter = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

_Hydrophobicity = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74, "G": 0.48,
                   "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18,
                   "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}

_hydrophilicity = {"A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0, "G": 0.0, "H": -0.5,
                   "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4,
                   "Y": -2.3, "V": -1.5}



def _mean(listvalue):
    
    return sum(listvalue) / len(listvalue)



def _std(listvalue, ddof=1):
    
    mean = _mean(listvalue)
    temp = [math.pow(i - mean, 2) for i in listvalue]
    res = math.sqrt(sum(temp) / (len(listvalue) - ddof))
    return res




def normalize_each_aap(aap):
   
    if len(list(aap.values())) != 20:
        print('You can not input the correct number of properities of Amino acids!')
    else:
        Result = {}
        for i, j in list(aap.items()):
            Result[i] = (j - _mean(list(aap.values()))) / _std(list(aap.values()), ddof=0)

        return Result



def _get_correlation_function(Ri='S', Rj='D', aap=[_Hydrophobicity, _hydrophilicity]):
    

    hydrophobicity = normalize_each_aap(aap[0])
    hydrophilicity = normalize_each_aap(aap[1])
    theta1 = math.pow(hydrophobicity[Ri] - hydrophobicity[Rj], 2)
    theta2 = math.pow(hydrophilicity[Ri] - hydrophilicity[Rj], 2)
    theta = round((theta1 + theta2) / 2.0, 2)
    return theta


def _get_sequence_order_correlation_factor(protein_sequence, k=1):
   
    length_sequence = len(protein_sequence)
    res = []
    for i in range(length_sequence - k):
        aa1 = protein_sequence[i]
        aa2 = protein_sequence[i + k]
        res.append(_get_correlation_function(aa1, aa2))
    result = round(sum(res) / (length_sequence - k), 3)
    return result


def get_aa_composition(protein_sequence):
    
    length_sequence = len(protein_sequence)
    result = {}
    for i in AALetter:
        result[i] = round(float(protein_sequence.count(i)) / length_sequence * 100, 3)
    return result


def _get_pseudo_aac1(protein_sequence, lamda=10, weight=0.05):
   
    rightpart = 0.0
    for i in range(lamda):
        rightpart = rightpart + _get_sequence_order_correlation_factor(protein_sequence, k=i + 1)
    aac = get_aa_composition(protein_sequence)

    result = {}
    temp = 1 + weight * rightpart
    for index, i in enumerate(AALetter):
        result['PAAC' + str(index + 1)] = round(aac[i] / temp, 3)

    return result


def _get_pseudo_aac2(protein_sequence, lamda=10, weight=0.05):
   
    rightpart = []
    for i in range(lamda):
        rightpart.append(_get_sequence_order_correlation_factor(protein_sequence, k=i + 1))

    result = {}
    temp = 1 + weight * sum(rightpart)
    for index in range(20, 20 + lamda):
        result['PAAC' + str(index + 1)] = round(weight * rightpart[index - 20] / temp * 100, 3)

    return result


def _get_pseudo_aac(ProteinSequence, lamda=10, weight=0.05):
    
    res = {}
    res.update(_get_pseudo_aac1(ProteinSequence, lamda=lamda, weight=weight))
    res.update(_get_pseudo_aac2(ProteinSequence, lamda=lamda, weight=weight))
    return res


# Conjoint-triad

AALetter = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

_repmat = {
    1: ["A", "G", "V"],
    2: ["I", "L", "F", "P"],
    3: ["Y", "M", "T", "S"],
    4: ["H", "N", "Q", "W"],
    5: ["R", "K"],
    6: ["D", "E"],
    7: ["C"],
}


def _Str2Num(proteinsequence):
    """
    translate the amino acid letter into the corresponding class based on the
    given form.
    """
    repmat = {}
    for i in _repmat:
        for j in _repmat[i]:
            repmat[j] = i

    res = proteinsequence
    for i in repmat:
        res = res.replace(i, str(repmat[i]))
    return res


def CalculateConjointTriad(proteinsequence):
    """
    Calculate the conjoint triad features from protein sequence.
    Useage:
    res = CalculateConjointTriad(protein)
    Input: protein is a pure protein sequence.
    Output is a dict form containing all 343 conjoint triad features.
    """
    res = {}
    proteinnum = _Str2Num(proteinsequence)
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                temp = str(i) + str(j) + str(k)
                res[temp] = proteinnum.count(temp)
    return res

def CT_processing(sequences):
	code=[]
	for i in sequences:
		DPC= CalculateConjointTriad(i)
		ct=(list(DPC.values()))
		code.append(ct)

	return code
