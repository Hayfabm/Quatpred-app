"""
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from deepchain.components import DeepChainApp
from features_extraction import (CalculateConjointTriad,
                                 calculate_dipeptide_composition)
from protlearn.features import aaindex1, paac
from tensorflow.keras.models import load_model

Score = Dict[str, float]
ScoreList = List[Score]




class App(DeepChainApp):
    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.num_gpus = 0 if device == "cpu" else 1
        self.NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
        self.max_seq_length = 200

        # NOTE: if you have issues at this step, please use h5py 2.10.0
        # by running the following command: pip install h5py==2.10.0

        self.model = load_model("model_checkpoint/model_20220126130246.hdf5")
        print(self.model.summary()) 

    @staticmethod
    def score_names() -> List[str]:
        return ["Quaternary_structure_prediction"]

    def compute_scores(self, sequences_list: List[str]) -> ScoreList:
        scores_list = []

        for sequence in sequences_list:

            # PseAAC 
            seqs = sequence
            paac_comp, desc = paac(seqs, lambda_=3, remove_zero_cols=True)
            
            # AAindex 
            seqs = sequence
            aaind, inds = aaindex1(seqs, standardize='zscore')
            
            #dipeptide
            code = []
            def DPC(seq):
                dip = calculate_dipeptide_composition(seq)
                dp=((list(dip.values())))
                code.append(dp)
                return code

            dip = DPC(sequence)
            dipeptide = np.array(dip)
           
            # Ctriad
            Ctriad = []
            def CT(seq):
                ctriad = CalculateConjointTriad(seq)
                ct=((list(ctriad.values())))
                Ctriad.append(ct)
                return Ctriad

            ctd = CT(sequence)
            conjoint_triad = np.array(ctd)
           

            # Dimension concatenation
            concat= np.concatenate(
                (
                
                dipeptide,
                paac_comp,
                conjoint_triad,
                aaind
                ), axis= 1
            )
            print(concat.shape)

            # forward pass throught the model
            model_output = self.model.predict(concat)
            score = np.argmax(model_output)
            print(score)
            scores_list.append(
                {self.score_names()[0]: model_output[0]}
            )  # model_output[1]: # Quaternary structure prediction 



        return scores_list
    
    


if __name__ == "__main__":
    sequences = ["MSDETNAAVWDDSLLVKTYDESVGLAREALARRLADSTNKREEENAAAAEEEAGEISATGGATSPEPVSFKVGDYARATYVDGVDYEGAVVSINEEKGTCVLRYLGYENEQEVLLVDLLPSWGKRVRREQFLIAKKDEDEQLSRPKASAGSHSKTPKSSRRSRISGGLVMPPMPPVPPMIVGQGDGAEQDFVAMLTAWYMSGYYTGLYQGKKEASTTSGKKKTPKK", #homodimer
                "MKFLLTTAFLILISLWVEEAYSKEKSSKKGKGKKKQYLCPSQQSAEDLARVPANSTSNILNRLLVSYDPRIRPNFKGIPVDVVVNIFINSFGSIQETTMDYRVNIFLRQKWNDPRLKLPDFRGSDALTVDPTMYKCLWKPDLFFANEKSANFHDVTQENILLFIFRDGDVLVSMRLSITLSCPLDLTLFPMDTQRCKMQLESFGYTTDDLRFIWQSGDPVQLEKIALPQFDIKKEDIEYGNCTKYYKGTGYYTCVEVIFTLRRQVGFYMMGVYAPTLLIVVLSWLSFWINPDASAARVPLGIFSVLSLASECTTLAAELPKVSYVKALDVWLIACLLFGFASLVEYAVVQVMLNNPKRVEAEKARIAKAEQADGKGGNVAKKNTVNGTGTPVHISTLQVGETRCKKVCTSKSDLRSNDFSIVGSLPRDFELSNYDCYGKPIEVNNGLGKSQAKNNKKPPPAKPVIPTAAKRIDLYARALFPFCFLFFNVIYWSIYL", #heteropentamer
                "MTFLLVSLLAFLSLGSGCHHRICHCWHRVFLCQESKVTEIPSDLPRNAVELRFVLTKLRVIPKGAFSGFGDLEKIEISQNDVLEVIEANVFFNLSKLHEIRIEKANNLLYIDTDAFQNLPNLRYLLISNTGIKHFPAVHKIQSLQKVLLDIQDNINIHTVERNSFMGLSFESMILWLNKNGIQEIHNCAFNGTQLDELNLSDNINLEELPNDVFQGASGPVILDISRTRIHSLPSYGLENLKKLRAKSTYNLKKLPSLDKFVALMEASLTYPSHCCAFANWRRPISELHPICNKSILRQVDDMTQARGQRVSLAEDEESSYTKGFDMMYSEFDYDLCNEVVDVTCSPKPDAFNPCEDIMGYDILRVLIWFISILAITGNIIVLMILITSQYKLTVPRFLMCNLAFADLCIGIYLLLIASVDIYTKSQYHNYAIDWQTGAGCDAAGFFTVFASELSVYTLTVITLERWHTITHAMQLECKVQLRHAAIIMLLGWIFAFMVALFPIFGISSYMKVSICLPMDIDSPLSQLYVMSLLVLNVLAFVVICCCYAHIYLTVRNPNIVSSSSDTKIAKRMAMLIFTDFLCMAPISFFAISASLKVPLITVSKSKILLVLFYPINSCANPFLYAIFTKNFRRDFFILMSRFGCYEMQAQTYRTETLSTAHNSHPRNGHCPPALRVTNYTLVPLTHLAQN", #homotrimer
                "MLSRCRSRLLHVLGLSFLLQTRRPILLCSPRLMKPLVVFVLGGPGAGKGTQCARIVEKYGYTHLSAGELLRDERKNPDSQYGELIEKYIKEGKIVPVEITISLLKREMDQTMAANAQRNKFLIDGFPRNQDNLQGWNKTMDGKADVSFVLFFDCNNEICIERCLERGKSSGRSDDNRESLEKRIQTYLQSTKPIIDLYEEMGKVKKIDASKSVDEVFDEVVQIFDKEG", #monomer
                "MATVPELDSEMIAFHSDENDLFFEVDGLQKMKSCFQSLDLSYPDESIQLQISKQDLNKSFRQVVSVIVAVEKLWNTPVPCPWTFQDEDLRTFFSFIFEEEPIFCDSWDGELIVADAPIRQLHCRLRDEQQKCLVLSDPCELKALHLNGQNINQQVVFSMSFVQGETSNNKIPVALGLKGKNLYLSCVMKGDTPTLQLESVDPKQYPKKKMEKRFVFNKIEVKTKVEFESAQFPNWYISTSQAEHKPVFLGNNSGQDLVDFTMESVSS" #monomer
    ]
    app = App("cpu")

    scores = app.compute_scores(sequences)
    print(scores)




