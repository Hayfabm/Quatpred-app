# Quatpred-app
Protein quaternary structure prediction 

## Quaternary structure predictor app 
The biological functions of proteins are ultimately related to their structures. According to their structural hierarchy, proteins are generally classified into four classes: primary, secondary, tertiary, and quaternary. Proteins with a quaternary structure are referred to as oligomeric proteins. They can be divided into two classes: homo-oligomers and hetero-oligomers; the former is composed of identical subunits while the latter is composed of non-identical subunits. Thus, it is highly desirable to develop an app to automatically classify the quaternary structure attributes of proteins from their sequences. 
Given a protein with L residues, the corresponding dipeptide composition (__DPC__), pseudo amino acid composition (__PseAAC__), and conjoint triad (__Ctriad__) features can be represented as three types of vectors with dimensionalities of 400, 23, and 343 respectively. These vectors can then be serially combined to form a final vector with the dimensionality of 766 as the input to the machine-learning model. In view of the diversity of protein attributes, using the 10-Fold Cross-Validation, our model outperforms __94.04%__ Accuracy, __72.68%__  sensitivity, __92.36%__  Specificity, and __78.65__ % MCC in an independent test set. 

![QS-drawio.png](https://i.postimg.cc/VvdVTzrJ/QS-drawio.png)
