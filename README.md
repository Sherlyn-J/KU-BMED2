# KU-BMED2

# Explainable AI model to identify biomarkers in Alzheimer’s disease
Authors: Sherlyn Jemimah, Aamna Mohammed AlShehhi* for the Alzheimer’s Disease Neuroimaging Initiative†

We developed a contrained, explainable AI model which utilizes SNPs, gene expression and clinical data from ADNI to predict the disease status of participants, and identify potential blood-based biomarkers for diagnosis. 
Model performance in testing yielded an accuracy of 89.3% and AUC of 97.2%. The model incorporates constraints based on Reactome pathway data, which enhances performance and explainability. We used SHapley Additive exPlanations (SHAP) to identify genes which could potentially serve as biomarkers and gain mechanistic insights into Alzheimer's.

# Data availability
The data that support the findings of this study are available from ADNI, but restrictions apply to the availability of these data, which were used under license for the current study, and so are not publicly available.

\* Corresponding author <br>
† Data used in preparation of this article were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: <a href="http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf"> link </a>


# Reactome pathways
Reactome data was downloaded from the repository provided by Elmarakeby et al. (2021): <a href="https://zenodo.org/record/5163213#.Y7wZgNVBxPY">https://zenodo.org/record/5163213#.Y7wZgNVBxPY</a>

Elmarakeby HA, Hwang J, Arafeh R, Crowdis J, Gang S, Liu D et al. Biologically informed deep neural network for prostate cancer discovery. Nature 2021 Oct;598(7880):348-352. doi: 10.1038/s41586-021-03922-4
