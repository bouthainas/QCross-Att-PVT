# QCross-Att-PVT Lung Analysis Architecture
![ ](https://github.com/bouthainas/QCross-Att-PVT/blob/main/PVT-ViT%20diagram.png)
## What is QCross-Att-PVT?
QCross-Att-PVT is a novel transformer-based architecture tailored for predicting the severity of lung infections from CT and Chest X-ray images. The model utilizes Pyramid Vision Transformers (PVTs) as parallel encoders, designed to capture multi-scale features from different image regions. A key innovation is the cross-gated attention mechanism, which facilitates the interaction between different regions of the image, allowing the model to focus on the most relevant areas for severity assessment. The outputs from the encoders are further processed by a feature aggregator, based on Vision Transformer (ViT) principles, to refine and integrate the extracted information into a single scalar severity score. Validated on the RALO CXR and Per-COVID-19 CT datasets, QCross-Att-PVT demonstrates superior performance offering a reliable and efficient solution for automated lung infection severity quantification.

# Team
## Core Contributors
* Biomedical Engineer, PhD Student, Bouthaina Slika, University of the Basque Country, Spain & Ho Chi Minh Open University, Vietnam bslika001@ikasle.ehu.eus
* Prof. Dr. Fadi Dornaika, IEEE member, Dept. of Artificial Intelligence, University of the Basque Country & IKERBAQUE foundation, Spain fadi.dornaika@ehu.es
* Prof. Dr. Karim Hammoudi, IEEE member, Group Imagery, Dept. of Computer Science, IRIMAS, Université de Haute-Alsace, France, karim.hammoudi@uha.fr
* Dr. Fares Bougourzi, Researcher in Data Science, Dept. of Digital System and Life Science, Junia, UMR 8520, CNRS, France faresbougourzi@gmail.com

## Collaborators
* Prof. Dr. Halim Benhabiles, Group BIO-MEMS, Dept. of Artificial Intelligence, JUNIA, CNRS, IEMN, University of Lille, halim.benhabiles@yncrea.fr
* Prof. Dr. Mahmoud Melkemi, Group Imagery, Dept. of Computer Science, IRIMAS, Université de Haute-Alsace, mahmoud.melkemi@uha.fr
* Prof. Dr. Adnance Cabani, Dept. of Computer Science, ESIGELEC/IRSEEM, Normandy University, adnane.cabani@esigelec.fr

# Reference
> Karim Hammoudi, Halim Benhabiles, Mahmoud Melkemi, Fadi Dornaika, Ignacio Arganda-Carreras, Dominique Collard, and Arnaud Scherpereel. Deep learning on chest x-ray images to detect and evaluate pneumonia cases at the era of covid-19. Journal of medical systems, 45(7):1–10, 2021. https://doi.org/10.1007/s10916-021-01745-4
```
@article{Hammoudi2021,
author={Hammoudi, Karim
and Benhabiles, Halim
and Melkemi, Mahmoud
and Dornaika, Fadi
and Arganda-Carreras, Ignacio
and Collard, Dominique
and Scherpereel, Arnaud},
title={Deep Learning on Chest X-ray Images to Detect and Evaluate Pneumonia Cases at the Era of COVID-19},
journal={Journal of Medical Systems},
year={2021},
month={June},
day={08},
issn={1573-689X},
doi={10.1007/s10916-021-01745-4},
url={https://doi.org/10.1007/s10916-021-01745-4}
}
```
> Bouthaina Slika, Fadi Dornaika, Karim Hammoudi, and Vinh Truong Hoang. Automatic quantification of lung infection severity in chest x-ray images. In 2023 IEEE Statistical Signal Processing (SSP) Workshop, pages 418–422. IEEE, 2023. https://doi.org/10.1109/SSP53291.2023.10207986
```
@inproceedings{slika2023ssp,
title={Automatic Quantification of Lung Infection Severity in Chest X-ray Images},
author={Slika, Bouthaina
and Dornaika, Fadi
and Hammoudi, Karim
and Hoang, Vinh Truong},
booktitle={2023 IEEE Statistical Signal Processing (SSP) Workshop},
pages={418--422},
year={2023},
organization={IEEE},
doi={10.1109/SSP53291.2023.10207986},
url={https://doi.org/10.1109/SSP53291.2023.10207986}
}
```
> Bouthaina Slika, Fadi Dornaika, Hamid Merdji, and Karim Hammoudi.Lung pneumonia severity scoring in chest X-ray images using transformers. In Medical & Biological Engineering & Computing, pages 1-19. Springer, 2024.https://doi.org/10.1007/s11517-024-03066-3
```
@article{slika2024lung,
title={Lung pneumonia severity scoring in chest X-ray images using transformers},
author={Slika, Bouthaina
and Dornaika, Fadi and
Merdji, Hamid and
Hammoudi, Karim},
journal={Medical & Biological Engineering & Computing},
pages={1--19},
year={2024},
publisher={Springer},
doi= {10.1007/s11517-024-03066-3},
url={https://doi.org/10.1007/s11517-024-03066-3}
}
```
> Bouthaina Slika, Fadi Dornaika, and Karim Hammoudi. Multi-Score Prediction for Lung Infection Severity in Chest X-Ray Images. In IEEE Transactions on Emerging Topics in Computational Intelligence, pages 1-7. IEEE,2024.https://doi.org/10.1009/TETCI.2024.3359082
```
@article{slika2024multi,
author={Slika, Bouthaina
and Dornaika, Fadi 
and Hammoudi, Karim},
title={Multi-Score Prediction for Lung Infection Severity in Chest X-Ray Images},
journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
pages={1--7},
year={2024},
month={January},
day={20},
publisher={IEEE},
doi={10.1009/TETCI.2024.3359082},
url={https://doi.org/10.1009/TETCI.2024.3359082}
}
```
![ ](https://github.com/bouthainas/PViTGAtt-IP/blob/main/Affiliations.png)
