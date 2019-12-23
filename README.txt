Samples
=======
* Samples are generated from breast tissue biopsy slides,
stained with hematoxylin and eosin (HE).
* prepared for histological study and labelled by pathologists of the P&D Lab
* breast tumor specimens assessed by Immunohistochemistry (IHC)
* Core Needle Biopsy (CNB) and Surgical Open Biopsy (SOB)
* section of ~3µm thickness

Image acquisition
=================
* Olympus BX-50 system microscope with a relay lens with magnification of 3.3× coupled to a Samsung digital color camera SCC-131AN
* magnification 40×, 100×, 200×, and 400× (objective lens 4×, 10×, 20×, and 40× with ocular lens 10×)
* camera pixel size 6.5 µm
* raw images without normalization nor color color standardization
* resulting images saved in 3-channel RGB, 8-bit depth in each channel, PNG format


Format of image filename
========================

   <BIOPSY_PROCEDURE>_<TUMOR_CLASS>_<TUMOR_TYPE>_<YEAR>-<SLIDE_ID>-<MAGNIFICATION>-<SEQ>

   <BIOPSY_PROCEDURE>::=CNB|SOB
   <TUMOR_CLASS>::=M|B
   <TUMOR_TYPE>::=<BENIGN_TYPE>|<MALIGNANT_TYPE>
   <BENIGN_TYPE>::=A|F|PT|TA
   <MALIGNANT_TYPE>::=DC|LC|MC|PC
   <YEAR>::=<DIGIT><DIGIT>
   <SLIDE_ID>::=<NUMBER><SECTION>
   <SEQ>::=<NUMBER>
   <MAGNIFICATION>::=40|100|200|400

   <NUMBER>::=<NUMBER><DIGIT>|<DIGIT>
   <SECTION>::=<SECTION><LETTER>|<LETTER>	
   <DIGIT>::=0|1|...|9
   <LETTER>::=A|B|...|Z 

* where
   CNB = Core Needle Biopsy (For future use)
   SOB = Surgical Open Biopsy
		
   B  = Benign
        A = Adenosis
   	F = Fibroadenoma
        TA = Tubular Adenoma
        PT = Phyllodes Tumor

   M  = Malignant
	DC = Ductal Carcinoma
        LC = Lobular Carcinoma
        MC = Mucinous Carcinoma (Colloid)
        PC = Papillary Carcinoma
