Dataset -  BreakHis

Files and uses
model.py - Txt file that has the model . One can train the model with the dataset mentioned to get an accuracy of 90.4%. 
Model is a tinny version of VGG16 that is tuned to get the best possible accuracy.

Augmentation.py - Randomly augments images to equalise the number of images in each category.

app.py - A sample flask app to deploy the model on the localhost with web as it's front end

others - Related to the flask file

To do:
- Run the augmentation.py to generate augmented images
- Merge the augmented images into the original image folders
- Train the model.py with the data set
- Run the app.py file
- Run the local host to find the prediction of an uploaded image



Contributions are welcome!.

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
