# Test steps
## Conversation1     Transform the data from sensor then train the model and reload the model 
###     step1:
Download the ipython file from : https://github.com/wk821917/wktest/blob/master/MY_IBM_TEST_3__r1ge3tK7X.ipynb
     
    Terminal: wget https://github.com/wk821917/wktest/blob/master/MY_IBM_TEST_3__r1ge3tK7X.ipynb


###     step2
Click 'Assets' in this page of this project then click 'New notebook'.

###     step3:
   Click 'From file' and choose the ipython file download before and create the notebook, wait the notebook start

###     step4:
   Click double-arrows arrowhead to restart the kernel and run all cells

###     step5:
   Repeating the cell 'client.training.get_status( run_uid )' can monitor the ML status

###     step6:
   The results can be find in the COS named 'wktest2' with the train-id(run_uid) 

###     Annotation:
If the cells runs fail ,maybe the ML service out of range(50 unit train hour),it can be solved by delete the Ml sevice and apply for a new one

###     step7:
Download the ipython file from : https://github.com/wk821917/wktest/blob/master/predict-test.ipynb
     
    Terminal: wget https://github.com/wk821917/wktest/blob/master/predict-test.ipynb

###     step8:
Click 'Assets' in this page of this project then click 'New notebook'.

###     step9:
   Click 'From file' and choose the ipython file download before and create the notebook, wait the notebook start

###     step10:
   Click double-arrows arrowhead to restart the kernel and run all cells

###     step11:
   Repeating the cell 'client.training.get_status( run_uid )' can monitor the ML status

###     step12:
   The results can be find in the COS named 'predict' with the train-id(run_uid) 


## Conversation2     Cluster by scikitlearn
### step1
Download the ipython file from : https://github.com/wk821917/wktest/blob/master/IBM_ML_KMeans_Test.ipynb
     
    Terminal: wget https://github.com/wk821917/wktest/blob/master/IBM_ML_KMeans_Test.ipynb


###     step2
Click 'Assets' in this page of this project then click 'New notebook'.

###     step3
Click 'From file' and choose the ipython file download before and create the notebook, wait the notebook start

###     step4
Click double-arrows arrowhead to restart the kernel and run all cells

###   step5
Find the result in the COS 'wktest3'

## Conversation3    Classifier by scikitlearn
### step1
Download the ipython file from :https://github.com/wk821917/wktest/blob/master/classifier_test.ipynb
     
    Terminal: wget https://github.com/wk821917/wktest/blob/master/classifier_test.ipynb

###     step2
Click 'Assets' in this page of this project then click 'New notebook'.

###     step3
Click 'From file' and choose the ipython file download before and create the notebook, wait the notebook start

###     step4
Click double-arrows arrowhead to restart the kernel and run all cells

###   step5
Find the result in the COS 'wktest4'

