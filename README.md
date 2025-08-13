# Aditya_L1_SWIS_processing


This is a simple prediction model of CME based on the Aditya L1's SWIS payload data. SWIS measures the Insitu Measurement of particles. It basically measure the particles are passing through that particular area where Aditya L1 is placed.

Brief about idea: 
                SWIS of ASPEX payload of Aditya L1 basically doing the In-Situ measurement, So we have the data of associated charracteristics of corresponding particles. Now the authority mentioned that use CACTUS catalog data to validate the ocurence of CME. CACTUS Catalog contains the LASCO coronagraph data of Sun. When there is any CME LASCO detectes instantly because it is doing remote sensing measurement. LASCO CACTUS catalog contains median velocity of CME. So we can calculate approx time taking to reach ejected CME from lower corona to L1 point(as Aditya- L1 is locating at L1 and SWIS is a IN-Situ instrument). So we can add the travelling time to the corresponding date that we can get the approx observation time CME by SWIS -ASPEX payload. (Here the velocity consider as a constant for simplify the Calculation.) Now based on that Idea we can Label the data that the variation of "Bulk speed in Alpha particle, protons" ; "Alpha and proton paticle density" ; "variation of fluxes" and the other parameters as the detected CME event of the particular date. Remaings are considered as non CME event. For that we need to consider a threshold velocity and threshold angular width from Cactus catalog and threshold in proton and alpha particle from Aditya L1 data. In that way we will be able to detect CMEs using SWIS data. Here velocity parameter that we have taken is constant for simplify the computation.

Steps to make the CME prediction model:
  1. Read the BLK file of SWIS data. Visualize the data structure.(SWIS file has TH1, TH2, BLK file with Level-1, Level-2 of processing. I used Level-2 BLK data)
  2. In a folder, I downloaded Level-2 data of SWIS payload's TH1, TH2 and BLK file. I seperated the BLK files and read this and save it in a csv format.
  3. By web-scarpping I got LASCO coronagraph data from the respective website, and saved as csv format.
  4. Now from the LASCO Coronagraph data I assigned the CMEs based on the proton bulk speed(v>600), for simple approximation
  5. So we get the dates of CMEs, from the LASCO coronagraph. I used simple distance and time formula to reach CMEs to L1 point. 
  6. Based on that date, I labeled the L2 BLK csvs files what are falls under CME and what are not falls under the CME.
  7. Now the problem falles under classification problem, I used Random forest to train the model and get prediction from it.
