
## **Manual to Run Kaggle Competition Code on Scholar Cluster**
Follow these steps to set up and run on the cluster the code provided for the Super Resolution project.
---
Steps 1 through 8 are based on the instructions in your syllabus on how to use the cluster. If you would like more details, please refer to that document (https://www.cs.purdue.edu/homes/ribeirob/courses/Fall2025/howto/cluster-how-to.html).
### **Part 1: Set Up the Environment**
#### 1. **Access the Scholar Cluster**
Log into the scholar cluster via SSH (see more info on the doc linked at the top).
#### 2. **Load Anaconda**
Write this in the terminal, then press enter:
```bash
$ module load conda/2024.09
```
#### 3. **Create the Conda Environment**
Write this in the terminal, then press enter:
```bash
$ conda create -n CS373SuperRes python=3.11 ipython ipykernel
```
Please check both job submission scripts so that the conda environment name is exactly the same as the one that you've just created.
#### 4. **Accept Installations**
If prompted to confirm, type **yes** to accept, then press enter.
#### 5. **Activate the New Environment**
Write this in the terminal, then press enter:
```bash
$ source activate CS373SuperRes
```
#### 6.	**Install Necessary Libraries**
Add the required Python libraries for image processing and deep learning. Write this in the terminal, then press enter:
```bash
$ pip install matplotlib numpy pandas opencv-python pillow tqdm
```
#### 7. **Accept Installations**
If prompted to confirm, type **yes** to accept, then press enter.
#### 8.	**Install PyTorch and Vision Libraries**
Write this in the terminal, then press enter:
```bash
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 9.	**Log Out of the Cluster**
Once the installation is complete, log out of the cluster. To do that, write this in the terminal, then press enter:
```bash
$ exit
```
#### 10. **Re-enter the Scholar Cluster**
Log in again to finalize the setup and make sure you can activate your environment 
```bash
$ module load conda/2024.09
$ source activate CS373SuperRes
```

---
### **Part 2: Prepare Files and Directory**
#### 11. **Create a Working Folder**
Create a folder to store the files for the project. For example, a folder named SuperResolutionProject. To do that, write this in the terminal, then press enter:
```bash
$ cd /scratch/scholar/$USER/
$ mkdir /scratch/scholar/$USER/SuperResolutionProject
```
#### 12. **Transfer local files to Scholar**
You have mainly 2 options: Jupyter Hub (recommended) or the scp command. Let's see how to move a file to scholar using both methods:
#### Option 1: Jupyter Hub
1. Go to (https://www.rcac.purdue.edu/compute/scholar)
2. Click on launch in the Jupyter Hub section
3. Log in:
    - Username: your Purdue username (generally the first part of your email, like: "USERNAME@purdue.edu")
    - Password: **MyPassword,BoilerKey**, where "MyPassword" should be the password used to access websites like mypurdue, and the boilerkey is the code that you find in the Duo app.
4. Click on the folder we **created earlier on the cluster**
5. Click upload
6. Select all the files that are in the zip provided (please unziped before selection the files) and click open
7. Click ok on all the warnings
8. Click upload on all file
9. Wait for all files to upload
10. At the end it should look like this
#### Option 2: SCP
On MacOS and Linux open your terminal, while on Windows you can use either the command line or powershell. Then type this:
```bash
scp PATH/TO/YOUR/FILE USERNAME@scholar.rcac.purdue.edu:/scratch/scholar/$USER/SuperResolutionProject
```
Where
- PATH/TO/YOUR/FILE: absolute local path of the file that you want to move
- USERNAME: your Purdue username (generally the first part of your email, like: "USERNAME@purdue.edu")
For example: ```scp ~/Download/SuperResolution.zip <yourusername>@scholar.rcac.purdue.edu:/scratch/scholar/$USER/SuperResolutionProject/```
You're going to login again, and the password like for the Jupyter Hub section should be this:
- Password: **MyPassword,BoilerKey**, where "MyPassword" should be the password used to access websites like mypurdue, and the boilerkey is the code that you find in the Duo app.
After successfully completing the login the upload will start. Please be aware that it may take a lot of time if you're moving more than 100 MB.
#### 13. **Check the Files were Correctly Transfer**
Now all the files in the provided zip should be in the SuperResolutionProject folder on the scholar cluster. To double check, write this in the terminal, then press enter:
```bash
$ cd /scratch/scholar/$USER/SuperResolutionProject/
$ ls
```
---
### **Part 3: Run the Training Code**
#### 14. **Submit Training Code**
Create your code and run the training script by summiting a slurm job by simply writing on the terminal (and then pressing enter) this: 
```bash
$ sbatch ./submit_job_train.sh
```
When the job is submitted you should see on the terminal an id for the job you just created.
#### 15. **Check Job Status**
Your job should be on the cluster queue, to check this use.
```bash
$ squeue -u $USER
```
#### 16. **Wait for Completion**
Training a super resolution model typically takes around 1-2 hours depending on the architecture and dataset size.
#### 17. **Check for Errors**
After the job is finished you can open the error log to confirm no errors occur (warnings are okay). Simply type and the press enter in the terminal this:
```bash
$ nano SuperResTrain.err
```
#### 18. **View Training Output**
Check the output file to ensure the training metrics were printed, meaning that the training process was completed successfully. On the temrinal do this:
```bash
$ nano SuperResTrain.out
```
#### 19. **Confirm Model File Creation**
Check that the trained model was saved, if it was a new file called "**super_resolution_model.pth**" should appear in the working folder. To check it, do this:
```bash
$ ls
```
---
### **Part 4: Run the Testing Code**
#### 20. **Submit Testing Job**
Run the test code by summiting a slurm job, just as we did before.
```bash
$ sbatch ./submit_job_test.sh
```
#### 21. **Check Job Status**
Your job should be on the cluster queue
```bash
$ squeue -u $USER
```
#### 22. **Wait for Completion**
Testing may take 30-60 minutes depending on the number of test images.
#### 23. **Check for Errors**
After the job is finished open the error log to confirm no errors are present (warnings are okay):
```bash
$ nano SuperResTest.err
```
#### 24. **Check Testing Output**
Check the output file, it should show PSNR and SSIM scores for your reconstructions.
```bash
$ nano SuperResTest.out
```
#### 25. **Confirm Submission Files Creation**
Check that the submission files were saved, if they were a new folder called "**reconstructed_images**" should appear in the working folder containing all your high-resolution test images.
```bash
$ ls -l reconstructed_images/
```
#### 26. **Verify File Count**
Ensure that all test images have been processed:
```bash
$ ls reconstructed_images/ | wc -l
```
This number should match the count of low-resolution test images in the `test_x` directory.

---
Follow these steps to complete your model training and testing on the scholar cluster for the Super Resolution task. Be sure to review the code comments thoroughly: they explain each part of the process and will help you understand the workflow. This will make it easier to make modifications that can improve the performance of your super resolution model. Good luck!
