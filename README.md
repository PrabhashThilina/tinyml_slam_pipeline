
Project Structure
This project consists of
- Virtual environment with all the dependencies.
- Set of full-precision pre-trained Python models.
- Set of quantized pre-trained Python models.
- Python scripts to run the models
- Some other utility classes

How to Run the Project
step 1 - Activate virual environment
           tinyml_slam_env sourc/bin/activate
step 2 - Run run_slam_pipeline.py
           python3 run_slam_pipeline.py
         OR
Alternatively : Run each model seperately   
                1. Run run_superpoint.py
                    python3 run_superpoint.py
                2. Run run_depth_estimation.py
                    python3 run_depth_estimation
                3. Run run_segmentation.py
                    python3 run_segmentation.py
                4. Run run_slam_back_end.py
                    python3 run_slam_back_end.py

Files :
superpoint.py - SuperPoint architecture
run_superpoint.py - Run SuperPoint for key point identificaiton
run_depth_estimation.py - Run MiDas for depth map cration
run_segmentaion.py - Run MobileNetV2 to remove dynamic objects
run_slam_back_end.py - Run SLAM back end (Pnt + RANSAC) and plot trajectory
image_pre_processor.py - Pre-process image data by reducing resolution and image count
compare_models.py - Compare memory, inference speed, throughput of two models

superpointv1.pth - Full precision SuperPoint model
superpoint_static_int8.pth - INT8 quantized SuperPoint model
midas_small.pth - Full precision Midas model
midas_small_dynami_quanted.onnx - INT8 dynamic quantied ONNX converted MiDas model 


Dataset Link
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg2_pioneer_slam3



Authors

Thilina Prabhash | GitHub: PrabhashThilina
