# My Project Portfolio

This portfolio showcases a selection of projects I undertook during my undergraduate studies. The majority of these projects were completed independently, driven by my own curiosity and enthusiasm, without the guidance of a mentor. They span a wide range of subjects, including 3D computer vision, traditional image processing, deep learning, intelligent algorithms, and SLAM.

These experiences, although not leading-edge research, have significantly enriched my academic journey and helped me identify my passions. I firmly believe that this valuable practical experience will greatly assist me in my future scientific research endeavors.

## Internship
* (New💥) A national patent for invention is pending.
* [Auto Keystone Correction Projector with Structured Light Pair](./internship/auto%20keystone%20correction%20projector%20with%20camera/README.md): The calibration process in this study involves local homography and Gray code, while the correction process is achieved by triangulating keypoint depths and subsequently fitting the projection plane. Concurrently, an accelerometer measures the direction of gravity. The correlation between the key points on the wall and those on the projection screen is employed to compute the homography matrix, which is then used to reverse-engineer the actual display area. The primary focus of this research is to obtain the largest and sharpest inner rectangle within an arbitrary convex projected quadrilateral. In the future, an algorithm for automatically avoiding obstacles on the wall will be developed to enhance the user experience.

* [Auto Keystone Correction Projector with TOF](./internship/auto%20keystone%20correction%20projector%20with%20TOF/): Plane detection is made possible using the VL53L5CX multi-point Time-of-Flight (TOF) sensor from STMicroelectronics. Data fluctuations are minimized through a filtering process, and robustness is further improved by introducing the Random Sample Consensus (RANSAC) algorithm. The projection mechanism of ultrashort focus projectors is described using an equivalent ideal pinhole model. In other aspects, this study follows a methodology similar to previous research.

<div align="center">
<img src="./images/projector.jpg" width=45% />
<img src="./images/projector2.jpg" width=35% />
</div>

<div align="center">
<img src="./images/camera1.gif" width=40% />
<img src="./images/tof.gif" width=40% />
</div>

<div align="center">
<img src="./images/calib1.png" width=40% />
<img src="./images/calib2.png" width=40% />
</div>

<div align="center">
<img src="./images/calib3.jpg" width=80% />
</div>


>**Praise from the project manager:**
You have refreshed my attitude towards the post-00s.
Introduce me to more students with good character like you, Zhang. I'll gather you all together as a team, you'll be the head.

<div align="center">
<img src="./images/p.jpg" width=75% />
</div>

## Robotics Team of BUAA

* [Auto-shoot Algorithm Based on Deep Learning for Racing Robot in CURC ROBOCON 2023](./robotics%20team%20of%20BUAA/ROBOCON2023/): Information for target identification and encoding is obtained through the fusion of data from laser radar, wheel odometry, and an Inertial Measurement Unit (IMU) serving as a priori localization. Target identification relies on a well-trained deep learning model with data augmentation. With the integration of localization data, precise angular deviations are computed. These deviations are subsequently sent to the motor driver chip, allowing for precise and automated shooting. Our robot's exceptional performance at the 2023 CURC ROBOCON competition served as a validation of the algorithm's precision and robustness.

[Related Video](https://www.bilibili.com/video/BV1mX4y1Y7Pd/?share_source=copy_web&vd_source=b58b58ccf1b63dc656c22a30535762cc)

<div align="center">
<img src="./images/RC2023_field.jpg" width=45% /><img src="./images/robocon.jpg" width=45% />
</div>     
<div align="center">
<img src="./images/elephant.jpg" width=45% /><img src="./images/rabbit.jpg" width=35% />
</div>       


* [(New💥) Decision-making models for autonomous robots for ROBOCON 2024](./robotics%20team%20of%20BUAA/ROBOCON2024/): The topic for ROBOCON2024 requires autonomous robots to achieve a significant victory in Zone 3. The victory condition is to occupy three granaries. Occupying a granary is achieved when at least two of your team's balls are present in the granary, with your team's ball at the top. This places a high demand on the autonomous decision-making algorithm for robots. To address this, I have designed an algorithm based on the minimax search with alpha-beta pruning, incorporating a simulation interface. What sets this algorithm apart from traditional turn-based game tree approaches is that it allows the robot to choose to skip its own turn and wait for the opponent to act. This approach is more in line with the context of this competition and will give our autonomous robots greater flexibility.

<div align="center">
<img src="./images/RC2024_field.jpg" width=75% />
</div>      

<div align="center">
<img src="./images/RC2024_1.gif" width=45% />
<img src="./images/RC2024_2.gif" width=45% />
</div>      

* Target trajectory analysis with stereo camera : To reduce the computational cost of the deep learning component, a sliding window is introduced, leveraging recognition results from the previous frame. The principles of triangulation are applied to calculate the three-dimensional coordinates of the target. Additionally, Kalman filtering is utilized to enhance data smoothness, predict missing identification information, and bolster overall system robustness.

<div align="center">
<img src="./images/stereo.gif" width=75% />
</div>

* [Team entry test](./robotics%20team%20of%20BUAA/training/README.md): Test I gave to prospective team members. It's a camera pose estimation task. In a scenario with known three-dimensional coordinates, we calculate the camera pose using the Perspective-n-Point (PNP) principle. This involves combining the corner detection results from the previous frame to establish correspondences between 2D points and 3D points. By continuously recognizing these points, we can trace and plot the camera's trajectory. I uploaded the demonstration video to the internet and received widespread attention and discussion.

[Related Video](https://www.bilibili.com/video/BV1TM4y1d7x5/?share_source=copy_web&vd_source=b58b58ccf1b63dc656c22a30535762cc)

<div align="center">
<img src="images/pnp.gif" width=50% />
<img src="images/pnp2.gif" width=30% />
</div>

* Team trainning: This slide serves as a technical guide for new team members. It's designed to instruct newcomers on essential topics, including image processing, 3D vision, as well as providing a brief introduction to Linux and ROS.

<div align="center">
<img src="./images/trainning.jpg" width=55% /><img src="./images/rm.jpg" width=42% />
</div>

## Soft Robotics Lab

* [An Aerial–Aquatic Hitchhiking Robot with Remora-Inspired Tactile Sensors and Thrust Vectoring Units](https://onlinelibrary.wiley.com/doi/10.1002/aisy.202300381): My primary responsibilities include debugging flight control systems, providing assistance with various experiments, and working on the deployment of SLAM (Simultaneous Localization and Mapping) and automatic navigation algorithms for the next generation of robots. Our work is published in Advanced Intelligent Systems.

[Related Video](https://www.bilibili.com/video/BV1g84y1d7YH/?vd_source=8d076f754e2a745bbc3e40e91e1024e0)

<div align="center">
<img src="images/drone1.jpg" width=40% />
<img src="images/drone2.jpg" width=40% />
</div>

<div align="center">
<img src="images/rtabmap.gif" width=40% />
<img src="images/rtabmap2.gif" width=40% />
</div>

<div align="center">
<img src="./images/orb.gif" width=40% />
<img src="./images/vins.gif" width=40% />
</div>

<div align="center">
<img src="./images/fastlio2.gif" width=80% />
</div>

>**Praise from the doctoral students of the research group, after the first group meeting.**
Zhou Hao, I think you're excellent, and you've got a rough idea of our project today. Our team members are all hardworking, and we've been striving to do some interesting and innovative research. If you're interested in this project, I'd like to invite you to join us. Let's work together and aim to publish a high-quality paper.

<div align="center">
<img src="./images/d.jpg" width=75% />
</div>

## AI Program in NUS
* [Seq2Seq population forecasting model](./AI%20program%20in%20NUS/): I led the team members to apply the Seq2Seq model to the population prediction assignment, not only in the basic regression model required by the professor. We won the winning team of the NUS Artificial Intelligence and Machine Learning Summer Course, and praised by Prof. Mehual Motani.

<div align="center">
<img src="images/NUS.jpg" width=55% />
<img src="images/NUS2.jpg" width=25% />
</div>

## Graduation Projects
* [(Pending💥) SLAM and obstacle Avoidance for Uavs based on event cameras](./graduation%20project/)

<div align="center">
<img src="images/gp1.png" width=80% />
</div>
<div align="center">
<img src="images/gp2.png" width=80% />
</div>


## Course Projects
I take every experiment in class seriously, cherish these practical opportunities, and always exceed the teacher's tasks. This seriousness is also reflected in my grades.


* (New💥)PointNet/PointNet++ point cloud segmentation: I led the team to dive into the architecture of PointNet and PointNet++. Through the common BackBone with different heads, the classification and segmentation tasks of point clouds are realized. I deeply explored the properties of T-Net, and tried to change the structure of T-Net, adding residual connections and so on to obtain different performance.

<div align="center">
<img src="images/pn3.png" width=80% />
</div>
<div align="center">
<img src="images/pn1.jpg" width=40% />
<img src="images/pn2.jpg" width=40% />
</div>

* [(New💥) Experiments on Eight-Puzzle graph search algorithms](./in-class%20experiments/8dig): BFS, DFS and A* search algorithms are used to solve the eight-digit game, and the differences in search strategies and performance are explored.

<div align="center">
<img src="images/bfs.jpg" width=40% />
<img src="images/astar.jpg" width=40% />
</div>

* [Comparison experiments between CNN and Dense](./in-class%20experiments/Comparison%20experiments%20between%20CNN%20and%20Dense/README.md): I was inspired by Professor Li Mu's *Drive into Deep Learning* and personally constructed various classic neural networks for the MNIST and Fashion MNIST datasets. I compared their performance and parameter differences. To gain a better understanding of how convolutional neural networks work, I visualized the results of each layer of LeNet. Below are some illustrative figures from my experimental report.

<div align="center">
<img src="images/cnn.png" width=80% />
</div>
<div align="center">
<img src="images/cnn2.png" width=80% />
</div>

* [Experiments on Medical Image segmentation (Liver)](./in-class%20experiments/Experiments%20on%20Medical%20Image%20segmentation/liver/README.md)



* [Experiments on Medical Image segmentation (Retinal vessels)](./in-class%20experiments/Experiments%20on%20Medical%20Image%20segmentation/Retinal%20vessels/README.md)
The above two are medical image segmentation experiments I conducted. I reproduced the classic U-Net using PyTorch and experimented with various hyperparameters to achieve good model training and convergence on a small dataset, resulting in a satisfying outcome. During liver CT segmentation, I noticed the issue of uneven distribution in the original data. After experiencing initial failures, I performed data normalization to overcome the challenge and ultimately achieved successful experimental results.

<div align="center">
<img src="images/liver1.jpg" width=80% />
<img src="images/liver2.jpg" width=80% />
</div>

<div align="center">
<img src="images/retinalvessels2.jpg" width=80% />
<img src="images/retinalvessels1.jpg" width=80% />
</div>


* [EEG-based Motor Imagery Classification](./in-class%20experiments/EEG-based%20Motor%20Imagery%20Classification/README.md): Inferring motor imagery through EEG signals has always been a challenge. I read the paper on EEGnet and implemented it using PyTorch, following its network structure. During the experiment, I gained a deep understanding of the significance of group convolution, depth-wise convolution, and point-wise convolution. In the end, my experimental results ranked at the top of the class for both binary and four-class classification tasks.

<div align="center">
<img src="images/eeg1.png" width=80% />
<img src="images/eeg2.png" width=80% />
</div>

* [Robot path planning experiments](./in-class%20experiments/Robot%20path%20planning%20experiments/README.md)  

 [Related Video](https://www.bilibili.com/video/BV1Us4y1g7rq/?share_source=copy_web&vd_source=b58b58ccf1b63dc656c22a30535762cc): I explored the differences in various heuristic functions in robot path planning tasks and summarized my findings in an experimental report. The visual process of the experiment has been uploaded to Bilibili.

<div align="center">
<img src="images/path.jpg" width=50% />
</div>

* [Recognition of handwritten Arabic characters](./in-class%20experiments/Recognition%20of%20handwritten%20Arabic%20characters/README.md)

<div align="center">
<img src="images/arb.jpg" width=80% />
</div>



* [GMM Built by EM Algorithm](./in-class%20experiments/GMM%20Built%20by%20EM%20Algorithm/)

<div align="center">
<img src="./images/gmm.jpg" width=80% />

</div>
to be continued

# Gallery
<div align="center">
<img src="./gallery/1.jpeg" width=34% />
<img src="./gallery/8.jpg" width=50.7% />
</div>

<div align="center">
<img src="./gallery/2.jpg" width=40% />
<img src="./gallery/3.jpg" width=40% />
</div>



<div align="center">
<img src="./gallery/4.jpg" width=40% />
<img src="./gallery/6.jpeg" width=40% />
</div>

<div align="center">
<img src="./gallery/7.jpg" width=40% />
<img src="./gallery/9.jpg" width=40% />
</div>


<div align="center">
<img src="./gallery/5.jpeg" width=45% />
<img src="./gallery/14.jpeg" width=36.6% />
</div>