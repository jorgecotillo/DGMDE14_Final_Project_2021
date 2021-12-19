## Final Project

# Team members
* Peter Chen
* Jorge Cotillo
* Jeremy Guo
* Sanjeev Nandal

# Final presentations video links
  * Report
    * [Project report](https://1drv.ms/w/s!Aqk-ltMQf5hamz8kTLw6veyveFsX?e=GWdt1v)
  * Presentation
    * [Recorded presentation](https://1drv.ms/v/s!Aqk-ltMQf5hamBquFCyuObmvhIE2?e=UlOe5a)
    * [PowerPoint presentation](https://1drv.ms/p/s!Aqk-ltMQf5hamBT8sNHEeF-S01kK?e=YmXmHN)
  * Demo
    * CNN
      * [CNN Demo](https://1drv.ms/v/s!Aqk-ltMQf5hamzSLoh1ttGQgzS0m?e=ym3bjP)
    * Landmark
      * [Landmark extraction](https://1drv.ms/v/s!Aqk-ltMQf5hamBsMahnw_wUIq2qx?e=eLRMWi)
      * [Monitoring website](https://1drv.ms/v/s!Aqk-ltMQf5hamzN37OUA7HWE-FtC?e=D1ph3b)

# Important links

* Data set:
  * University of Texas Arlington https://sites.google.com/view/utarldd/home

* Baseline code:
  * https://www.kaggle.com/aadityasinghal/facial-expression-recognization-using-tensorflow

* For image capture videostream
  * `apt-get install cmake`

* Tensorflow installation on rpi 4:
  * https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html

* For playsound to work, run:
  * `pip install pygame`
  * `sudo apt-get install libsdl2-mixer-2.0-0`

# Acknowledgement
* The landmark method uses the calculation covered in Soukupová and Čech’s paper in 2016 titled” Real-Time Eye Blink Detection Using Facial Landmarks” to calculate the eye aspect ratio at each frame captured by the Raspberry Pi camera 
* The landmark method followed the same structure of code with Adrian Rosebrock’s blog named “Drowsiness detection with OpenCV” to detect micro-sleeps

* The data set for drowsiness data is from the University of Texas Arlington dataset titled "UTA Real-life Drowsiness Dataset." This dataset forms the basis for the training and validation data sets used in the CNN model.

* CNN model architecture and code is based on by Analytics Vidhya, 2021. "Facia expression detection using Machine Learning in Python."

# References
Analytics Vidhya, 2021. Facial expression detection using Machine Learning in Python. Available at: <https://medium.com/analytics-vidhya/facial-expression-detection-using-machine-learning-in-python-c6a188ac765f> [Accessed 19 December 2021].


Ghoddoosian, R., Galib, M. and Athitsos, V., 2019. A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection. [online] arXiv.org. Available at: <https://arxiv.org/abs/1904.07312> [Accessed 18 December 2021].

Rosebrock, A., 2017. Drowsiness detection with OpenCV - PyImageSearch. [online] PyImageSearch. Available at: <https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/> [Accessed 19 December 2021].





