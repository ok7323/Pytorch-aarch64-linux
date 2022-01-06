# Pytorch mnist application

# Mnist inference

Ex)
python3 inference.py

# Test Result

* True inference sample images  
  + Result: target "7" -> output "7" nbsp file_path: ./images/true_sample/0.jpg  
  + Result: target "2" -> output "2" nbsp file_path: ./images/true_sample/1.jpg  
  + Result: target "1" -> output "1" nbsp file_path: ./images/true_sample/2.jpg  
  + Result: target "0" -> output "0" nbsp file_path: ./images/true_sample/3.jpg  
  + Result: target "4" -> output "4" nbsp file_path: ./images/true_sample/4.jpg  
  
* False inference sample images  
  + Result: target "4" -> output "6" nbsp file_path: ./images/false_sample/247.jpg  
  + Result: target "6" -> output "0" nbsp file_path: ./images/false_sample/259.jpg  
  + Result: target "2" -> output "7" nbsp file_path: ./images/false_sample/321.jpg  
  + Result: target "9" -> output "4" nbsp file_path: ./images/false_sample/359.jpg  
  + Result: target "6" -> output "0" nbsp file_path: ./images/false_sample/445.jpg  

* Number result  
  + Number of 0 images: 980 EA nbsp Accuracy: 99.49 %  
  + Number of 1 images: 1135 EA nbsp Accuracy: 99.56 %  
  + Number of 2 images: 1032 EA nbsp Accuracy: 97.58 %  
  + Number of 3 images: 1010 EA nbsp Accuracy: 98.12 %  
  + Number of 4 images: 982 EA nbsp Accuracy: 98.57 %  
  + Number of 5 images: 892 EA nbsp Accuracy: 98.77 %  
  + Number of 6 images: 958 EA nbsp Accuracy: 98.54 %  
  + Number of 7 images: 1028 EA nbsp Accuracy: 98.15 %  
  + Number of 8 images: 974 EA nbsp Accuracy: 97.74 %  
  + Number of 9 images: 1009 EA nbsp Accuracy: 96.83 % 

* Total result 
  + Number of total images: 10000 EA  
  + Average loss: 0.0492 nbsp Accuracy: 98.34 %  
