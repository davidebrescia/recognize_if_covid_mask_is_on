# No-mask detector

Let's classify images. 
Here's the problem: a picture depict a certain number of covid-masked people (maybe zero). Our task is discriminating the images depending on the following cases: 
1) All the people in the image are wearing a mask 
2) No person in the image is wearing a mask
3) Someone in the image is not wearing a mask

Example of the 1st, 2nd and 3rd class:

![masks](https://user-images.githubusercontent.com/92381157/137192244-67ab0690-d9c3-435b-a11a-9eb6ab564cfa.png)

A Convolutional Neural Network has been trained to classify the image among the three classes.
For more details, please read "Report.pdf" and the Python notebooks. For the dataset, contact me.

## Results
The best model has an accuracy on the test dataset equal to 93.8%.

