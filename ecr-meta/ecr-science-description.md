### Measuring Visual Disorder 
The broken window theory suggests that disordered environments 
are correlated with rule - breaking and social disorder. To aid 
social scientists in studying this hypothesis, I created software
that can measure visual disorder in urban environments. Using 
computer vision techniques, I extracted features (mean and standard
deviation of hue, saturation, and value, entropy, edge density, and 
straight edge density) from images which I then used to create a 
random forest model that can predict the disorder rating of an image 
with 71 percent accuracy. This software was then deployed on 
geographically distributed Sage nodes with camera used to capture 
images that are run through the machine learning model to predict the
disorder of the area captured. My results revealed that mean saturation 
and mean hue were the most important low - level features in determining 
what makes an urban environment disorderly while entropy and standard 
deviation of value were the least important. Most features were found to 
have little correlation with each other.