# Deploy-historocal_places_InGradio

This repository was a collaboration between me and Albaraa shehada as a team work project required from the Deep learning training course.
The main idea for this project was building AI model able to predict in 6 historical places in Jordan: Ajlun castle, Jarash, Petra, Um Qais, Wadi Rum and Roman theater(Amman). Then use this model to make a web Application to predict of this places from the images using Hugging face space and gradio code. 
In this project we applied this steps:
1. problem understand.
2. Data collection: data collected from scratch using: python code using bing_downloader library, Image extraxtor web Application to extract images from URL, and finally from Google images we collect more then 5000 images, we used all data to train on VGG16 and Inceptionv3: you can find it in this link: [(https://www.kaggle.com/datasets/ishraqaldagamseh/last-alldata)](https://www.kaggle.com/datasets/ishraqaldagamseh/last-alldata)  Resnet50 and EfficientB0 using the same data but splitted peviously and you can find it in this link: https://www.kaggle.com/datasets/ishraqaldagamseh/hist-train-test.
3. Modeling using 4 pre-trined models: Resnet50, EfficientB0, VGG16 and InceptionV3.
4. Evaluate the model performance using learning curves.
5. Build gradio App using Hugging face space.
6. Now the App is Ready to use.
   ![image](https://github.com/ishraq-dagamseh/Deploy-historocal_places_InGradioo/assets/16488773/24ba0417-8799-4182-806e-d8dd22ed7600)

8. You can try the App from this link: https://huggingface.co/spaces/IshraqTariq92/Smart_Tour_Guide
9. And the final presentation available on: https://prezi.com/view/qp8u0XCsUfixMslk5ws6/
   

