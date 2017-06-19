categories={'ds','normal'};
imds=imageDatastore('F:\Gate Roorki\P3\PUSHKAR SIR\folder1','includeSubfolders',true,'LabelSource','foldernames');
[training_set,test_set]=splitEachLabel(imds,0.99);
g1=length(training_set.Files);
g2=length(test_set.Files);
training_features=[];
test_features=[];

for k=1:g1
    im=imread(training_set.Files{k});
    h=HOG(im);
    training_features=[training_features,h];
end
training_label=training_set.Labels;

sv=fitcecoc(training_features',training_label);

im=imread('F:\Gate Roorki\P3\PUSHKAR SIR\folder1\samp.jpg');
h=HOG(im);
test_features=[test_features,h];
out=predict(sv,test_features')