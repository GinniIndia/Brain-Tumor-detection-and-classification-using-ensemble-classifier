function varargout = temp(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @temp_OpeningFcn, ...
                   'gui_OutputFcn',  @temp_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% --- Executes just before temp is made visible.
function temp_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
ss = ones(200,200);
axes(handles.axes2);
imshow(ss);
axes(handles.axes3);
imshow(ss);
% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = temp_OutputFcn(hObject, eventdata, handles) 

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)

load  train_input1;
load train_sout1;

X = train_input1;
Y = train_sout1;

t = templateSVM('Standardize', true, 'KernelFunction', 'polynomial');
Mdl = fitcecoc(X, Y, 'Learners', t);
count = 0; TN=0; TP=0; FP=0; FN=0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:2172
    p = predict(Mdl,X(i,:));
    if(p == string(Y(i)))
        count = count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/2172); 
end
specificity = TN/ (TN+FP);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F1 = 2*precision*recall / (precision+recall);
YI = recall + specificity -1;
delete(hWaitBar);
accuracy = count/2172*100;
load  train_input2;
load train_sout2;

X = train_input2;
Y = train_sout2;

count = 0; TN=0; TP=0; FP=0; FN=0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:384
    p = predict(Mdl,X(i,:));
    if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/384); 
end
specificity2 = TN/ (TN+FP);
precision2 = TP/(TP+FP);
recall2 = TP/(TP+FN);
F12 = 2*precision2*recall2 / (precision2+recall2);
YI2 = recall2 + specificity2 -1;
delete(hWaitBar);
accuracy2 = count/384*100;

SVM = 0

specificity = specificity*.85 + specificity2*.15
precision = precision*.85 + precision2*.15
recall = recall*.85 + recall2*.15
F1 = F1*.85 + F12*.15
YI = YI*.85 + YI2*.15
accuracy = accuracy*.85 + accuracy2*.15
sprintf('Accuracy of RBF kernel is: %g%%', accuracy);
set(handles.edit1,'string',accuracy);

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)

load  train_input1;
load train_sout1;

X = train_input1;
Y = train_sout1;

Mdl = fitcnb(X, Y, 'DistributionNames', 'kernel');
count = 0; TP =0; TN =0; FP = 0; FN = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:2172
    p = predict(Mdl,X(i,:));
   if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/2172); 
end
specificity = TN/ (TN+FP);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F1 = 2*precision*recall / (precision+recall);
YI = recall + specificity -1;
delete(hWaitBar);
accuracy = count/2172*100;

load  train_input2;
load train_sout2;

X = train_input2;
Y = train_sout2;

count = 0; TP =0; TN =0; FP = 0; FN = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');
for i=1:384
    p = predict(Mdl,X(i,:));
   if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/384); 
end
specificity2 = TN/ (TN+FP);
precision2 = TP/(TP+FP);
recall2 = TP/(TP+FN);
F12 = 2*precision2*recall2 / (precision2+recall2);
YI2 = recall2 + specificity2 -1;
delete(hWaitBar);
accuracy2 = count/384*100;

NB = 0

specificity = specificity*.85 + specificity2*.15
precision = precision*.85 + precision2*.15
recall = recall*.85 + recall2*.15
F1 = F1*.85 + F12*.15
YI = YI*.85 + YI2*.15
accuracy = accuracy*.85 + accuracy2*.15

sprintf('Accuracy of Naive Baise is: %g%%', accuracy);
set(handles.edit2,'string',accuracy);

    
% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)

load  train_input1;
load train_sout1;

X = train_input1;
Y = train_sout1;

NumTrees = 100;
Mdl = TreeBagger(NumTrees, X, Y, 'OOBPrediction', 'on');
count = 0; TP =0; TN =0; FP = 0; FN = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:2172
    p = predict(Mdl,X(i,:));
   if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/2172); 
end
specificity = TN/ (TN+FP);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F1 = 2*precision*recall / (precision+recall);
YI = recall + specificity -1;
delete(hWaitBar);
accuracy = count/2172*100;

load  train_input2;
load train_sout2;

X = train_input2;
Y = train_sout2;

count = 0; TP =0; TN =0; FP = 0; FN = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:384
    p = predict(Mdl,X(i,:));
   if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/384); 
end
specificity2 = TN/ (TN+FP);
precision2 = TP/(TP+FP);
recall2 = TP/(TP+FN);
F12 = 2*precision2*recall2 / (precision2+recall2);
YI2 = recall2 + specificity2 -1;
delete(hWaitBar);
accuracy2 = count/384*100;

RF = 0

specificity = specificity*.85 + specificity2*.15
precision = precision*.85 + precision2*.15
recall = recall*.85 + recall2*.15
F1 = F1*.85 + F12*.15
YI = YI*.85 + YI2*.15
accuracy = accuracy*.85 + accuracy2*.15

sprintf('Accuracy of Random Forest is: %g%%', accuracy);
set(handles.edit3,'string',accuracy);


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
load train_input1;
load train_sout1;
load train_output1;

X = train_input1;
Y = train_sout1;
x = train_input1';
t = train_output1';
trainFcn = 'trainbr'; 
hiddenLayerSize = 20;
net = patternnet(hiddenLayerSize, trainFcn);
%net.trainParam.showWindow = false;
[net,tr] = train(net,x,t);  %% tr findout
count = 0; TN =0; TP =0; FN = 0; FP = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:2172
    p = net(x(:,i));
    [val,idx] = max(p);
    if(t(idx,i)==1)
        count=count+1;
        if(Y(i)=="Benign")
            TP = TP+1;
        elseif(Y(i)=="Malignant")
            TN = TN+1;
        end
    elseif(Y(i)=="Benign")
        FP = FP+1;
    elseif(Y(i)=="Malignant")
        FN = FN+1;
    end
   waitbar(i/2172); 
end
specificity = TN/ (TN+FP);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F1 = 2*precision*recall / (precision+recall);
YI = recall + specificity -1;
delete(hWaitBar);
accuracy = count/2172*100;

load train_input2;
load train_sout2;
load train_output2;

X = train_input2;
Y = train_sout2;
x = train_input2';
t = train_output2';


count = 0; TN =0; TP =0; FN = 0; FP = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:384
    p = net(x(:,i));
    [val,idx] = max(p);
    if(t(idx,i)==1)
        count=count+1;
        if(Y(i)=="Benign")
            TP = TP+1;
        elseif(Y(i)=="Malignant")
            TN = TN+1;
        end
    elseif(Y(i)=="Benign")
        FP = FP+1;
    elseif(Y(i)=="Malignant")
        FN = FN+1;
    end
   waitbar(i/384); 
end
specificity2 = TN/ (TN+FP);
precision2 = TP/(TP+FP);
recall2 = TP/(TP+FN);
F12 = 2*precision2*recall2 / (precision2+recall2);
YI2 = recall2 + specificity2 -1;
delete(hWaitBar);
accuracy2 = count/384*100;
NN = 0
specificity = specificity*.85 + specificity2*.15
precision = precision*.85 + precision2*.15
recall = recall*.85 + recall2*.15
F1 = F1*.85 + F12*.15
YI = YI*.85 + YI2*.15
accuracy = accuracy*.85 + accuracy2*.15

sprintf('Accuracy of ANN is: %g%%', accuracy);
set(handles.edit4,'string',accuracy);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
load train_input1;
load train_sout1;

X = train_input1;
Y = train_sout1;

Mdl = fitctree(X, Y);
count = 0; TP =0; TN =0; FP = 0; FN = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:2172
    p = predict(Mdl,X(i,:));
   if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/2172); 
end
specificity = TN/ (TN+FP);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F1 = 2*precision*recall / (precision+recall);
YI = recall + specificity -1;
delete(hWaitBar);
accuracy = count/2172*100;

load train_input2;
load train_sout2;

X = train_input2;
Y = train_sout2;
count = 0; TP =0; TN =0; FP = 0; FN = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:384
    p = predict(Mdl,X(i,:));
   if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/384); 
end
specificity2 = TN/ (TN+FP);
precision2 = TP/(TP+FP);
recall2 = TP/(TP+FN);
F12 = 2*precision2*recall2 / (precision2+recall2);
YI2 = recall2 + specificity2 -1;
delete(hWaitBar);
accuracy2 = count/384*100;
DT = 0
specificity = specificity*.85 + specificity2*.15
precision = precision*.85 + precision2*.15
recall = recall*.85 + recall2*.15
F1 = F1*.85 + F12*.15
YI = YI*.85 + YI2*.15
accuracy = accuracy*.85 + accuracy2*.15

sprintf('Accuracy of Decision Tree is: %g%%', accuracy);
set(handles.edit5,'string',accuracy);


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
load train_input1;
load train_sout1;

X = train_input1;
Y = train_sout1;

k = 1;
Mdl = fitcknn(X, Y, 'NumNeighbors', k);

count = 0; TP =0; TN =0; FP = 0; FN = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:2172
    p = predict(Mdl,X(i,:));
    if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/2172); 
end
specificity = TN/ (TN+FP);
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = 2*precision*recall / (precision+recall)
YI = recall + specificity -1;
delete(hWaitBar);
accuracy = count/2172*100;

load train_input2;
load train_sout2;

X = train_input2;
Y = train_sout2;

count = 0; TP =0; TN =0; FP = 0; FN = 0;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');

for i=1:384
    p = predict(Mdl,X(i,:));
    if(p == string(Y(i)))
        count=count+1;
        if(p=="Benign")
            TP = TP+1;
        elseif(p=="Malignant")
            TN = TN+1;
        end
    elseif(p=="Benign")
        FP = FP+1;
    elseif(p=="Malignant")
        FN = FN+1;
    end
   waitbar(i/384); 
end
specificity2 = TN/ (TN+FP);
precision2 = TP/(TP+FP)
recall2 = TP/(TP+FN)
F12 = 2*precision2*recall2 / (precision2+recall2)
YI2 = recall2 + specificity2 -1;
delete(hWaitBar);
accuracy2 = count/384*100;
KNN = 0
specificity = specificity*.85 + specificity2*.15
precision = precision*.85 + precision2*.15
recall = recall*.85 + recall2*.15
F1 = F1*.85 + F12*.15
YI = YI*.85 + YI2*.15
accuracy = accuracy*.85 + accuracy2*.15

sprintf('Accuracy of KNN is: %g%%', accuracy);
set(handles.edit6,'string',accuracy);


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)

load train_input1;
load train_sout1;

X = train_input1;
Y = train_sout1;

hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');
count = 0; TP =0; TN =0; FP = 0; FN = 0;
NumTrees = 100;
  Mdl1 = TreeBagger(NumTrees, X, Y, 'OOBPrediction', 'on');
    %t = templateSVM('Standardize', true, 'KernelFunction', 'quadratic');
    % Fit a multiclass model for Support Vector Machine
    %Mdl = fitcecoc(X, Y, 'Learners', t);
    k = 1;
    Mdl2 = fitcknn(X, Y, 'NumNeighbors', k);
    % Predict the output of an identified model
    %Predictions = predict(Mdl, input_image)
     %Mdl = fitcnb(X, Y, 'DistributionNames', 'kernel');
      Mdl3 = fitctree(X, Y);

for i=1:2172
    % Predict the output of an identified model
    p1 = predict(Mdl1, X(i,:));  
    p2 = predict(Mdl2, X(i,:));
    p3 = predict(Mdl3, X(i,:));
    
    right = 0; left = 0;
    if(p1 == string(Y(i)))
        right=right+1;
    else
        left=left+1;
    end
    if(p2 == string(Y(i)))
        right=right+1;
    else
        left=left+1;
    end
    if(p3 == string(Y(i)))
        right=right+1;
    else
        left=left+1;
    end
    if(left<right)
        count=count+1;
         if(Y(i)=="Benign")
            TP = TP+1;
        elseif(Y(i)=="Malignant")
            TN = TN+1;
        end
    elseif(Y(i)=="Benign")
        FP = FP+1;
    elseif(Y(i)=="Malignant")
        FN = FN+1;
    end
    waitbar(i/2172);
end
specificity = TN/ (TN+FP);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F1 = 2*precision*recall / (precision+recall);
YI = recall + specificity -1;
delete(hWaitBar);
accuracy = count/2172*100;


load train_input2;
load train_sout2;

X = train_input2;
Y = train_sout2;
TP1 = TP;
TN1 = TN;
FN1 = FN;
FP1 = FP;
hWaitBar = waitbar(0,'Evaluating Maximum Accuracy');
count = 0; TP =0; TN =0; FP = 0; FN = 0;

for i=1:384
    % Predict the output of an identified model
    p1 = predict(Mdl1, X(i,:));  
    p2 = predict(Mdl2, X(i,:));
    p3 = predict(Mdl3, X(i,:));
    
    right = 0; left = 0;
    if(p1 == string(Y(i)))
        right=right+1;
    else
        left=left+1;
    end
    if(p2 == string(Y(i)))
        right=right+1;
    else
        left=left+1;
    end
    if(p3 == string(Y(i)))
        right=right+1;
    else
        left=left+1;
    end
    if(left<right)
        count=count+1;
         if(Y(i)=="Benign")
            TP = TP+1;
        elseif(Y(i)=="Malignant")
            TN = TN+1;
        end
    elseif(Y(i)=="Benign")
        FP = FP+1;
    elseif(Y(i)=="Malignant")
        FN = FN+1;
    end
    waitbar(i/384);
end
delete(hWaitBar);
specificity2 = TN/ (TN+FP);
precision2 = TP/(TP+FP);
recall2 = TP/(TP+FN);
F12 = 2*precision2*recall2 / (precision2+recall2)
YI2 = recall2 + specificity2 -1;
accuracy2 = count/384*100;
TP1 = TP1+TP
TN1 = TN1+TN
FP1 = FP1+FP
FN1 = FN1+FN
Proposed = 0
specificity = specificity*.85 + specificity2*.15
precision = precision*.85 + precision2*.15
recall = recall*.85 + recall2*.15
F1 = F1*.85 + F12*.15
YI = YI*.85 + YI2*.15

accuracy = accuracy*.85 + accuracy2*.15
sprintf('Accuracy of Fusion Classifier is: %g%%', accuracy);
set(handles.edit7,'string',accuracy);


function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[FileName,PathName] = uigetfile('*.jpg;*.png;*.bmp','Pick an MRI Image');
if isequal(FileName,0)||isequal(PathName,0)
    warndlg('User Press Cancel');
else
    P = imread([PathName,FileName]);
    P = imresize(P,[200,200]);
   % input =imresize(a,[512 512]); 
  
  axes(handles.axes2)
  imshow(P);%title('Brain MRI Image');
  handles.ImgData = P;
end
if isfield(handles, 'ImgData')
    I = handles.ImgData;
    %gray = rgb2gray(I);
    img = im2bw(I,.6);
    img = bwareaopen(img,80); 
    axes(handles.axes3)
    imshow(img);%title('Segmented Image');
   
    handles.seg_img = img;
    %imwrite(img ,['pic2.jpg'])
    guidata(hObject,handles);

img = double(img);
[A1,H1,V1,D1] = swt2(img,1,'db4');
%[A2,H2,V2,D2] = swt2(A1,1,'db4');
%[A3,H3,V3,D3] = swt2(A2,1,'db4');
DWT_feat = [A1,H1,V1,D1]; %%imp
G = pca(DWT_feat);
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

load feat_input;
load feat_output;
load string_output;

X = feat_input;
Y = string_output;

NumTrees = 100;
  Mdl1 = TreeBagger(NumTrees, X, Y, 'OOBPrediction', 'on');
k = 1;
    Mdl2 = fitcknn(X, Y, 'NumNeighbors', k);
 Mdl3 = fitctree(X, Y);
 p1 = predict(Mdl1, feat);  
 p2 = predict(Mdl2, feat);
 p3 = predict(Mdl3, feat);
 right = 0; left = 0;
 if(p1 == string("Malignant"))
        right=right+1;
    else
        left=left+1;
    end
    if(p2 == string("Malignant"))
        right=right+1;
    else
        left=left+1;
    end
    if(p3 == string("Malignant"))
        right=right+1;
    else
        left=left+1;
    end
    if(right>left)
        species = "Malignant";
    else
        species = "Benign";
    end
 
 if strcmpi(species,'MALIGNANT')
     helpdlg(' Malignant Tumor ');
     disp(' Malignant Tumor ');
 else
     helpdlg(' Benign Tumor ');
     disp(' Benign Tumor ');
 end
 set(handles.edit9,'string',species);
 % Put the features in GUI
set(handles.edit10,'string',Mean);
set(handles.edit11,'string',Standard_Deviation);
set(handles.edit12,'string',Entropy);
set(handles.edit13,'string',RMS);
set(handles.edit14,'string',Variance);
set(handles.edit15,'string',Smoothness);
set(handles.edit16,'string',Kurtosis);
set(handles.edit17,'string',Skewness);
set(handles.edit18,'string',IDM);
set(handles.edit19,'string',Contrast);
set(handles.edit20,'string',Correlation);
set(handles.edit21,'string',Energy);
set(handles.edit22,'string',Homogeneity);
end


% --- Executes during object creation, after setting all properties.
function text4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double


% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit14_Callback(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit14 as text
%        str2double(get(hObject,'String')) returns contents of edit14 as a double


% --- Executes during object creation, after setting all properties.
function edit14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit15_Callback(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit15 as text
%        str2double(get(hObject,'String')) returns contents of edit15 as a double


% --- Executes during object creation, after setting all properties.
function edit15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit16_Callback(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit16 as text
%        str2double(get(hObject,'String')) returns contents of edit16 as a double


% --- Executes during object creation, after setting all properties.
function edit16_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit17_Callback(hObject, eventdata, handles)
% hObject    handle to edit17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit17 as text
%        str2double(get(hObject,'String')) returns contents of edit17 as a double


% --- Executes during object creation, after setting all properties.
function edit17_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit18_Callback(hObject, eventdata, handles)
% hObject    handle to edit18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit18 as text
%        str2double(get(hObject,'String')) returns contents of edit18 as a double


% --- Executes during object creation, after setting all properties.
function edit18_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit19_Callback(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit19 as text
%        str2double(get(hObject,'String')) returns contents of edit19 as a double


% --- Executes during object creation, after setting all properties.
function edit19_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit20_Callback(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit20 as text
%        str2double(get(hObject,'String')) returns contents of edit20 as a double


% --- Executes during object creation, after setting all properties.
function edit20_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit21_Callback(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit21 as text
%        str2double(get(hObject,'String')) returns contents of edit21 as a double


% --- Executes during object creation, after setting all properties.
function edit21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit22_Callback(hObject, eventdata, handles)
% hObject    handle to edit22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit22 as text
%        str2double(get(hObject,'String')) returns contents of edit22 as a double


% --- Executes during object creation, after setting all properties.
function edit22_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
