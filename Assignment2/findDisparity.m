function dispM = findDisparity(IL, IR, maxDisp, windowSize, method)
    IL=double(IL);
    IR=double(IR);
    [m,n]=size(IR);
    IR=[zeros(m,maxDisp),IR]; % Pad zeros and the left side of IR
    window=ones(windowSize,windowSize); % Create window for Convolution
    shift=floor(windowSize/2); % index for copying the matrix after conv
    IR_Cell=cell(maxDisp+1,1); % Store different matrixes after shifting
    
    for d=0:maxDisp
        % Create different matrixes for each shifting
        IR_Cell{d+1}=IR(:,1+maxDisp-d:n+maxDisp-d);
    end
    
    disparity=zeros(m,n,maxDisp+1);
    for d=0:maxDisp
        if method=='SSD'
            difference=(IL-IR_Cell{d+1}).^2; % formula
            Conv=conv2(difference,window); % Conv with ones window to aggregate them
            disparity(:,:,d+1)=Conv(1+shift:size(Conv,1)-shift,1+shift:size(Conv,2)-shift);
        elseif method=='SAD'
            difference=abs(IL-IR_Cell{d+1}); % formula
            Conv=conv2(difference,window); % Conv with ones window to aggregate them
            disparity(:,:,d+1)=Conv(1+shift:size(Conv,1)-shift,1+shift:size(Conv,2)-shift);
        elseif method=='NCC'
            Numerator=IL.*IR_Cell{d+1}; % formula
            Numerator_C=conv2(Numerator,window);
            Numerator_C=Numerator_C(1+shift:size(Numerator_C,1)-shift,1+shift:size(Numerator_C,2)-shift);
            IL_2=IL.^2;
            IL_2_C=conv2(IL_2,window);
            IL_2_C=IL_2_C(1+shift:size(IL_2_C,1)-shift,1+shift:size(IL_2_C,2)-shift);
            IR_2=IR_Cell{d+1}.^2;
            IR_2_C=conv2(IR_2,window);
            IR_2_C=IR_2_C(1+shift:size(IR_2_C,1)-shift,1+shift:size(IR_2_C,2)-shift);
            Denominator_C=sqrt(IR_2_C.*IL_2_C);
            difference=Numerator_C./(Denominator_C+1);
            disparity(:,:,d+1)=difference;
        end
    end 
    
    if method=='NCC'
        [~,dispM]=max(disparity,[],3); % Index of the maximum value
    else
        [~,dispM]=min(disparity,[],3); % Index of the minimum value
    end
    dispM=dispM-1;
end

