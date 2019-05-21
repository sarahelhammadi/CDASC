A = importdata('karate.txt');
truelabels = importdata('karatey.txt');

addpath('svm-matlab/common_files');
k=5;
Max_no_questions=300; % Update the number of questions, we want to ask to users
pairs = [];
pairs_prop = [];
indpo = [];
acc = [];
nbCluster=2;
nmi_all=[];
CC=[];
N=size(A,1);

% Step 1: Create the pairwise similarity matrix

display('Step 1: Compute Similarity matrix');

Sim2=zeros(N,N);

for i=1:N 
    for j=1:N
        nij=find(A(:,i)==1 & A(:,j)==1);
        ki=sum(A(:,i)==1);
        kj=sum(A(:,j)==1);
        %Sim(i,j)=length(nij)/(ki+kj);
                Sim(i,j)=length(nij)/sqrt(ki*kj);

    end
end



%replace diagonal elements with 0

display('Step 2: Find kNearestNeighbors');


[neighborIds neighborDistances] = kNearestNeighbors_mod(1-Sim, k);

%find Similarity
%Sim=-dist_mat.^2;
constraints=zeros(N,N);

%%%%%%%%%%%%%%%%%%%%%
ConnectedComp=zeros(length(N));
for i=1:N 
    ConnectedComp(i)=i;
end

noQuestions=0;
while noQuestions < Max_no_questions



        [NcutDiscrete,output,NcutEigenvalues, U] = ncutW(Sim,nbCluster);

        Outputs = zeros(N,1);
        for ii = 1:nbCluster
            index = find(NcutDiscrete(:,ii)==1);
            Outputs(index) = ii;
        end
        if length(unique(Outputs))==nbCluster
            nmi_index=nmi(truelabels,Outputs)
            nmi_all=[nmi_all; [noQuestions nmi_index]];
        end
    %Compute V measure
    goldla = unique(truelabels);%c
    resla = unique(Outputs); %k
    comatrix = zeros(length(resla),length(goldla)); %k by c
    for i = 1:size(comatrix,1)
        co_i = find(Outputs==resla(i));
        for j = 1:size(comatrix,2)
               co_j = find(truelabels == goldla(j));
               c_intersect = intersect(co_i,co_j);
               comatrix(i,j) = length(c_intersect);
        end
    end
    [v,hc,hk,h_ck,h_kc] = calculate_v_measure (comatrix);
    %               RI  = RandIndex(c1,c2)
    %               CRI   = CRI(c1,c2,noQuestions)
    %               NMI = nmi(c1,c2)
    %               JC_now  = Jaccard_Coefficient(c1);

%%%%%%

    %Calculate Entropy
    for ii=1:N
            count = zeros(nbCluster,1);
            temp = Outputs(neighborIds(:,ii));
            for j = 1:length(temp)
                 count(temp(j)) = count(temp(j)) + 1;
            end
            count = count/sum(count);
            entropy(ii) = 1-sum(temp==Outputs(ii))/length(temp);           
    end

    entropy = entropy + 1e-10;
    entropy(indpo) = -1;
    [Y,index] = max(entropy);        
    indpo = [indpo;index];

            while isempty(find(constraints(index,neighborIds(:,index))==0))
                entropy(indpo) = -1;
                [Y,index] = max(entropy);        
                indpo = [indpo;index];
                    if length(unique(entropy))==1 && unique(entropy) == -1
                        error('End');
                        break;
                    end

        end


    % Purify NN graph
        pairstemp = [];       
        pairstemp1 = [];
        for ii = 1:length(neighborIds(:,index))      
            if constraints(index,neighborIds(ii,index))==0

                              disp(['Quering: ' num2str(index) ',' num2str(neighborIds(ii,index))])
                if truelabels(index) ~= truelabels(neighborIds(ii,index)) %&& ...
                    pairstemp = [ pairstemp;[index neighborIds(ii,index) -1 0] ];
                    pairstemp1 = [ pairstemp1;[index neighborIds(ii,index) -1 0] ];
                    %Sim(index, neighborIds(ii,index)) = -10000;
                    %Sim(neighborIds(ii,index), index) = -10000;
                    %s(y1,3)=-10000;
                    %s(y2,3)=-10000;
                    constraints(index,neighborIds(ii,index))=-1;
                    constraints(neighborIds(ii,index),index)=-1;
                    ccw=find(ConnectedComp==ConnectedComp(neighborIds(ii,index)));
                    constraints(index,ccw)=-1;
                    constraints(ccw,index)=-1;
                    ccw2=find(ConnectedComp==ConnectedComp(index));
                    constraints(ccw2,neighborIds(ii,index))=-1;
                    constraints(neighborIds(ii,index),ccw2)=-1;
                    [AMat,BMat] = meshgrid(ccw, ccw2);
                    constraints(AMat(:),BMat(:))=-1;
                    constraints(BMat(:),AMat(:))=-1;
                else
                    pairstemp1 = [ pairstemp1;[index neighborIds(ii,index) 1 0] ];
                    %Sim(index, neighborIds(ii,index)) = 0;
                    %Sim(neighborIds(ii,index), index) = 0;
                    %s(y1,3)=0;
                    %s(y2,3)=0;
                    constraints(index,neighborIds(ii,index))=1;
                    constraints(neighborIds(ii,index),index)=1;
                    ccw=find(ConnectedComp==ConnectedComp(neighborIds(ii,index)));
                    constraints(index,ccw)=1;
                    constraints(ccw,index)=1;
                    ccw2=find(ConnectedComp==ConnectedComp(index));
                    constraints(ccw2,neighborIds(ii,index))=1;
                    constraints(neighborIds(ii,index),ccw2)=1;
                    [AMat,BMat] = meshgrid(ccw, ccw2);
                    constraints(AMat(:),BMat(:))=1;
                    constraints(BMat(:),AMat(:))=1;
                    ccw3=find(constraints(neighborIds(ii,index),:)==-1);
                    constraints(index,ccw3)=-1;
                    constraints(ccw3,index)=-1;
                    [AMat,BMat] = meshgrid(ccw2, ccw3);
                    constraints(AMat(:),BMat(:))=-1;
                    constraints(BMat(:),AMat(:))=-1;


                    ccw4=find(constraints(index,:)==-1);
                    constraints(neighborIds(ii,index),ccw4)=-1;
                    constraints(ccw4,neighborIds(ii,index))=-1;
                    [AMat,BMat] = meshgrid(ccw4, ccw);
                    constraints(AMat(:),BMat(:))=-1;
                    constraints(BMat(:),AMat(:))=-1;

                end
                noQuestions=noQuestions+1
                %constraints=propConstraints_faster(constraints);
                Sim=adjustSimilarity_faster_as(Sim,constraints);
                                            Ac=constraints;
                Ac(Ac<0)=0;
                [numberofConnectedComp,ConnectedComp] = graphconncomp(sparse(Ac), 'Directed', false);
                numberofConnectedComp
                CC=[CC;[noQuestions numberofConnectedComp]];

            end
            
        end
                    
        
v
end
