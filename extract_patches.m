function [patches, count, y] = extract_patches(I, A, rfsize, numpatch, var, stride)
% extract 3D patches from a volume
% # 'random': numpatch, nolabel: A = [], count = 0
% # 'positive': count, y = 1
% stride [r c s] 3*1 vector, []
patches = [];
y = [];
count = 0;
center = (rfsize-1)/2;

switch (var)
    case 'random'
        count = numpatch;
        patches = zeros(numpatch, prod(rfsize));
        y = zeros(numpatch, 1);
        for i=1:numpatch
            if (mod(i,10000) == 0)
                fprintf('Extracting patch: %d / %d\n', i, numpatch);
            end
            r = random('unid', size(I,1) - rfsize(1) + 1); % row
            c = random('unid', size(I,2) - rfsize(2) + 1); % column
            s = random('unid', size(I,3) - rfsize(3) + 1); % slice
            patch = I(r:r+rfsize(1)-1,c:c+rfsize(2)-1,s:s+rfsize(3)-1);
            patches(i,:) = patch(:)';
            y(i) = A(r+center(1),c+center(2),s+center(3));
        end   
   
    case 'positive' % all the positive pixels, overlapping patches
        ind = find(A(:)==1);
        [row, col, sli] = ind2sub(size(A),ind);
        for i = 1:length(row)
            if (mod(i,10000) == 0)
                fprintf('Extracting patch: %d / %d\n', i, length(row));
            end
            r = row(i);
            c = col(i);
            s = sli(i);
            if sum([r c s]-center>=0 & [r c s]+center<=size(I)) == 3
                count = count+1;
                patch = I(r-center(1):r+center(1),c-center(2):c+center(2),s-center(3):s+center(3));
                patches(count,:) = patch(:)';
            end
        end
        y = ones(count,1);
        
    case 'stride' % stride >= rfsize non-overlapping
        rind = 1:stride(1):size(I,1)-rfsize(1)+1;
        cind = 1:stride(2):size(I,2)-rfsize(2)+1;
        sind = 1:stride(3):size(I,3)-rfsize(3)+1;
        row = repmat(rind,1,length(cind)*length(sind));
        col = repmat(repelem(cind,1,length(rind)),1,length(sind));
        sli = repelem(sind,1,length(rind)*length(cind));
        count = length(row);
        patches = zeros(count,prod(rfsize));
        y = zeros(count,1);
        for i = 1:count
            if (mod(i,10000) == 0)
                fprintf('Extracting patch: %d / %d\n', i, count);
            end
            r = row(i);
            c = col(i);
            s = sli(i);
            patch = I(r:r+rfsize(1)-1,c:c+rfsize(2)-1,s:s+rfsize(3)-1);
            patches(i,:) = patch(:)';
            y(i) = A(r+center(1),c+center(2),s+center(3));            
        end
        
    otherwise
        error('Unknown patch var type.');
        
end


    


