%%
% Lee Zhen Wei U1922813B CZ4003 Computer Vision
%2.1 a
imagesrc = imread('images\mrt-train.jpg');
whos Pc
%see the size (I always do)
[m n k]=size(imagesrc);
%display it
%imagesc(imagesrc); axis on;

greyImage = rgb2gray(imagesrc);

%2.1 b %showing original image with lousy contrast
figure ('Name','Lousy contrast')
imshow(greyImage);

%2.1 c
%original gray levels
min(greyImage(:)), max(greyImage(:));

%2.1 d
newImg = imsubtract(greyImage,13);
newImg2 = immultiply(newImg,1.335);

%2.1 e
%grey image after subtracting minimum gray level
figure('Name','Image Before Contrast Stretching, Subtracted Gray level')
imshow(newImg)
figure('Name','Resulting Image After Contrast Stretching')
imshow(newImg2)
%after correction, we now have min gray level of 0 and max of 255.
min(newImg2(:)), max(newImg2(:))

%%
%2.2 Histogram Equalization
%2.2 a
%
%greyImage is P 
figure('Name','Original 10 bins')
imhist(greyImage,10);
figure('Name','Original 256 bins')
imhist(greyImage,256);
%the histogram with 256 bin shows finer changes and more detail, as
%compared to the histogram with 10 bins, which has the same approximate
%shape but contain much less details , and omits extreme spikes and data
%changes such as the spike visible at 170-200 using 256 bins, but is
%completely absent in the 10 bin histogram. Hence the 10 bin histogram is
%more smooth.

%2.2 b
P3 = histeq(greyImage,255);
figure('Name','After HistEqualization 10 bins')
imhist(P3,10);
figure('Name','After HistEqualization 256 bins')
imhist(P3,256);
figure
imshow(P3);
%the histogram with 10 bins looks very equalized while the histogram with
%256 bins still contrain wild spikes but is now more distributed across the
%entire range of values from 0-255.

%2.2 c
P4 = histeq(P3,255);
figure('Name','Rerun HistEqualization 10 bins')
imhist(P4,256);
figure('Name','Rerun HistEqualization 10 bins')
imhist(P4,10);
figure
imshow(P3);
%The histogram stays the same, there is no effect,i believe this is because it has
%it has already been equalized. 
%close all
%%
%Linear Spatial Filtering 2.3
%2.3 a
%Gaussian Averaging filter. 
close all
x= -2:2;
y= -2:2;
sigma = 1;
[X,Y] = meshgrid(x,y);
X
Y
%%
e_numerator = exp(-(X.^2+Y.^2)./(2*sigma^2));
denominator = 2*pi*sigma.^2;
filter1 = e_numerator./denominator;
filter1 = filter1 ./sum(filter1(:));
figure('Name','Filter 1 with sigma 1')
mesh(filter1);
%filter2
sigma = 2;
e_numerator = exp(-(X.^2+Y.^2)./(2*sigma^2));
denominator = 2*pi*sigma.^2;
filter2 = e_numerator./denominator;
filter2 = filter2 ./sum(filter2(:));
figure('Name','Filter 2 with sigma 2')
mesh(filter2);

%b
%gaussian noise image processing.
%image has additive gaussian noise
GP = imread('images/ntugn.jpg');
figure 
imshow(GP);

%close all
%c
GP_R =uint8(conv2(GP,filter1));
figure ('Name','Gaussian Noise Filter 1')
imshow(GP_R);
GP_R =uint8(conv2(GP,filter2));
figure ('Name','Gaussian Noise Filter 2')
imshow(GP_R);
%filter2 is more effective at removing the additive gaussian noise than
%filter1, filter1 barely improves upon the original image, and is almost
%not noticable 
%the trade off for using both filters, is that the image appears more
%smoothened as compared to the original image, although noise is indeed
%reduced, as seen in filter2, the image looks smoothened and more blurred,
%due to the use of the gaussian filter, especially filter2 with a higher
%sigma level.
%d 
%image has additive speckle noise 
GP = imread('images/ntusp.jpg');
%figure 
imshow(GP);

GP_R =uint8(conv2(GP,filter1));
figure ('Name', 'Speckle Noise Filter 1')
imshow(GP_R);
GP_R2 =uint8(conv2(GP,filter2));
figure  ('Name', 'Speckle Noise Filter 2')
imshow(GP_R2);
%Gaussian filter works well to filter the image with gaussian noise, 
%Filter 1 is better at removing the gaussian noise from ntugn , 
%Filter 2 with higher sigma value makes the image appear more smoothened
%and blur.
%Both filter 1 and 2 do not manage to remove the speckle noise from ntu_sp
%filter1 ineffective and instead seems to add more noise onto the image and seriously degrade
%the quality of the image,
%filter2 performs slightly better, but it too has a negative effect on the
%original image. Hence both filters do not improve the image at all, and
%are not good at handling speckle noise.
%close all
%2.4

%gaussian noise image processing with medfilt 
GP = imread('images/ntugn.jpg');
%figure 
%imshow(GP);

GP_R =uint8(medfilt2(GP,[3 3]));
figure ('Name','Medfilt gaussian 3,3')
imshow(GP_R);
GP_R =uint8(medfilt2(GP,[5 5]));
figure ('Name','Medfilt gaussian 5 5')
imshow(GP_R);
%Gaussian filtering works better than medfilt for gaussian noise, as
%it preserves more of thesharpness of the image and also more effectively reduces
%the noise, as compared to median filtering with neighbourhood of 3.
% median filtering with neighbourhood of 5 produces very bad results where
% the whole image looks overly smoothened and all many details are lost. 

%speckle noise image processing with medfilt
GP = imread('images/ntusp.jpg');
GP_R =uint8(medfilt2(GP,[3 3]));
figure ('Name','Medfilt speckle 3,3')
imshow(GP_R);
GP_R =uint8(medfilt2(GP,[5 5]));
figure ('Name','Medfilt speckle 5 5')
imshow(GP_R);
%median filtering works better for the speckle noise image, as it manages
%to remove most of the speckle and keep the image clear and sharp,
%especially when the neighbourhood is 3,3, we have to be careful not to
%overdo the median filtering neighbourhood sampling as the 5,5
%neighbourhood turns the image into a smudged mess that looks like a
%watercolor painting 

%hence medfilt is better at handling speckle noise as compared to gaussian
%filtering, but gaussian filtering is better at handling gaussian noise.

%the tradeoff is that medfilt neighbour sampling needs to be carefully
%controlled, else it will result in a smudged mess, that is overly
%smoothened and has the effect of turning the image to look as if it were
%painted with watercolouring technique.
%close all
%%
close all
FFimg = imread('images/pckint.jpg');
imshow(FFimg);
%matrix of complex values
F = fft2(FFimg);
size(F)
%real matrix, abs..
S = abs(F);
figure('Name','power spectrum')
%0.1 scaling to visualize the frequency components easily.
imagesc(fftshift(S.^0.1));
colormap('default');
figure
imagesc(S.^0.1);
[x,y] = ginput(1);
plot(x,y);
x
y
%% 
close all
%neighbour 5 works very well. 
neighbour = 5;
x=16;
y=248;
x2= 241;
y2= 9;
for a = 1:256
    for b = 1:256
        if(a>=x-neighbour && a<=x+neighbour &&b>=y-neighbour && b<= y+neighbour)
            F(a,b) = 0;
        end
         if(a>=x2-neighbour && a<=x2+neighbour &&b>=y2-neighbour && b<= y2+neighbour)
            F(a,b) = 0;
        end
            
    end
end

S = abs(F);
imagesc(S.^0.1);
%%
%ifftImg = ifft2(F);
S2=ifft2(F);
imshow(S2,[]);
%%

%%
%matrix of complex values
primateImg = imread('images/primatecaged.jpg');

primate = rgb2gray(primateImg);
imshow(primate);
F = fft2(primate);
size(F)
%real matrix, abs..
S = abs(F);
figure('Name','power spectrum Monkey')
%0.1 scaling to visualize the frequency components easily.
imagesc(fftshift(S.^0.1));
colormap('default');
figure
imagesc(S.^0.1);
[x,y] = ginput(1);
plot(x,y);
x
y
%%
close all 
F = fft2(primate);
neighbour = 5;
x=252;
y=11;
x2= 248;
y2= 22;
x3=5;
y3=247;
x4=10;
y4=238;
for a = 1:256
    for b = 1:256
        if(a>=x-neighbour && a<=x+neighbour &&b>=y-neighbour && b<= y+neighbour)
            F(a,b) = 0;
        end
         if(a>=x2-neighbour && a<=x2+neighbour &&b>=y2-neighbour && b<= y2+neighbour)
            F(a,b) = 0;
         end
         if(a>=x3-neighbour && a<=x3+neighbour &&b>=y3-neighbour && b<= y3+neighbour)
            F(a,b) = 0;
         end
         if(a>=x4-neighbour && a<=x4+neighbour &&b>=y4-neighbour && b<= y4+neighbour)
            F(a,b) = 0;
         end
         if(a>=6-neighbour && a<=6+neighbour &&b>=6-neighbour && b<= 6+neighbour)
            F(a,b) = 0;
        end
        
            
    end
end


resultImg=ifft2(F);
%fourier spectrum
figure
absF = abs(F);
imagesc(absF.^0.1);
colormap('default');
figure 
imshow(resultImg,[]);
%have attempted to remove the cage by setting the neighbours around the
%peaks in the fourier spectrum to 0, the issue is that by using the crude
%method of manually identifing these points and removing them using a
%neighbour bounding box, I also remove details from parts of the image that
%is not part of the 'cage'. This brute approach manages to remove the
%strongly represented parts of the cage but also removes some details
%because i am not able to more accurately and precisely remove the peaks
%and also the gradient spillover that also constitute parts of the cage in
%the fourier domain. I would need a better approach to precisely and
%selectively pick out all relevant peaks with varying neighbourhood size
%bounding volumes to more effectively remove these issues
%%
%a
bookImg = imread('images/book.jpg');
imshow(bookImg);
%b
[X Y] = ginput(4);
%must click in this sequence else the image may appear rotated 
%topleft,topright,bottomright,bottomleft
xDim = [0; 210; 210; 0];
yDim = [0; 0; 297; 297];
%c
%u = A \ v;
%v is vector for x^n(im) y^n(im).... 
v = [xDim(1); yDim(1); xDim(2); yDim(2); xDim(3); yDim(3); xDim(4); yDim(4)];
A = [
    [X(1), Y(1), 1, 0, 0, 0, -xDim(1)*X(1), -xDim(1)*Y(1)];
    [0, 0, 0, X(1), Y(1), 1, -yDim(1)*X(1), -yDim(1)*Y(1)];
    [X(2), Y(2), 1, 0, 0, 0, -xDim(2)*X(2), -xDim(2)*Y(2)];
    [0, 0, 0, X(2), Y(2), 1, -yDim(2)*X(2), -yDim(2)*Y(2)];
    [X(3), Y(3), 1, 0, 0, 0, -xDim(3)*X(3), -xDim(3)*Y(3)];
    [0, 0, 0, X(3), Y(3), 1, -yDim(3)*X(3), -yDim(3)*Y(3)];
    [X(4), Y(4), 1, 0, 0, 0, -xDim(4)*X(4), -xDim(4)*Y(4)];
    [0, 0, 0, X(4), Y(4), 1, -yDim(4)*X(4), -yDim(4)*Y(4)];
];
%projective transformation using the matrix A based on input V
u = A\v;

%convert projective transformation to normal matrix form. 
U = reshape([u;1],3,3)';
%transform original coordinates
w = U*[X';Y';ones(1,4)];
w = w./(ones(3,1)*w(3,:));

%d

%transformative matrix... how much to transform by projection.
T = maketform('projective',U');
P2 = imtransform(bookImg,T,'XData',[0 210], 'YData', [0 297]);

%e
%display final image after projection.
imshow(P2);
%the resulting image shows that no additional details are obtained from the
%image but we are able to successfully crop the image, and obtain the
%relevant pixels that we require, and to project those pixels and map them
%to fit and scale to the dimensions that we expect it to with the use of
%this projection matrix, this results in a visible improvement , at least
%in the terms of presentation as images that otherwise may be disorderly ,
%when shot in a slanted manner, can be salvaged and transformed into useful
%images using this projection function. this effect has also been observed
%to be used in Microsoft's office lens application and I have been a big
%user of it, and use it to scan alot of my documents, it is fascinating to
%see that the actual implementation is so straight forward and I am very
%impressed by the results, perhaps by combining it with other functions we
%may clean up the image and make it even better, this post-processing is
%also done by office lens and have good enough results that i submit to
%school when they require some documentation from me. All in all, have
%learnt alot of useful techniques from this lab experiment. 

