%CMPT361 Spring 2022
%Ali Tohidi, 301355519

% ** PART 1 **
% Read the images
% highFreq Photo by David Clode on Unsplash -> https://unsplash.com/photos/GOv0VZSc4Fg
hp = imread('highFreq.png');
lp = imread('lowFreq.png');

% Convert images to double
lp = im2double(lp);
hp = im2double(hp);

% Conver images to grayscale 
lp = rgb2gray(lp);
hp = rgb2gray(hp);

% Crop the images ro 500*500
hp = hp(1:500, 151:650);
lp = lp(187:end, 16:end);

% Save the resulting images in HP and LP .png files
imwrite(hp,'assets/HP.png','PNG');
imwrite(lp,'assets/LP.png','PNG');

% ** PART 2 **
% Take the Fourier transforms of HP and LP
low_fou =fft2(lp);
high_fou =fft2(hp);

% Visualize their magnitude.
imshow([fftshift(abs(low_fou)) fftshift(abs(high_fou))]/1)

% Save the frequency response magnitude as images with adjusted brightness
imwrite(fftshift(abs(low_fou))/50,'assets/LP-freq.png','PNG');
imwrite(fftshift(abs(high_fou))/50,'assets/HP-freq.png','PNG');

% ** PART 3 **
% Define a Sobel kernel, a Gaussian kernel, and a derivative-of-Gaussian
% (DoG) Kernel

gaussian_kernel = fspecial('gaussian', 20, 2.5);
sobel_kernel = fspecial('sobel');
dog_kernel = conv2(gaussian_kernel, sobel_kernel);

% Visualize the kernels
surf(gaussian_kernel)
surf(sobel_kernel)
surf(dog_kernel)

% Save the surf visualizations of the Gaussian and DoG kernels as 
% gaus-surf.png and dog-surf.png
saveas(surf(gaussian_kernel), 'assets/gaus-surf.png');
saveas(surf(dog_kernel), 'assets/dog-surf.png');

% Apply the Gaussian filter to HP and LP
hp_filt = imfilter(hp, gaussian_kernel);
lp_filt = imfilter(lp, gaussian_kernel);

% Save the filtered images in HP and LP .png files
imwrite(hp_filt,'assets/HP-filt.png','PNG');
imwrite(lp_filt,'assets/LP-filt.png','PNG');

% Compute the frequency domain representation of the filtered images
hp_filt_freq = fft2(hp_filt);
lp_filt_freq = fft2(lp_filt);

% Visualize their magnitude.
imshow([fftshift(abs(lp_filt_freq)) fftshift(abs(hp_filt_freq))]/50)

% Save the frequency response magnitude as images with adjusted brightness
imwrite(fftshift(abs(hp_filt_freq))/50, 'assets/HP-filt-freq.png','PNG');
imwrite(fftshift(abs(lp_filt_freq))/50, 'assets/LP-filt-freq.png','PNG');

% Compute the Fourier transform of the DoG kernels using the transform size
% of 500x500
dog_fft = fft2(dog_kernel, 500, 500);

% Apply this filter to both images in the frequency domain
hp_dogfilt_freq = dog_fft .* high_fou;
lp_dogfilt_freq = dog_fft .* low_fou;

% Save the freq. domain versions
imwrite(abs(fftshift(hp_dogfilt_freq)), 'assets/HP-dogfilt-freq.png','PNG');
imwrite(abs(fftshift(lp_dogfilt_freq)), 'assets/LP-dogfilt-freq.png','PNG');

% Convert them back to spatial domain
hp_dogfilt = ifft2(hp_dogfilt_freq);
lp_dogfilt = ifft2(lp_dogfilt_freq);

% Save the filtered images
imwrite(hp_dogfilt .* 10, 'assets/HP-dogfilt.png', 'PNG');
imwrite(lp_dogfilt .* 10, 'assets/LP-dogfilt.png', 'PNG');

% ** PART 4 **
% Subsample the two images (using 1:2:end) to half the size in both dimensions
hp_sub2 = hp(1:2:end, 1:2:end);
lp_sub2 = lp(1:2:end, 1:2:end);

% Save the subsampled images
imwrite(hp_sub2, 'assets/HP-sub2.png', 'PNG');
imwrite(lp_sub2, 'assets/LP-sub2.png', 'PNG');

% Compute the freq. domain versions of them
hp_sub2_freq = abs(fftshift(fft2(hp_sub2)))/50;
lp_sub2_freq = abs(fftshift(fft2(lp_sub2)))/50;

% Save the freq. domain versions of them
imwrite(hp_sub2_freq, 'assets/HP-sub2-freq.png', 'PNG');
imwrite(lp_sub2_freq, 'assets/LP-sub2-freq.png', 'PNG');

% Subsample the two images (using 1:4:end) to half the size in both dimensions
hp_sub4 = hp(1:4:end, 1:4:end);
lp_sub4 = lp(1:4:end, 1:4:end);

% Save the subsampled images
imwrite(hp_sub4, 'assets/HP-sub4.png', 'PNG');
imwrite(lp_sub4, 'assets/LP-sub4.png', 'PNG');

% Compute the freq. domain versions of them
hp_sub4_freq = abs(fftshift(fft2(hp_sub4)))/50;
lp_sub4_freq = abs(fftshift(fft2(lp_sub4)))/50;

% Save the freq. domain versions of them
imwrite(hp_sub4_freq, 'assets/HP-sub4-freq.png', 'PNG');
imwrite(lp_sub4_freq, 'assets/LP-sub4-freq.png', 'PNG');

% Apply anti-aliasing to HP for half and quarter options.
hp_filtered = imfilter(hp, fspecial('gaussian', 3, 1));
hp_filtered_more = imfilter(hp, fspecial('gaussian', 13, 3.75));
hp_sub2_aa = hp_filtered(1:2:end, 1:2:end);
hp_sub4_aa = hp_filtered_more(1:4:end, 1:4:end);

% Save the subsampled images
imwrite(hp_sub2_aa, 'assets/HP-sub2-aa.png', 'PNG');
imwrite(hp_sub4_aa, 'assets/HP-sub4-aa.png', 'PNG');

% Compute the freq. domain versions of them
hp_sub2_freq_aa = fftshift(abs(fft2(hp_sub2_aa))/50);
hp_sub4_freq_aa = fftshift(abs(fft2(hp_sub4_aa)))/50;

% Save the freq. domain versions of them
imwrite(hp_sub2_freq_aa, 'assets/HP-sub2-aa-freq.png', 'PNG');
imwrite(hp_sub4_freq_aa, 'assets/HP-sub4-aa-freq.png', 'PNG');

% ** Part 5 **
% Apply Canny edge detection and get the default parameters computed by 
% Matlab to HP
[cannyedge_hp, thresh_hp] = edge(hp, 'canny');
imwrite(edge(hp, 'canny', [0.05, 0.13]), 'assets/HP-canny-optimal.png', 'PNG');
imwrite(edge(hp, 'canny', [0.001, 0.002]), 'assets/HP-canny-lowlow.png', 'PNG');
imwrite(edge(hp, 'canny', [0.09, 0.091]), 'assets/HP-canny-highlow.png', 'PNG');
imwrite(edge(hp, 'canny', [0.001, 0.3]), 'assets/HP-canny-lowhigh.png', 'PNG');
imwrite(edge(hp, 'canny', [0.19, 0.3]), 'assets/HP-canny-highhigh.png', 'PNG');

% Apply Canny edge detection and get the default parameters computed by 
% Matlab to LP
[cannyedge_lp, thresh_lp] = edge(hp, 'canny');
imwrite(edge(lp, 'canny', [0.02, 0.11]), 'assets/LP-canny-optimal.png', 'PNG');
imwrite(edge(lp, 'canny', [0.001, 0.002]), 'assets/LP-canny-lowlow.png', 'PNG');
imwrite(edge(lp, 'canny', [0.08, 0.081]), 'assets/LP-canny-highlow.png', 'PNG');
imwrite(edge(lp, 'canny', [0.001, 0.3]), 'assets/LP-canny-lowhigh.png', 'PNG');
imwrite(edge(lp, 'canny', [0.2, 0.3]), 'assets/LP-canny-highhigh.png', 'PNG');












