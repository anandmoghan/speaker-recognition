function [tot_dur, sp_dur, endpts, astr] = nist_segmenter(speechFilename, channel)
% runs Sohn's statistical model based VAD as implemented in Voicebox tookit
% 
% Author: Omid Sadjadi
% Email:  omid.sadjadi@nist.gov

if nargin < 2, channel = 'a'; end

[~,~,ext] = fileparts(speechFilename);

if strcmp(ext,'.sph')
    [s, fs] = readsph(speechFilename);
elseif strcmp(ext,'.flac')
    [s, fs] = flacread2(speechFilename);
else
    try 
        [s, fs] = audioread(speechFilename);
    catch
        error('Unknown file extension for %s\n', speechFilename);
    end
end

if fs ~= 8000
    s = resample(s, 8000, fs);
    fs = 8000;
end

astr = 'OK';
endpts = [];

if fs ~= 8000
    fprintf(2, 'FS: oh dear! something fishy about the sampling frequency of this file: %s\n', speechFilename);
    astr = sprintf('FS');
    tot_dur = 0;
    sp_dur = 0;
    return
end

achan = 1;
if channel=='b'
    achan = 2;
end
s = s(:, achan);

tot_dur = length(s) / fs; 

if ( max(s) - mean(s) ) < 1e-6
    fprintf(2, 'LEVEL: oh dear! something fishy about the content of this file: %s\n', speechFilename);
    astr = sprintf('LEVEL');
    sp_dur = 0;
    return 
end

s = add_dither(s, fs);

[sad, endpts] = voice_act_detect(s, fs);

sp_dur = sum(sad);
if sp_dur == 0
    fprintf(2, 'SAD: oh dear! something fishy about the content of this file: %s\n', speechFilename);
    astr = sprintf('SAD');
%     sad = true(3, 1);
end


function [sad, endpoints] = voice_act_detect(s, fs)

% min_sil_dur = 50;
% min_sp_dur = 50;

min_sil_dur = 90;
min_sp_dur = 50;

pad_len_be = 10;
pad_len_en = 10;

[b, a] = butter(4, [300, 3400]/fs);
s = filter(b, a, s);
s_vad = my_awgn(s, 20);  % masking by white Gaussian noise
pp.pr = 0.98;
pp.ne = 1;
vs = vadsohn(s_vad, fs, 't', pp);  % make sure voicebox is in the path
sad = vs(:, 3) == 1;
[sad, endpoints] = smooth_n_extend_sad(sad, min_sil_dur, min_sp_dur, pad_len_be, pad_len_en);


function s_plus_n = my_awgn(signal, snr)
% adds WGN to signal at specified SNR. This helps mask cross-talks.
rng(7);
s_std = std(signal);
n_std = s_std / 10^(snr / 20);
noise = n_std * randn(size(signal));
s_plus_n = signal + noise;


function [sad, endpoints] = smooth_n_extend_sad(sad, min_sil_dur, min_sp_dur, pad_len_be, pad_len_en)
% smooths and extends detected speech segments
sad_len = length(sad);
sad_aux = [0; sad; 0];
endpoints = sad2endpoints(sad_aux);
endpoints = endpoints(diff(endpoints, 1, 2) >= 3, :);
sad = endpoints2sad(endpoints, sad_len);
sad_aux = [0; 1 - sad; 0];
endpoints = sad2endpoints(sad_aux);
endpoints = endpoints(diff(endpoints, 1, 2) >= min_sil_dur, :);
sad = ~endpoints2sad(endpoints, sad_len);
sad_aux = [0; sad; 0];
endpoints = sad2endpoints(sad_aux);
endpoints = endpoints(diff(endpoints, 1, 2) >= min_sp_dur, :);
endpoints(:, 1) = max(endpoints(:, 1) - pad_len_be, 1);
endpoints(:, 2) = min(endpoints(:, 2) + pad_len_en, sad_len);
sad = endpoints2sad(endpoints, sad_len);


function endpoints = sad2endpoints(sad)
% converts binary SAD array to frame-level endpoints 
sad_diff = find(sad(2:end) - sad(1:end-1));
endpoints = [sad_diff(1:2:end), sad_diff(2:2:end)-1];


function sad = endpoints2sad(endp, sad_len)
% converts frame-level endpoints to binary SAD array
sad = false(sad_len, 1);
for ix = 1 : size(endp, 1)
    sad(endp(ix, 1):endp(ix, 2)) = true;
end


function s = add_dither(s, fs)
% removes DC component of the signal and add a small dither
rng(7);
if fs == 16e3
    alpha = 0.99;
elseif fs == 8e3 
    alpha = 0.999;
else
    error('only 8 and 16 kHz data are supported!');
end
s = filter([1 -1], [1 -alpha], s); % remove DC
dither = rand(size(s)) + rand(size(s)) - 1; 
s = s + 1e-6 * std(s) * dither;