%{

This script performs speech activity detection (SAD) on audio files, and
writes the detected speech boundaries (in seconds) to a 3-column
space-delimited ASCII text file. It also checks for issues such as files
with only digital zeros or currupted files.

We use matlab parpool along with parfor to speed up the SAD process through
parallelization.

Author: Omid Sadjadi Email:  omid.sadjadi@nist.gov

%}

function gen_sad_marks(list, num_workers)
% runs the nist_segmenter in parallel on audio files listed in "list", and
% saves the SAD labels in "outdir". You may specify the number of parallel
% workers as well.

if nargin<2, num_workers = feature('numcores') - 2; end

if ischar(num_workers), num_workers = str2double(num_workers); end

parpool('local', num_workers)

fid = fopen(list, 'rt');
C = textscan(fid, '%s %s %s', 'Delimiter', ',');
fclose(fid);

infilenames  = C{1};
channels = C{2};
outfilenames = C{3};


nfiles = length(infilenames);
parts = 100; % modify this based on your resources
nbatch = floor( nfiles/parts + 0.99999 );
for batch = 1 : nbatch
    start = 1 + ( batch - 1 ) * parts;
    fin = min(batch * parts, nfiles);
    %     len = fin - start + 1;
    index = start : fin;
    infilenames_b = infilenames(index);
    channels_b = channels(index);
    outfilenames_b = outfilenames(index);
    parfor ix = 1 : length(infilenames_b)
        % [~,basename,~] = fileparts(infilenames_b{ix});
        % sadFilename = fullfile(outdir, [pathstr,'/',basename, '.sad']);
        sadFilename = outfilenames_b{ix}
        % here, we assume all audio files are single channel, otherwise you
        % should also specify the side (i.e., 'a' or 'b')
        [~, ~, speech_endpts, ~] = nist_segmenter(infilenames_b{ix}, channels_b{ix});
        path = fileparts(sadFilename);
        if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
        fid = fopen(sadFilename, 'wt');
        for jx = 1 : size(speech_endpts, 1)
            fprintf(fid, '%s %.2f %.2f\n', infilenames_b{ix}, ... 
                    speech_endpts(jx, 1)/100, speech_endpts(jx, 2)/100);
        end
        fclose(fid);
    end
end
