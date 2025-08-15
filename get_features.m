function features=get_features(file_input,lead_names_target)



% header_tmp=header{1};
% header_tmp=strsplit(header_tmp,' ');
% fs=str2double(header_tmp{3});
%
% signals = read_challenge_signals(file,header);


window_dur = 60;    % Process in 1-minminute windows
start_time = 0;     % Start from 1 second
stop_time = 0;      % Process until the end
flatten_flag = 1;   % Stack features by channel
csv_path = './';

features =  process_ecg_wfdb(file_input, csv_path, lead_names_target, window_dur, start_time, stop_time, flatten_flag);



end

