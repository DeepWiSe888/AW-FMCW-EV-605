function framelist = read_frame(file_name)
    framelist = [];

    fid = fopen(file_name);
    if (fid < 0)
        return;
    end
    
    framelist.timestamp = 0;
    framelist.frameno = 0;
    framelist.txno = 0;
    framelist.rxno = 0;
    framelist.adc = zeros(990,1);
    
    headflag = 'wirush-vpas:';
    recv_buffer = [];
    cnt = 1;
    while ~feof(fid)
        tmp_pack = fread(fid,4096,'uint8');
        recv_buffer = [recv_buffer;tmp_pack];
        while 1
            start_idx = find_head(recv_buffer,headflag);
            if start_idx <= 0
                break;
            end
            end_idx = find_head(recv_buffer(start_idx + length(headflag):end),headflag);
            if end_idx <= 0
                break;
            end
            buf = recv_buffer(start_idx:end_idx + length(headflag) - 1);
            recv_buffer = recv_buffer(end_idx + length(headflag):end);

            cursor = length(headflag)+ 1;
            [timestamp,tx,rx,fn,adc] = parse_from_file(buf,cursor);

            if isempty(timestamp) || isempty(tx) || isempty(rx)|| isempty(fn)|| isempty(adc)
                break;
            end

            framelist(cnt).timestamp = timestamp;
            framelist(cnt).frameno = fn;
            framelist(cnt).txno = tx;
            framelist(cnt).rxno = rx;
            framelist(cnt).adc = adc;

            cnt = cnt + 1; 
        end
    end
    fclose(fid);
end