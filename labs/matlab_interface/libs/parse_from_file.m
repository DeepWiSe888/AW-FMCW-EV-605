function [timestamp,tx,rx,fn,adc] = parse_from_file(buf,cursor)
    sec = typecast(uint8(buf(cursor:cursor + 8 - 1)), 'uint64');
    cursor = cursor + 8;
    
    usec = typecast(uint8(buf(cursor:cursor + 8 - 1)), 'uint64');
    cursor = cursor + 8;
    timestamp = cast(sec,'double') + cast(usec,'double') / 1000000;
    
    rx = typecast(uint8(buf(cursor:cursor + 4 - 1)), 'uint32');
    cursor = cursor + 4;

    tx = typecast(uint8(buf(cursor:cursor + 4 - 1)), 'uint32');
    cursor = cursor + 4;

    fn = typecast(uint8(buf(cursor:cursor + 4 - 1)), 'uint32');
    cursor = cursor + 4;

    adc = typecast(uint8(buf(cursor:end)), 'single');
end