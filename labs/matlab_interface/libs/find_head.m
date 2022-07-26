function [pos]= find_head(buf,headflag)
    headflag = headflag';
    pos = 0;
    for i=1:length(buf)-length(headflag)
        something = char(buf(i:i+length(headflag) - 1));
        if(strcmp(something,headflag))
            pos = i;
            return;
        end
    end
end