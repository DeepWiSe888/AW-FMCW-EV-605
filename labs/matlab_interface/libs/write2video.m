function write2video(filename,frames,fps)
    % create the video writer with 1 fps
    writerObj = VideoWriter(filename,'MPEG-4');
    writerObj.FrameRate = fps;
    % set the seconds per image
    % open the video writer
    open(writerObj);
    % write the frames to the video
    [F_len,~,~,~] = size(frames);
    for i=1:F_len
        % convert the image to a frame
        frame = squeeze(frames(i,:,:,:));    
        writeVideo(writerObj, frame);
    end
    % close the writer object
    close(writerObj);
end