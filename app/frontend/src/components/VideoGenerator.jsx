import React, { useEffect, useRef, useState } from "react";

function VideoGenerator() {
  const videoRef = useRef();
  const [ws, setWs] = useState(null);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/ws/generate");

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.video_path && videoRef.current) {
        videoRef.current.src = "http://localhost:8000" + data.video_path;
        videoRef.current.play();
      }
    };

    setWs(socket);
    return () => socket.close();
  }, []);

  return (
    <div>
      <video ref={videoRef} controls width="640" height="360" />
    </div>
  );
}

export default VideoGenerator;