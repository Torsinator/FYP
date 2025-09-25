import React, { useState, useRef } from "react";
import api from "../api";

// Define the states
const STATES = {
  IDLE: "idle",
  WAITING_RESPONSE: "waiting_response",
  RESPONSE_RECEIVED: "response_received",
  CLARIFY: "clarify_needed",
  TRAJECTORY_READY: "video_ready",
};

function ChatVideoModule() {
    const [state, setState] = useState(STATES.IDLE);
    const [chatInput, setChatInput] = useState("");
    const [response, setResponse] = useState(null);
    const [videoUrl, setVideoUrl] = useState(null);

    const videoRef = useRef();

    const sendChat = async () => {
    if (!chatInput || (state !== STATES.IDLE && state !== STATES.CLARIFY)) return;
    setState(STATES.WAITING_RESPONSE);

    try {
        // Send chat to backend
        const res = await api.post("/command", { message: chatInput });

        if (res.data.clarify) {
        setState(STATES.CLARIFY);
        return;
        }

        setResponse(res.data);
        setState(STATES.RESPONSE_RECEIVED);

        // Fetch the generated video
        const videoRes = await api.get(`/video/${res.data.video_path}`, {
            responseType: "blob",
        });
        const url = URL.createObjectURL(videoRes.data);
        setVideoUrl(url);
        setState(STATES.TRAJECTORY_READY);
    } catch (err) {
        console.error(err);
        setState(STATES.IDLE);
    }
    };


    const reset = () => {
        if (videoUrl) URL.revokeObjectURL(videoUrl);
        setState(STATES.IDLE);
        setChatInput("");
        setResponse(null);
        setVideoUrl(null);
    };

  return (
    <div>
      {/* Chat input */}
      {(state === STATES.IDLE || state === STATES.WAITING_RESPONSE || state === STATES.CLARIFY) && (
        <div>
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") sendChat();
            }}
            disabled={state !== STATES.IDLE && state !== STATES.CLARIFY}
          />
          <button onClick={sendChat} disabled={state !== STATES.IDLE && state !== STATES.CLARIFY}>
            Send
          </button>
        </div>
      )}

        {/* Backend response */}
      {(state === STATES.WAITING_RESPONSE) && (
        <div>
          <p><strong>Waiting for trajectory</strong></p>
        </div>
      )}

      {/* Backend response */}
      {(state !== STATES.IDLE && state !== STATES.WAITING_RESPONSE) && (
        <div>
          <p><strong>Response:</strong></p>
          <p>Reasoning: {response.reasoning}</p>
          <p>States: {response.states}</p>
          <p>Weights: {response.weights}</p>
        </div>
      )}

      {/* Video */}
      {state === STATES.TRAJECTORY_READY && videoUrl && (
        <div>
          <video ref={videoRef} src={videoUrl} controls width="640" height="360" />
        </div>
      )}

      {/* Reset button */}
        <button onClick={reset}>Reset</button>
    </div>
  );
}

export default ChatVideoModule;
