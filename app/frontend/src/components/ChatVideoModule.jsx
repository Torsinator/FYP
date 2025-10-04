import React, { useState, useRef, useEffect } from "react";
import api from "../api";
import "../css/ChatVideoModule.css";

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
            const res = await api.post("/command", { message: chatInput });
            setResponse(res.data);
            setState(STATES.RESPONSE_RECEIVED);

            if (res.data.clarify) {
                setState(STATES.CLARIFY);
                return;
            }

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
        api.get("/reset");
    };

    useEffect(() => {
    // On first load (or refresh), tell backend to reset
        reset()
    }, []); // empty deps -> only runs on mount

    return (
        <div className="chat-video-container">
            <div className="chat-input-section">
                <textarea
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

            {state === STATES.WAITING_RESPONSE && (
                <div className="waiting">Waiting for trajectory...</div>
            )}

            {(state !== STATES.IDLE && state !== STATES.WAITING_RESPONSE) && response && (
                <div className="response-section">
                    <p><strong>Response:</strong></p>
                    <p><em>Reasoning:</em> {response.reasoning}</p>
                    <p><em>States:</em> {response.states}</p>
                    <p><em>Weights:</em> {response.weights}</p>
                </div>
            )}

            {state === STATES.TRAJECTORY_READY && videoUrl && (
                <div className="video-section">
                    <video ref={videoRef} src={videoUrl} controls width="640" height="360" />
                </div>
            )}

            <button className="reset-button" onClick={reset}>Reset</button>
        </div>
    );
}

export default ChatVideoModule;
