import { useState, useRef } from "react";
import axios from "axios";
import { FaMicrophone, FaPaperclip, FaPaperPlane } from "react-icons/fa";

export default function AudioEmotionAnalyzer() {
  const [file, setFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [emotion, setEmotion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        try {
          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
          setFile(new File([audioBlob], "recorded_audio.wav", { type: "audio/wav" }));
        } catch (error) {
          console.error("Error in recording stop event:", error);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type.startsWith("audio/")) {
      setFile(selectedFile);
    } else {
      alert("Please select a valid audio file.");
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select or record an audio file");

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    console.log(file);
    try {
      const response = await axios.post("http://localhost:8000/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log(response);
      setEmotion(response.data.emotion);
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Error analyzing emotion. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-sky-100 p-8">
      <h1 className="text-3xl font-bold m-[-20px] text-gray-800"><img className="w-42 h-42" src="/LogoB.png" alt="" /></h1>
      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-md flex flex-col space-y-4">
        <div className="flex items-center bg-gray-100 rounded-lg p-2 space-x-3">
          <label htmlFor="audio-file" className="p-3 cursor-pointer text-gray-600 hover:text-gray-800">
            <FaPaperclip size={24} />
          </label>
          <input type="file" accept="audio/*" onChange={handleFileChange} className="hidden" id="audio-file" />
          <div className="flex-grow bg-white rounded-lg p-2 text-gray-600 border border-gray-300">
            {file ? file.name : "Record or upload an audio file..."}
          </div>
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`p-3 rounded-full transition duration-300 ${isRecording ? "bg-red-500 text-white" : "text-red-500 hover:bg-red-100"}`}
          >
            <FaMicrophone size={24} />
          </button>
          <button
            onClick={handleUpload}
            disabled={!file || isLoading}
            className="p-3 rounded-full text-blue-500 hover:bg-blue-100 disabled:opacity-50 transition duration-300"
          >
            <FaPaperPlane size={24} />
          </button>
        </div>
        {emotion && (
          <div className="mt-4 text-center">
            <h2 className="text-xl font-semibold text-gray-700">Predicted Emotion:</h2>
            <p className="text-2xl font-bold text-blue-500">{emotion}</p>
          </div>
        )}
      </div>
    </div>
  );
}
