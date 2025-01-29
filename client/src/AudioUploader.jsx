import { useState } from "react";
import axios from "axios";

export default function AudioUploader() {
  const [file, setFile] = useState(null);
  const [emotion, setEmotion] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an audio file");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8000/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setEmotion(response.data.emotion);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <input type="file" accept="audio/*" onChange={handleFileChange} />
      <button className="bg-blue-500 text-white px-4 py-2 mt-2" onClick={handleUpload}>
        Upload & Predict
      </button>
      {emotion && <p className="mt-4 text-lg font-bold">Predicted Emotion: {emotion}</p>}
    </div>
  );
}
