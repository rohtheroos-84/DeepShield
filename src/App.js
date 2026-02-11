import React, { useState, useEffect, useCallback, useRef } from "react";
import Particles from "./components/Particles";
import DecryptedText from "./components/DecryptedText";
import RotatingText from "./components/RotatingText";
import SplitText from "./components/SplitText";
import LiveBar from "./components/LiveBar";
import ConstellationBackground from "./components/ConstellationBackground";
import "./App.css";

const teamMembers = [
  {
    name: "Yadeesh T",
    github: "https://github.com/Yadeesht",
    linkedin: "https://www.linkedin.com/in/yadeesh-t-259640288",
    email: "yadeesh005@gmail.com",
    image: "/dp.jpg",
  },
  {
    name: "Gokul Ram K",
    github: "https://github.com/GOKULRAM-K",
    linkedin: "https://www.linkedin.com/in/gokul-ram-k-277a6a308",
    email: "gokul.ram.kannan21@gmail.com",
    image: "/dp.jpg",
  },
  {
    name: "Rohit N",
    github: "https://github.com/rohtheroos-84",
    linkedin: "https://www.linkedin.com/in/rohit-n-1b0984280",
    email: "rohit84.official@gmail.com",
    image: "/dp.jpg",
  },
  {
    name: "Rahul B",
    github: "https://github.com/RahulB-24",
    linkedin: "https://www.linkedin.com/in/rahul-balachandar-a9436a293",
    email: "rahulbalachandar24@gmail.com",
    image: "/dp.jpg",
  },
];

const handleAnimationComplete = () => {
  console.log("All letters have animated!");
};

function App() {
  const [file, setFile] = useState(null);
  const [fileURL, setFileURL] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [processingComplete, setProcessingComplete] = useState(false);
  const [homeKey, setHomeKey] = useState(0);
  const resultRef = useRef(null);

  // Revoke old object URL when a new file is selected or component unmounts
  useEffect(() => {
    return () => {
      if (fileURL) {
        URL.revokeObjectURL(fileURL);
      }
    };
  }, [fileURL]);

  const handleSmoothScroll = useCallback((event) => {
    event.preventDefault();
    const targetId = event.currentTarget.getAttribute("href").substring(1);
    const targetElement = document.getElementById(targetId);
    if (targetElement) {
      window.scrollTo({
        top: targetElement.offsetTop - 60,
        behavior: "smooth",
      });
    }
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      const sections = document.querySelectorAll("section");
      const navLinks = document.querySelectorAll(".nav-links a");
      const scrollPos = window.scrollY + 80;
      sections.forEach((section) => {
        if (
          section.offsetTop <= scrollPos &&
          section.offsetTop + section.offsetHeight > scrollPos
        ) {
          navLinks.forEach((link) => {
            link.classList.remove("active");
            if (link.getAttribute("href").substring(1) === section.id) {
              link.classList.add("active");
            }
          });
        }
      });
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    const homeSection = document.getElementById("home");
    if (!homeSection) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setHomeKey((prev) => prev + 1);
        }
      },
      { threshold: 0.5 }
    );
    observer.observe(homeSection);
    return () => observer.disconnect();
  }, []);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      if (
        !selectedFile.type.startsWith("video/") &&
        !selectedFile.type.startsWith("image/")
      ) {
        alert("Invalid file type. Please upload a video or image.");
        return;
      }
      if (fileURL) {
        URL.revokeObjectURL(fileURL);
      }
      setFile(selectedFile);
      setFileURL(URL.createObjectURL(selectedFile));
      setResult(null);
      setProcessingComplete(false);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      if (
        !droppedFile.type.startsWith("video/") &&
        !droppedFile.type.startsWith("image/")
      ) {
        alert("Invalid file type. Please upload a video or image.");
        return;
      }
      if (fileURL) {
        URL.revokeObjectURL(fileURL);
      }
      setFile(droppedFile);
      setFileURL(URL.createObjectURL(droppedFile));
      setResult(null);
      setProcessingComplete(false);
    }
  };

  const handleDetect = async () => {
    if (!file) {
      alert("Please upload a video/image first.");
      return;
    }
    setLoading(true);
    setProcessingComplete(false);
    setResult(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let accumulated = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          setProcessingComplete(true);
          break;
        }
        accumulated += decoder.decode(value, { stream: true });
        const events = accumulated.split("\n\n");
        accumulated = events.pop();
        events.forEach((eventStr) => {
          if (eventStr.startsWith("data: ")) {
            const jsonStr = eventStr.replace("data: ", "");
            try {
              const data = JSON.parse(jsonStr);
              setResult(data);
            } catch (e) {
              console.warn("Failed to parse SSE data:", e);
            }
          }
        });
      }
    } catch (error) {
      console.error("Error during detection:", error);
      alert("Error during detection. Make sure the backend server is running.");
    }
    setLoading(false);
  };

  const finalVerdict =
    processingComplete && result
      ? result.real_count > result.fake_count
        ? "Real Content"
        : "Deepfake Detected"
      : "";

  const verdictBgColor =
    finalVerdict === "Real Content"
      ? "#00c853"
      : finalVerdict === "Deepfake Detected"
      ? "#ff3d00"
      : "transparent";

  return (
    <div className="app">
      <ConstellationBackground />
      <nav className="navbar" id="main-nav">
        <div className="logo">
          <span className="logo-static">Deep</span>
          <RotatingText
            texts={["Shield", "Fake?"]}
            mainClassName="rotating-text"
            staggerFrom="last"
            initial={{ y: "100%", opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: "-120%", opacity: 0 }}
            staggerDuration={0.025}
            transition={{ type: "spring", damping: 30, stiffness: 400 }}
            rotationInterval={2000}
          />
        </div>
        <ul className="nav-links">
          <li>
            <a href="#home" onClick={handleSmoothScroll}>
              Home
            </a>
          </li>
          <li>
            <a href="#architecture" onClick={handleSmoothScroll}>
              WorkFlow
            </a>
          </li>
          <li>
            <a href="#about" onClick={handleSmoothScroll}>
              About Us
            </a>
          </li>
        </ul>
      </nav>
      <br /> <br /> <br />
      <section id="home" className="container">
        <div className="project-title">
          <span className="title-deep">Deep</span>
          <span className="title-shield">Shield</span>
        </div>
        <div className="tagline-container">
          <SplitText
            key={homeKey}
            text="Exposing the Truth, One Frame at a Time."
            className="tagline"
            delay={20}
            animationFrom={{ opacity: 0, transform: "translate3d(0,20px,0)" }}
            animationTo={{
              opacity: 1,
              transform: "translate3d(0,0,0)",
              color: "white",
            }}
            easing="easeOutCubic"
            threshold={0.2}
            rootMargin="-20px"
            onLetterAnimationComplete={handleAnimationComplete}
          />
        </div>
        <br /> <br />
        <div className="upload-box" onDragOver={handleDragOver} onDrop={handleDrop}>
          <input
            type="file"
            accept="video/*,image/*"
            onChange={handleFileChange}
            disabled={loading}
          />
        </div>
        {fileURL && (
          <div className="media-preview">
            {file.type.startsWith("video/") ? (
              <video src={fileURL} controls autoPlay loop className="media" />
            ) : (
              <img src={fileURL} alt="Uploaded Preview" className="media" />
            )}
          </div>
        )}
        <button className="detect-btn" onClick={handleDetect} disabled={loading}>
          {loading ? "Analyzing..." : "Detect Deepfake"}
        </button>
        {result && (
          <div className="result" ref={resultRef}>
            <pre>
              {`Total Frames: ${result.total_frames}
Real Count: ${result.real_count}
Fake Count: ${result.fake_count}`}
            </pre>
            <LiveBar
              realPercentage={result.real_percentage}
              fakePercentage={result.fake_percentage}
            />
            {processingComplete && (
              <div
                className="final-verdict"
                style={{
                  backgroundColor: verdictBgColor,
                  padding: "10px",
                  borderRadius: "5px",
                  marginTop: "10px",
                  color: "#fff",
                  fontWeight: "bold",
                }}
              >
                {finalVerdict}
              </div>
            )}
          </div>
        )}
      </section>
      <br /> <br /> <br />
      <section id="architecture">
        <h2>WorkFlow</h2>
        <div className="Arch_div">
          <img src="/arch.png" alt="System Architecture" className="arch-img" />
        </div>
      </section>

      <section id="about">
        <h2>About Us</h2>
        <br /> <br />
        <div className="cards-container">
          {teamMembers.map((member, index) => (
            <div key={index} className="card">
              <img src={member.image} alt="Profile" className="profile-pic" />
              <h3>
                <br />
                <DecryptedText
                  text={member.name}
                  animateOn="view"
                  revealDirection="start"
                  style={{ fontSize: "24px", color: "white" }}
                />
              </h3>
              <br />
              <p>
                <a href={member.github} target="_blank" rel="noopener noreferrer">
                  <DecryptedText text="GitHub" animateOn="view" revealDirection="start" />
                </a>{" "}
                |{" "}
                <a href={member.linkedin} target="_blank" rel="noopener noreferrer">
                  <DecryptedText text="LinkedIn" animateOn="view" revealDirection="start" />
                </a>
              </p>
              <p className="email">
                <DecryptedText
                  text={member.email}
                  animateOn="view"
                  revealDirection="start"
                />
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;
