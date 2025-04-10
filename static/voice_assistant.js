function startSession(patientName) {
    fetch("/start_session", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "patient_name=" + encodeURIComponent(patientName),
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
        document.getElementById("status").innerText = data.message; // Display on screen
    });
}

function enterPatientName() {
    let patientName = prompt("Enter Patient Name:");
    if (patientName) {
        startSession(patientName);
    } else {
        recognizeVoice(); // If no input, use voice
    }
}

function recognizeVoice() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US"; // Set language
    recognition.start();

    document.getElementById("status").innerText = "Listening... 🎤";

    recognition.onresult = function(event) {
        let patientName = event.results[0][0].transcript; // Get speech text
        document.getElementById("status").innerText = "Recognized: " + patientName;
        startSession(patientName); // Start session
    };

    recognition.onerror = function() {
        document.getElementById("status").innerText = "Error recognizing speech!";
    };
}
